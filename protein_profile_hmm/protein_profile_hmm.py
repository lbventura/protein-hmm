"""Glocal profile Hidden Markov Model for protein sequence comparison.

This is a simple re-implementation of HMMER's core
profile-HMM idea for educational purposes: a position-specific HMM with Match/Insert/Delete states per
consensus column, trained from a multiple sequence alignment, and used to
score query sequences against the family.

All emission and transition probabilities are stored in log2 space. Scoring
returns complete-path log-odds bit scores relative to the product of background
residue frequencies conditional on query length. Exact zero probabilities
are represented by negative infinity; impossible queries have score -inf.

The allowed transitions are:

- M : Match state; a sequence residue is aligned to consensus position k.
- I : Insert state; an extra sequence residue occurs between consensus positions k and k+1.
- D: Delete state; consensus position k is skipped, with no corresponding sequence residue.
- B: Begin state; a silent entry state into the profile consensus.
- End: Silent absorbing destination after the complete query has been consumed.

Transition layout (slot index k = 0..K):

    index   meaning
      0     M_k -> M_{k+1}    (T_MM)
      1     M_k -> I_k        (T_MI)
      2     M_k -> D_{k+1}    (T_MD)
      3     I_k -> M_{k+1}    (T_IM)
      4     I_k -> I_k        (T_II)
      5     D_k -> M_{k+1}    (T_DM)
      6     D_k -> D_{k+1}    (T_DD)
      7     B   -> M_1        (T_BM, only at k=0)
      8     B   -> D_1        (T_BD, only at k=0)
      9     M_K -> End        (T_ME, only at k=K)
     10     I_K -> End        (T_IE, only at k=K)
     11     D_K -> End        (T_DE, only at k=K)

At slot zero T_MI denotes B -> I0 and competes with T_BM and T_BD.
Terminal Match and Insert states choose between insertion and End; D_K
exits with probability one. End is silent and has no emission or DP row.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import numpy.typing as npt

from .amino_acids import BACKGROUND_FREQ, N_AMINO_ACIDS, alignment_width, encode, encode_rows
from .msa import MSA

NEG_INF: float = -np.inf
T_MM = 0
T_MI = 1
T_MD = 2
T_IM = 3
T_II = 4
T_DM = 5
T_DD = 6
T_BM = 7
T_BD = 8
T_ME = 9
T_IE = 10
T_DE = 11
N_TRANS = 12

# Limit event/index temporaries to one batch; a single wider row is still allowed.
_MAX_BATCH_CELLS = 250_000
# Source rows and destination columns are B, M, I, D; edges into B are excluded.
_TRANSITION_TYPES = np.array(
    [
        [-1, T_BM, T_MI, T_BD],
        [-1, T_MM, T_MI, T_MD],
        [-1, T_IM, T_II, -1],  # I->D must be repaired before counting.
        [-1, T_DM, -1, T_DD],  # D->I must be repaired before counting.
    ],
    dtype=np.intp,
)


class State(IntEnum):
    """Integer event codes in the transition lookup's row/column order."""

    BEGIN = 0
    MATCH = 1
    INSERT = 2
    DELETE = 3


def log2sumexp2(values: Iterable[float]) -> float:
    """log2(sum(2**v for v in values)), numerically stable."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return NEG_INF
    m = float(arr.max())
    if m == NEG_INF:
        return NEG_INF
    return m + float(np.log2(np.sum(2.0 ** (arr - m))))


@dataclass
class AlignmentTrace:
    """Ordered training paths, stored as parallel arrays with one entry per event.

    Each sequence starts with Begin at position 0. Match/Delete positions are
    1..K; Insert positions are 0..K. Only Match and Insert emit a residue;
    Begin and Delete carry -1. Insert-column gaps produce no event.
    Sequence indices refer to rows within the current batch. Raw alignment
    traces can contain adjacent I/D events; repair them before counting.
    """

    sequence: npt.NDArray[np.intp]
    state: npt.NDArray[np.int8]
    position: npt.NDArray[np.intp]
    residue: npt.NDArray[np.int8]


def _trace_alignment(
    encoded: npt.NDArray[np.int8], is_match: npt.NDArray[np.bool_]
) -> AlignmentTrace:
    """Turn alignment cells into raw B/M/I/D paths for a batch of rows.

    Preserve column assignments here; _repair_trace resolves adjacent I/D
    events before emission and transition counting.

    Example:

        encoded = encode_rows(["AGGC", "A..C", "A..."])
        is_match = np.array([True, False, False, True])
        trace = _trace_alignment(encoded, is_match)

    The alignment rows produce these paths, with one event per visited state:

        Row 1: AGGC -> B -> M1(A) -> I1(G) -> I1(G) -> M2(C)
        Row 2: A..C -> B -> M1(A) --------------------> M2(C)
        Row 3: A... -> B -> M1(A) --------------------> D2

    The returned AlignmentTrace stores these events in four parallel arrays.
    The first five entries describe Row 1's path, the next three describe
    Row 2's path, and the final three describe Row 3's path:

        sequence: [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
        state:    [0, 1, 2, 2, 1, 0, 1, 1, 0, 1, 3]
        position: [0, 1, 1, 1, 2, 0, 1, 2, 0, 1, 2]
        residue:  [-1, 0, 5, 5, 1, -1, 0, 1, -1, 0, -1]

    At any index, sequence identifies the input row (0, 1, or 2); state
    records B=0, M=1, I=2, or D=3; position records the number after M/I/D
    in the path, or 0 for B. The residue is its AA_INDEX code (A=0, G=5,
    C=1), or -1 for the non-emitting B and D states.

    For example, Row 1's first I1(G) is at index 2: reading that index
    across the arrays gives (sequence, state, position, residue) =
    (0, 2, 1, 5). The second I1(G) gives the same values at index 3.
    Insert-column gaps in Rows 2 and 3 produce no events. Row 3's final
    D2 is the last entry, (2, 3, 2, -1): it skips match position 2 without
    emitting a residue.
    """
    n_sequences, n_columns = encoded.shape
    has_residue = encoded >= 0
    states = np.empty((n_sequences, n_columns + 1), dtype=np.int8)
    states[:, 0] = State.BEGIN
    # A residue visits M or I; a gap visits D or produces no event (-1).
    residue_states = np.where(is_match, State.MATCH, State.INSERT).astype(np.int8)
    gap_states = np.where(is_match, State.DELETE, -1).astype(np.int8)
    states[:, 1:] = np.where(has_residue, residue_states, gap_states)

    residues = np.full_like(states, -1)
    residues[:, 1:] = encoded
    # In M1 I1 I1 M2, cumulative match counts give positions 1, 1, 1, 2.
    positions = np.concatenate(([0], np.cumsum(is_match, dtype=np.intp)))

    # nonzero visits each row in alignment order and omits insert-column gaps.
    sequence, column = np.nonzero(states >= 0)
    return AlignmentTrace(
        sequence=sequence,
        state=states[sequence, column],
        position=positions[column],
        residue=residues[sequence, column],
    )


def _repair_trace(trace: AlignmentTrace) -> AlignmentTrace:
    """Collapse adjacent I/D or D/I events into Match, from left to right.

    Each pair keeps the Delete position and Insert residue. For example,
    D2 -> I2(G) -> D3 becomes M2(G) -> D3. I1(G) -> I1(W) -> D2
    instead becomes I1(G) -> M2(W). Residue order and K are preserved.

    In each group of overlapping pairs, merge the first pair and every other
    pair after it into a Match.

    For example, take one alignment row with these fixed column assignments
    (M denotes a match column, I an insert column, and '.' a gap):

        row:     . G . W A C G . Y E
        column:  M I M I M M I M I M

        trace = _trace_alignment(
            encode_rows([".G.WACG.YE"]),
            np.array([True, False, True, False, True, True, False, True, False, True]),
        )

    Indices below refer to the original trace arrays, including Begin at index 0:

        index:  0  1   2      3   4      5      6      7      8   9      10
        event:  B  D1  I1(G)  D2  I2(W)  M3(A)  M4(C)  I4(G)  D5  I5(Y)  M6(E)

    A candidate's left index i identifies the pair of events (i, i+1).
    Candidates 1, 2, 3 overlap in the first D/I/D/I group; candidates 7, 8
    overlap in the later I/D/I group, separated by the two Match events:

        candidate_left_indices: [1, 2, 3, 7, 8]
        repair_group_starts:    [T, F, F, T, F]
        group_first_indices:   [1, 1, 1, 7, 7]
        offsets within group:  [0, 1, 2, 0, 1]

    Even offsets select repair_left_indices = [1, 3, 7], giving disjoint
    pairs (1, 2), (3, 4), and (7, 8). Selection restarts at index 7.
    The chosen pairs are transformed as follows:

        D1 I1(G) -> M1(G)
        D2 I2(W) -> M2(W)
        I4(G) D5 -> M5(G)

    where

        insert_indices: [2, 4, 7]  # with the residues G, W, G, respectively
        delete_indices: [1, 3, 8]  # Their profile positions are 1, 2, 5.
        repaired path:  B -> M1(G) -> M2(W) -> M3(A) -> M4(C)
                         -> M5(G) -> I5(Y) -> M6(E)

    Each Delete becomes Match and each paired Insert is removed. The emitted
    sequence remains GWACGYE and all six profile positions remain represented.
    """
    is_insert = trace.state == State.INSERT
    is_delete = trace.state == State.DELETE
    # Each index identifies the left event of a pair; its right event is at index + 1.
    candidate_left_indices = np.flatnonzero(
        (is_insert[:-1] & is_delete[1:]) | (is_delete[:-1] & is_insert[1:])
    )
    pairs_in_same_sequence = (
        trace.sequence[candidate_left_indices] == trace.sequence[candidate_left_indices + 1]
    )
    candidate_left_indices = candidate_left_indices[pairs_in_same_sequence]
    if candidate_left_indices.size == 0:
        return trace

    repair_group_starts = np.empty(candidate_left_indices.size, dtype=bool)
    repair_group_starts[0] = True
    repair_group_starts[1:] = np.diff(candidate_left_indices) > 1
    group_first_indices = np.maximum.accumulate(
        np.where(repair_group_starts, candidate_left_indices, 0)
    )
    # Restart leftmost, non-overlapping selection in each repair group.
    repair_left_indices = candidate_left_indices[
        (candidate_left_indices - group_first_indices) % 2 == 0
    ]
    insert_indices = np.where(
        is_insert[repair_left_indices], repair_left_indices, repair_left_indices + 1
    )
    delete_indices = np.where(
        is_delete[repair_left_indices], repair_left_indices, repair_left_indices + 1
    )

    # Keep the Delete's position and the Insert's residue, whichever event comes first.
    states = trace.state.copy()
    residues = trace.residue.copy()
    states[delete_indices] = State.MATCH
    residues[delete_indices] = trace.residue[insert_indices]
    keep = np.ones(trace.state.size, dtype=bool)
    keep[insert_indices] = False
    return AlignmentTrace(
        sequence=trace.sequence[keep],
        state=states[keep],
        position=trace.position[keep],
        residue=residues[keep],
    )


def _count_emissions(trace: AlignmentTrace, state: State, n_slots: int) -> npt.NDArray[np.int64]:
    """Count residues emitted by Match or Insert states, without priors.

    Match position k occupies emission row k-1; insertion slot k occupies
    row k. Repeated visits to an insert slot contribute separate observations.

    For example, select both columns of ["AC", "A."] as match positions:

        encoded = encode_rows(["AC", "A."])
        trace = _trace_alignment(encoded, np.array([True, True]))
        counts = _count_emissions(trace, State.MATCH, n_slots=2)

    Corresponding to the following paths:

        Row 1: AC -> B -> M1(A) -> M2(C)
        Row 2: A. -> B -> M1(A) -> D2

    The result has shape (2, 20), with only these nonzero entries:

        counts[0, AA_INDEX["A"]] == 2  # Both rows emit A at M1.
        counts[1, AA_INDEX["C"]] == 1  # Only Row 1 emits C at M2.

    B and D2 emit nothing. For the same trace,
    _count_emissions(trace, State.INSERT, n_slots=3) returns a (3, 20)
    array of zeros as neither path visits I0, I1, or I2.
    """
    emits = trace.state == state
    slots = trace.position[emits]
    if state == State.MATCH:
        slots = slots - 1
    residues = trace.residue[emits]
    counts = np.zeros((n_slots, N_AMINO_ACIDS), dtype=np.int64)
    # add.at accumulates repeated (slot, residue) pairs instead of overwriting.
    np.add.at(counts, (slots, residues), 1)
    return counts


def _count_transitions(trace: AlignmentTrace, n_slots: int) -> npt.NDArray[np.int64]:
    """Count adjacent transitions and one terminal exit per repaired training path.

    B->I0 uses T_MI. Unsupported edges, including unrepaired I/D pairs,
    raise ValueError. At least one match position is required. End transitions
    are counted from final events without adding End events to the trace.

    For example, select both columns of ["AC", "A."] as match positions:

        encoded = encode_rows(["AC", "A."])
        trace = _trace_alignment(encoded, np.array([True, True]))
        counts = _count_transitions(trace, n_slots=3)

    Corresponding to the following paths:

        Row 1: AC -> B -> M1(A) -> M2(C)
        Row 2: A. -> B -> M1(A) -> D2

    The result has shape (3, N_TRANS). Rows identify source positions
    0, 1, and 2; columns identify transition types. Five entries are nonzero:

        counts[0, T_BM] == 2  # B -> M1 in both rows.
        counts[1, T_MM] == 1  # M1 -> M2 in Row 1.
        counts[1, T_MD] == 1  # M1 -> D2 in Row 2.

        counts[2, T_ME] == 1  # M2 -> End in Row 1.
        counts[2, T_DE] == 1  # D2 -> End in Row 2.
    """
    if n_slots < 2:
        raise ValueError("Transition counting requires at least one match position")
    same_sequence = trace.sequence[:-1] == trace.sequence[1:]
    transition = _TRANSITION_TYPES[trace.state[:-1], trace.state[1:]]
    # Keep indices aligned with the trace, excluding pairs that cross sequence boundaries.
    invalid_edges = np.flatnonzero(same_sequence & (transition < 0))
    if invalid_edges.size:
        edge = int(invalid_edges[0])
        source = State(int(trace.state[edge])).name
        destination = State(int(trace.state[edge + 1])).name
        raise ValueError(
            f"Unsupported trace transition in sequence {trace.sequence[edge]}: "
            f"{source} at {trace.position[edge]} -> "
            f"{destination} at {trace.position[edge + 1]}; repair the trace before counting"
        )

    counts = np.zeros((n_slots, N_TRANS), dtype=np.int64)
    source_position = trace.position[:-1][same_sequence]
    np.add.at(counts, (source_position, transition[same_sequence]), 1)

    # Sequence boundaries and the final event identify one exit per repaired path.
    final_indices = np.flatnonzero(np.r_[~same_sequence, True])
    exit_types = np.array([-1, T_ME, T_IE, T_DE])[trace.state[final_indices]]
    if np.any(exit_types < 0):
        raise ValueError("Training paths must end at a Match, Insert, or Delete state")
    np.add.at(counts, (trace.position[final_indices], exit_types), 1)
    return counts


def _count_profile(
    encoded: npt.NDArray[np.int8], is_match: npt.NDArray[np.bool_]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Count match emissions, insert emissions, and transitions from one repaired trace.

    Repair raw I/D pairs once, so both emission and transition counts describe
    the same path. Repair can move insert observations into Match states;
    it preserves the globally selected match columns and their number K.

    K is the number of alignment columns selected as match positions. For
    example, with is_match = [True, False, False, True]:

        A  G  G  C
        A  .  .  C
        A  .  .  C
        M1 I1 I1 M2

    These four columns give K=2 match positions: M1 counts three A residues,
    and M2 counts three C residues. Both insert columns contribute to I1,
    which counts two G residues.

    Insertions can occur before, between, and after the match positions:

        I0  M1  I1  M2  I2

    Match emission counts have K rows (M1..MK); insert emission counts have
    K+1 rows (I0..IK). Each emission-count row has one column per amino acid.
    """
    trace = _repair_trace(_trace_alignment(encoded, is_match))
    K = int(is_match.sum())
    return (
        _count_emissions(trace, State.MATCH, K),
        _count_emissions(trace, State.INSERT, K + 1),
        _count_transitions(trace, K + 1),
    )


def _count_alignment(
    rows: list[str], symfrac: float
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Select global match columns, then collect raw counts in bounded batches.

    Example:

        match, insert, trans = _count_alignment(["AC", "A."], symfrac=0.5)

    Encoding gives [[0, 1], [0, -1]], where A=0, C=1, and the gap is -1.
    Counting residues in each column gives occupancy = [2, 1]. Dividing
    by the two input rows gives [1.0, 0.5]; both fractions meet the >= 0.5
    threshold, so is_match = [True, True] and K=2.

    The encoded alignment is reused in _count_profile(encoded, is_match),
    which counts these paths:

        Row 1: AC -> B -> M1(A) -> M2(C)
        Row 2: A. -> B -> M1(A) -> D2

    The returned arrays contain raw counts.
    Match counts have shape (2, 20), insert counts have
    shape (3, 20), and transition counts have shape (3, N_TRANS). Their
    only nonzero entries are:

        match[0, AA_INDEX["A"]] == 2  # Both rows emit A at M1.
        match[1, AA_INDEX["C"]] == 1  # Row 1 emits C at M2.
        trans[0, T_BM] == 2          # B -> M1 in both rows.
        trans[1, T_MM] == 1          # M1 -> M2 in Row 1.
        trans[1, T_MD] == 1          # M1 -> D2 in Row 2.
        trans[2, T_ME] == 1          # M2 -> End in Row 1.
        trans[2, T_DE] == 1          # D2 -> End in Row 2.

    All insert counts are zero because neither path visits an insert state.
    """
    if not rows:
        raise ValueError("Cannot build profile from empty MSA")
    width = alignment_width(rows)
    batch_rows = max(1, _MAX_BATCH_CELLS // max(width, 1))

    # Every batch must use the same column assignments, derived from all rows.
    encoded = None
    if len(rows) <= batch_rows:
        encoded = encode_rows(rows)
        occupancy = np.count_nonzero(encoded >= 0, axis=0)
    else:
        occupancy = np.zeros(width, dtype=np.int64)
        for start in range(0, len(rows), batch_rows):
            batch = encode_rows(rows[start : start + batch_rows])
            occupancy += np.count_nonzero(batch >= 0, axis=0)
    is_match = occupancy / len(rows) >= symfrac
    K = int(is_match.sum())
    if K == 0:
        raise ValueError("MSA has no match columns under symfrac")

    if encoded is not None:
        return _count_profile(encoded, is_match)

    match_counts = np.zeros((K, N_AMINO_ACIDS), dtype=np.int64)
    insert_counts = np.zeros((K + 1, N_AMINO_ACIDS), dtype=np.int64)
    trans_counts = np.zeros((K + 1, N_TRANS), dtype=np.int64)
    for start in range(0, len(rows), batch_rows):
        batch = encode_rows(rows[start : start + batch_rows])
        match, insert, trans = _count_profile(batch, is_match)
        match_counts += match
        insert_counts += insert
        trans_counts += trans
    return match_counts, insert_counts, trans_counts


def _normalise_rows(counts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Normalize rows, using a uniform distribution for wholly unobserved rows."""
    # Scale before summing so even large finite pseudocounts remain normalized.
    maxima = counts.max(axis=1, keepdims=True)
    scaled = np.divide(counts, maxima, out=np.zeros_like(counts), where=maxima > 0)
    sums = scaled.sum(axis=1, keepdims=True)
    probs = np.full_like(counts, 1.0 / counts.shape[1])
    return np.divide(scaled, sums, out=probs, where=sums > 0)


def _normalise_transitions(
    counts: npt.NDArray[np.float64], pseudocount: float = 0.0
) -> npt.NDArray[np.float64]:
    """Smooth and normalize legal outgoing alternatives; unused cells stay zero."""
    probs = np.zeros_like(counts, dtype=float)
    groups = (
        (slice(0, 1), [T_BM, T_MI, T_BD]),
        (slice(0, 1), [T_IM, T_II]),
        (slice(1, -1), [T_MM, T_MI, T_MD]),
        (slice(1, -1), [T_IM, T_II]),
        (slice(1, -1), [T_DM, T_DD]),
        (slice(-1, None), [T_MI, T_ME]),
        (slice(-1, None), [T_II, T_IE]),
        (slice(-1, None), [T_DE]),
    )
    for slots, transitions in groups:
        probs[slots, transitions] = _normalise_rows(counts[slots, transitions] + pseudocount)
    return probs


def _log_probabilities(probs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert exact zero probabilities to negative infinity without warnings."""
    logs = np.full_like(probs, NEG_INF)
    return np.log2(probs, out=logs, where=probs > 0)


@dataclass
class ProfileHMM:
    """A glocal profile HMM.

    The profile requires ``K >= 1`` Match columns (consensus positions). Conceptually
    there are also ``K`` Delete states (silent, one per Match column) and
    ``K + 1`` Insert states (one before column 1, one between each pair of
    consecutive match columns, one after column K).

    Begin chooses between M1, I0, and D1. Terminal M_K/I_K have explicit
    competing exits to silent End, while D_K exits with probability one.
    Direct construction must supply all 12 transition columns and exits.
    All log-probability arrays are in log2 space, with -inf for exact zeros.
    Bit scores use the ``BACKGROUND_FREQ`` residue product conditional on length.
    """

    K: int
    log_match_emit: npt.NDArray[np.float64]
    log_insert_emit: npt.NDArray[np.float64]
    log_trans: npt.NDArray[np.float64]
    log_bg: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.K < 1:
            raise ValueError("ProfileHMM requires at least one match position (K >= 1)")

    @classmethod
    def build_from_msa(
        cls,
        msa: MSA,
        symfrac: float = 0.5,
        pseudocount: float = 1.0,
    ) -> ProfileHMM:
        """Build a profile HMM from an MSA by counting + Laplace smoothing.

        Match columns are chosen by the symfrac rule; the remaining columns are
        treated as inserts attached to the nearest preceding match column.
        Each aligned sequence defines a training path. Adjacent I/D events
        are repaired into Match events before array operations count residues
        and transitions, including one exit per row, from the shared paths.
        Match emissions and legal outgoing alternatives get the same finite,
        nonnegative ``pseudocount`` per alternative; inserts get 20 background-
        weighted counts in total. At zero pseudocount, wholly unobserved groups
        are uniform and observed zero alternatives remain impossible. Unused
        transition cells stay zero. Large alignments use row batches with one
        global match mask. Probabilities are stored in log2 space, with -inf
        representing exact zeros.
        """
        if not np.isfinite(pseudocount) or pseudocount < 0:
            raise ValueError("pseudocount must be finite and nonnegative")
        match_counts, insert_counts, trans_counts = _count_alignment(msa.rows, symfrac)
        K = len(match_counts)

        # Add priors once, after combining all batches. Inserts retain their
        # fixed total of 20 background-weighted counts, independent of pseudocount.
        match_probs = _normalise_rows(match_counts.astype(float) + pseudocount)
        insert_probs = _normalise_rows(insert_counts + N_AMINO_ACIDS * BACKGROUND_FREQ)
        trans_probs = _normalise_transitions(trans_counts.astype(float), pseudocount)

        log_match_emit = _log_probabilities(match_probs)
        log_insert_emit = _log_probabilities(insert_probs)
        log_trans = _log_probabilities(trans_probs)
        log_bg = _log_probabilities(BACKGROUND_FREQ)

        return cls(
            K=K,
            log_match_emit=log_match_emit,
            log_insert_emit=log_insert_emit,
            log_trans=log_trans,
            log_bg=log_bg,
        )

    def _match_odds(self, k: int, aa: int) -> float:
        return float(self.log_match_emit[k, aa] - self.log_bg[aa])

    def _insert_odds(self, k: int, aa: int) -> float:
        return float(self.log_insert_emit[k, aa] - self.log_bg[aa])

    def _init_score_rows(
        self, seq: str
    ) -> tuple[
        list[int], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        """Encode the query and initialize Match/Insert/Delete rows of shape (K+1,).

        M0 represents Begin at score zero. The Delete row contains the complete
        leading deletion path; all other entries start at NEG_INF.

        For a profile built from ["AGGCE", "A..CE", "A...E"], the training paths are:

            Row 1: AGGCE → B → M1(A) → I1(G) → I1(G) → M2(C) → M3(E)
            Row 2: A..CE → B → M1(A) ────────────────→ M2(C) → M3(E)
            Row 3: A...E → B → M1(A) ────────────────→ D2    → M3(E)

        Query AAAA initializes these score rows before consuming any residues:

            encoded: [0, 0, 0, 0]
            Match:   [0, NEG_INF, NEG_INF, NEG_INF]
            Insert:  [NEG_INF, NEG_INF, NEG_INF, NEG_INF]
            Delete:  [NEG_INF, log2(1/6), log2(1/12), log2(1/36)]

        B -> D1: path probability of 1/6
        B -> D1 -> D2: 1/6 * 1/2 = 1/12
        B -> D1 -> D2 -> D3: 1/6 * 1/2 * 1/3 = 1/36

        For more details, see the "Transition Probabilities" section of protein_profile_hmm/viterbi_first_residue_walkthrough.md.
        """
        x = []
        for ch in seq:
            aa = encode(ch)
            if aa >= 0:
                x.append(aa)
        K = self.K
        match = np.full(K + 1, NEG_INF)
        insert = np.full(K + 1, NEG_INF)
        delete = np.full(K + 1, NEG_INF)
        match[0] = 0.0
        delete[1] = float(self.log_trans[0, T_BD])
        for k in range(2, K + 1):
            delete[k] = delete[k - 1] + float(self.log_trans[k - 1, T_DD])
        return x, match, insert, delete

    def _viterbi_row(
        self,
        aa: int,
        previous: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return one query residue's Match/Insert/Delete scores using best paths.

        Return three new arrays of shape (K+1,), indexed by profile position.

        Example: Training sequences

                ["AGGCE", "A..CE", "A...E"],

        with query AAAA. Before consuming the query's first A, previous contains the
        rows from _init_score_rows (M0 represents Begin):

            position:   0        1        2        3
            Match:      0     NEG_INF  NEG_INF  NEG_INF
            Insert:  NEG_INF  NEG_INF  NEG_INF  NEG_INF
            Delete:  NEG_INF   -2.585   -3.585   -5.170

        Calling _viterbi_row(AA_INDEX["A"], previous) returns the following
        rows:

            position:   0        1        2        3
            Match:   NEG_INF   0.570   -4.365   -5.015
            Insert:  -2.585   NEG_INF  NEG_INF  NEG_INF
            Delete:  NEG_INF NEG_INF  -1.015   -2.600

        Each candidate extends a predecessor's best path by adding the
        edge's log2 transition probability. Match and Insert also add
        log2(P(A | destination) / P(A | background)) for emitting this A.
        The choices follow the transition layout at the top of this module:

            destination    predecessors                 transitions
            Match[k>=2]    previous M/I/D at k-1        T_MM / T_IM / T_DM
            Insert[k]      previous M/I at k            T_MI / T_II
            Delete[k>=2]   new M/D at k-1               T_MD / T_DD

        In the first row of the matrix above, M1 is reached from Begin. M2 and M3 can only
        come from previous D1 and D2: previous M1/M2 and I1/I2 are unreachable.
        For example, with b = P(A | background), M2 is:

            B -> D1 -> M2(A): log2(1/6) + log2(1/2) + log2((1/22)/b)
                             = -4.365

        Among Insert states, only I0 is reachable, via the virtual M0 -> I0
        edge. Insert reads previous Match scores, so the new M1(A) cannot
        feed I1 until another residue is consumed. Delete emits nothing and uses the
        newly computed row: M1(A) -> D2 -> D3 gives -1.015 and -2.600.
        At D3, this beats the alternative new M2(A) -> D3 path (-6.687).
        For more details, see the "i = 1, k = 3" section of protein_profile_hmm/viterbi_first_residue_walkthrough.md.
        """
        previous_match, previous_insert, previous_delete = previous

        K = self.K
        match_scores = np.full(K + 1, NEG_INF)
        insert_scores = np.full(K + 1, NEG_INF)
        delete_scores = np.full(K + 1, NEG_INF)

        trans = self.log_trans
        match_odds = self.log_match_emit[:, aa] - self.log_bg[aa]
        insert_odds = self.log_insert_emit[:, aa] - self.log_bg[aa]

        # Begin and I0 are the only predecessors of M1.
        best = max(previous_match[0] + trans[0, T_BM], previous_insert[0] + trans[0, T_IM])
        match_scores[1] = match_odds[0] + best

        # Source positions 1..K-1 feed destination matches 2..K.
        from_match = previous_match[1:K] + trans[1:K, T_MM]
        from_insert = previous_insert[1:K] + trans[1:K, T_IM]
        from_delete = previous_delete[1:K] + trans[1:K, T_DM]
        best_match = np.maximum(np.maximum(from_match, from_insert), from_delete)
        match_scores[2:] = match_odds[1:] + best_match

        from_match = previous_match + trans[:, T_MI]
        from_insert = previous_insert + trans[:, T_II]
        best_insert = np.maximum(from_match, from_insert)
        insert_scores[:] = insert_odds + best_insert

        # D1 is only reachable before consuming residues.
        # Computing Dk requires the update to D(k-1).
        for k in range(2, K + 1):
            from_match = match_scores[k - 1] + trans[k - 1, T_MD]
            from_delete = delete_scores[k - 1] + trans[k - 1, T_DD]
            delete_scores[k] = max(from_match, from_delete)

        return match_scores, insert_scores, delete_scores

    def _forward_row(
        self,
        aa: int,
        previous: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return one query residue's scores by summing path weights in log2 space.

        previous contains Match/Insert/Delete rows for the query prefix before
        this residue. Each entry is the log2 sum of the weights of all paths
        ending at that state. Return three new arrays of shape (K+1,), indexed
        by profile position.

        Each candidate adds an edge's log2 transition probability to its
        predecessor's score. The choices follow the transition layout at the
        top of this module, as in _viterbi_row:

            destination    predecessors                 transitions
            Match[k>=2]    previous M/I/D at k-1        T_MM / T_IM / T_DM
            Insert[k]      previous M/I at k            T_MI / T_II
            Delete[k>=2]   new M/D at k-1               T_MD / T_DD

        For Match, nested np.logaddexp2 calls sum the contributions
        from M, I, and D into total_match. Match and Insert then add the
        emission log-odds log2(P(aa | destination) / P(aa | background)).
        These operations run across profile positions using array slices.
        M1 is handled separately: its predecessors are Begin (stored in M0)
        via T_BM and previous I0 via T_IM.

        The score itself uses total_match or total_insert, summing all predecessor paths.

        Dk (k >= 2) can be reached from the Match or Delete state at the
        preceding profile position. Because neither transition consumes a residue,
        both predecessors belong to the same query row and no emission odds are added.

        Example: train on ["AGGCE", "A..CE", "A...E"] and score query AAAA.
        Before the first A, previous contains the rows from _init_score_rows:

            position:   0        1        2        3
            Match:      0     NEG_INF  NEG_INF  NEG_INF
            Insert:  NEG_INF  NEG_INF  NEG_INF  NEG_INF
            Delete:  NEG_INF   -2.585   -3.585   -5.170

        Calling _forward_row(AA_INDEX["A"], previous) returns:

            position:   0        1        2        3
            Match:   NEG_INF   0.570   -4.365   -5.015
            Insert:  -2.585   NEG_INF  NEG_INF  NEG_INF
            Delete:  NEG_INF NEG_INF  -1.015   -2.517

        Each reachable Match/Insert state and D2 has only one path at this step, so their scores equal
        Viterbi's. D3 combines two paths within the new row:

            via D2: B -> M1(A) -> D2 -> D3    score -2.600
            via M2: B -> D1 -> M2(A) -> D3    score -6.687

        Forward computes logaddexp2(-2.600, -6.687), around -2.517;
        Viterbi takes their maximum, -2.600.

        For the derivation, see "i = 1, k = 3: reach the last profile position" in
        protein_profile_hmm/viterbi_first_residue_walkthrough.md.
        """
        previous_match, previous_insert, previous_delete = previous

        K = self.K
        match_scores = np.full(K + 1, NEG_INF)
        insert_scores = np.full(K + 1, NEG_INF)
        delete_scores = np.full(K + 1, NEG_INF)

        trans = self.log_trans
        match_odds = self.log_match_emit[:, aa] - self.log_bg[aa]
        insert_odds = self.log_insert_emit[:, aa] - self.log_bg[aa]

        from_begin = previous_match[0] + trans[0, T_BM]
        from_insert_zero = previous_insert[0] + trans[0, T_IM]
        match_scores[1] = match_odds[0] + np.logaddexp2(from_begin, from_insert_zero)

        from_match = previous_match[1:K] + trans[1:K, T_MM]
        from_insert = previous_insert[1:K] + trans[1:K, T_IM]
        from_delete = previous_delete[1:K] + trans[1:K, T_DM]
        total_match = np.logaddexp2(np.logaddexp2(from_match, from_insert), from_delete)
        match_scores[2:] = match_odds[1:] + total_match

        from_match = previous_match + trans[:, T_MI]
        from_insert = previous_insert + trans[:, T_II]
        total_insert = np.logaddexp2(from_match, from_insert)
        insert_scores[:] = insert_odds + total_insert

        for k in range(2, K + 1):
            from_match = match_scores[k - 1] + trans[k - 1, T_MD]
            from_delete = delete_scores[k - 1] + trans[k - 1, T_DD]
            delete_scores[k] = np.logaddexp2(from_match, from_delete)

        return match_scores, insert_scores, delete_scores

    def _terminal_candidates(
        self,
        rows: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]],
    ) -> tuple[float, float, float]:
        """Extend the final Match/Insert/Delete scores through End exactly once."""
        match, insert, delete = rows
        return (
            float(match[self.K] + self.log_trans[self.K, T_ME]),
            float(insert[self.K] + self.log_trans[self.K, T_IE]),
            float(delete[self.K] + self.log_trans[self.K, T_DE]),
        )

    def viterbi(self, seq: str) -> float:
        """Return the best complete path's log-odds bit score using O(K) storage.

        A complete path consumes all encoded residues and reaches silent End.
        The denominator is the background residue product conditional on length.
        Impossible queries return negative infinity.
        """
        x, match, insert, delete = self._init_score_rows(seq)
        previous = (match, insert, delete)
        for aa in x:
            previous = self._viterbi_row(aa, previous)
        return max(self._terminal_candidates(previous))

    def forward(self, seq: str) -> float:
        """Return all complete paths' log-odds bit score using O(K) storage.

        Sum path probabilities through End, relative to the background residue
        product conditional on length. Impossible queries return negative infinity.
        """
        x, match, insert, delete = self._init_score_rows(seq)
        previous = (match, insert, delete)
        for aa in x:
            previous = self._forward_row(aa, previous)
        return log2sumexp2(self._terminal_candidates(previous))
