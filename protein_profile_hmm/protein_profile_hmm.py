"""Glocal profile Hidden Markov Model for protein sequence comparison.

This is a simple re-implementation of HMMER's core
profile-HMM idea for educational purposes: a position-specific HMM with Match/Insert/Delete states per
consensus column, trained from a multiple sequence alignment, and used to
score query sequences against the family.

All emission and transition probabilities are stored in log2 space. Scoring
returns log-odds bit scores relative to a background null model: a positive
bit score means the sequence is more likely under the family model than under
the null.

The allowed transitions are:

- M : Match state; a sequence residue is aligned to consensus position k.
- I : Insert state; an extra sequence residue occurs between consensus positions k and k+1.
- D: Delete state; consensus position k is skipped, with no corresponding sequence residue.
- B: Begin state; a silent (non-counting) entry state into the profile consensus.

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
"""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import numpy.typing as npt

from .amino_acids import BACKGROUND_FREQ, N_AMINO_ACIDS, alignment_width, encode, encode_rows
from .msa import MSA

NEG_INF: float = -1e30
T_MM = 0
T_MI = 1
T_MD = 2
T_IM = 3
T_II = 4
T_DM = 5
T_DD = 6
T_BM = 7
T_BD = 8
N_TRANS = 9

# Limit event/index temporaries to one batch; a single wider row is still allowed.
_MAX_BATCH_CELLS = 250_000
# Source rows and destination columns are B, M, I, D; edges into B are excluded.
_TRANSITION_TYPES = np.array(
    [
        [-1, T_BM, T_MI, T_BD],
        [-1, T_MM, T_MI, T_MD],
        [-1, T_IM, T_II, T_IM],  # Preserve I->D folding into I->M.
        [-1, T_DM, T_DM, T_DD],  # Preserve D->I folding into D->M.
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
    if m <= NEG_INF / 2:
        return NEG_INF
    return m + float(np.log2(np.sum(2.0 ** (arr - m))))


@dataclass
class AlignmentTrace:
    """Ordered training paths, stored as parallel arrays with one entry per event.

    Each sequence starts with Begin at position 0. Match/Delete positions are
    1..K; Insert positions are 0..K. Only Match and Insert emit a residue;
    Begin and Delete carry -1. Insert-column gaps produce no event.
    Sequence indices refer to rows within the current batch.
    """

    sequence: npt.NDArray[np.intp]
    state: npt.NDArray[np.int8]
    position: npt.NDArray[np.intp]
    residue: npt.NDArray[np.int8]


def _trace_alignment(encoded: npt.NDArray[np.int8], is_match: np.ndarray) -> AlignmentTrace:
    """Turn alignment cells into ordered B/M/I/D paths for a batch of rows.

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
    """Count consecutive states within each training path, without priors.

    Keep the current counting conventions: B->I0 uses T_MI, I->D uses T_IM,
    and D->I uses T_DM. No End event or transition is added to the trace.

    For example, select both columns of ["AC", "A."] as match positions:

        encoded = encode_rows(["AC", "A."])
        trace = _trace_alignment(encoded, np.array([True, True]))
        counts = _count_transitions(trace, n_slots=3)

    Corresponding to the following paths:

        Row 1: AC -> B -> M1(A) -> M2(C)
        Row 2: A. -> B -> M1(A) -> D2

    This function counts the transitions between consecutive states in each path.
    The result has shape (3, N_TRANS). Rows identify source positions
    0, 1, and 2; columns identify transition types. Only three entries
    are nonzero:

        counts[0, T_BM] == 2  # B -> M1 in both rows.
        counts[1, T_MM] == 1  # M1 -> M2 in Row 1.
        counts[1, T_MD] == 1  # M1 -> D2 in Row 2.

    Neither M2 nor D2 has an outgoing transition to End; the counts at source position 2 stay zero.
    """
    same_sequence = trace.sequence[:-1] == trace.sequence[1:]
    source_state = trace.state[:-1][same_sequence]
    destination_state = trace.state[1:][same_sequence]
    source_position = trace.position[:-1][same_sequence]
    transition = _TRANSITION_TYPES[source_state, destination_state]

    counts = np.zeros((n_slots, N_TRANS), dtype=np.int64)
    np.add.at(counts, (source_position, transition), 1)
    return counts


def _count_profile(
    encoded: npt.NDArray[np.int8], is_match: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count match emissions, insert emissions, and transitions from one trace.

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
    trace = _trace_alignment(encoded, is_match)
    K = int(is_match.sum())
    return (
        _count_emissions(trace, State.MATCH, K),
        _count_emissions(trace, State.INSERT, K + 1),
        _count_transitions(trace, K + 1),
    )


def _count_alignment(rows: list[str], symfrac: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    sums = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, np.where(sums <= 0, 1.0, sums))


def _normalise_transitions(counts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Normalize each source's alternatives, retaining current boundary rules."""
    probs = counts.copy()
    groups = (
        probs[0:1, T_BM : T_BD + 1],  # Begin's M/D choice.
        probs[0:1, T_MI : T_MI + 1],  # Virtual M0->I0.
        probs[0:1, T_IM : T_II + 1],
        probs[1:-1, T_MM : T_MD + 1],  # Internal M, I, and D sources.
        probs[1:-1, T_IM : T_II + 1],
        probs[1:-1, T_DM : T_DD + 1],
        probs[-1:, T_MI : T_MI + 1],  # Terminal insertion and self-loop.
        probs[-1:, T_II : T_II + 1],
    )
    for group in groups:
        sums = group.sum(axis=1, keepdims=True)
        np.divide(group, sums, out=group, where=sums > 0)
    return probs


@dataclass
class ProfileHMM:
    """A glocal profile HMM.

    The profile has ``K`` Match columns (consensus positions). Conceptually
    there are also ``K`` Delete states (silent, one per Match column) and
    ``K + 1`` Insert states (one before column 1, one between each pair of
    consecutive match columns, one after column K).

    All log-prob arrays are in log2 space. Bit scores are computed as
    log-odds relative to ``BACKGROUND_FREQ``.
    """

    K: int
    log_match_emit: np.ndarray
    log_insert_emit: np.ndarray
    log_trans: np.ndarray
    log_bg: np.ndarray

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
        Each aligned sequence defines a training path; array operations count
        the residues and transitions along those paths. Match emissions and
        transitions get ``pseudocount`` per cell; inserts get 20 background-
        weighted counts in total. Large alignments use row batches with one
        global match mask. Probabilities are stored in log2 space.
        """
        match_counts, insert_counts, trans_counts = _count_alignment(msa.rows, symfrac)
        K = len(match_counts)

        # Add priors once, after combining all batches. Inserts retain their
        # fixed total of 20 background-weighted counts, independent of pseudocount.
        match_probs = _normalise_rows(match_counts.astype(float) + pseudocount)
        insert_probs = _normalise_rows(insert_counts + N_AMINO_ACIDS * BACKGROUND_FREQ)
        trans_probs = _normalise_transitions(trans_counts.astype(float) + pseudocount)

        log_match_emit = np.log2(np.clip(match_probs, 1e-300, None))
        log_insert_emit = np.log2(np.clip(insert_probs, 1e-300, None))
        log_trans = np.log2(np.clip(trans_probs, 1e-300, None))
        log_bg = np.log2(np.clip(BACKGROUND_FREQ, 1e-300, None))

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

    def _init_score_rows(self, seq: str) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
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
            Delete:  [NEG_INF, log2(1/5), log2(1/10), log2(1/30)]

        B -> D1: path probability of 1/5
        B -> D1 -> D2: 1/5 * 1/2 = 1/10
        B -> D1 -> D2 -> D3: 1/5 * 1/2 * 1/3 = 1/30

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
        if K >= 1:
            delete[1] = float(self.log_trans[0, T_BD])
            for k in range(2, K + 1):
                delete[k] = delete[k - 1] + float(self.log_trans[k - 1, T_DD])
        return x, match, insert, delete

    def _viterbi_row(
        self,
        aa: int,
        previous: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return one query residue's Match/Insert/Delete scores using best paths.

        Return three new arrays of shape (K+1,), indexed by profile position.

        Example: Training sequences

                ["AGGCE", "A..CE", "A...E"],

        with query AAAA. Before consuming the query's first A, previous contains the
        rows from _init_score_rows (M0 represents Begin):

            position:   0        1        2        3
            Match:      0     NEG_INF  NEG_INF  NEG_INF
            Insert:  NEG_INF  NEG_INF  NEG_INF  NEG_INF
            Delete:  NEG_INF   -2.322   -3.322   -4.907

        Calling _viterbi_row(AA_INDEX["A"], previous) returns the following
        rows:

            position:   0        1        2        3
            Match:   NEG_INF   0.833   -4.102   -4.752
            Insert:    0     NEG_INF  NEG_INF  NEG_INF
            Delete:  NEG_INF NEG_INF  -0.752   -2.336

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

            B -> D1 -> M2(A): log2(1/5) + log2(1/2) + log2((1/22)/b)
                             = -4.102

        Among Insert states, only I0 is reachable, via the virtual M0 -> I0
        edge. Insert reads previous Match scores, so the new M1(A) cannot
        feed I1 until another residue is consumed. Delete emits nothing and uses the
        newly computed row: M1(A) -> D2 -> D3 gives -0.752 and -2.336.
        At D3, this beats the alternative new M2(A) -> D3 path (-6.424).
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

        if K >= 1:
            # Begin and I0 are the only predecessors of M1.
            best = max(previous_match[0] + trans[0, T_BM], previous_insert[0] + trans[0, T_IM])
            if best > NEG_INF / 2:
                match_scores[1] = match_odds[0] + best

            # Source positions 1..K-1 feed destination matches 2..K.
            from_match = previous_match[1:K] + trans[1:K, T_MM]
            from_insert = previous_insert[1:K] + trans[1:K, T_IM]
            from_delete = previous_delete[1:K] + trans[1:K, T_DM]
            best_match = np.maximum(np.maximum(from_match, from_insert), from_delete)
            match_scores[2:] = np.where(
                best_match > NEG_INF / 2, match_odds[1:] + best_match, NEG_INF
            )

        from_match = previous_match + trans[:, T_MI]
        from_insert = previous_insert + trans[:, T_II]
        best_insert = np.maximum(from_match, from_insert)
        insert_scores[:] = np.where(best_insert > NEG_INF / 2, insert_odds + best_insert, NEG_INF)

        # D1 is only reachable before consuming residues.
        # Computing Dk requires the update to D(k-1).
        for k in range(2, K + 1):
            from_match = match_scores[k - 1] + trans[k - 1, T_MD]
            from_delete = delete_scores[k - 1] + trans[k - 1, T_DD]
            delete_scores[k] = max(NEG_INF, from_match, from_delete)

        return match_scores, insert_scores, delete_scores

    def _forward_row(
        self,
        aa: int,
        previous: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            Delete:  NEG_INF   -2.322   -3.322   -4.907

        Calling _forward_row(AA_INDEX["A"], previous) returns:

            position:   0        1        2        3
            Match:   NEG_INF   0.833   -4.102   -4.752
            Insert:    0     NEG_INF  NEG_INF  NEG_INF
            Delete:  NEG_INF NEG_INF  -0.752   -2.254

        Each reachable Match/Insert state and D2 has only one path at this step, so their scores equal
        Viterbi's. D3 combines two paths within the new row:

            via D2: B -> M1(A) -> D2 -> D3    score -2.336
            via M2: B -> D1 -> M2(A) -> D3    score -6.424

        Forward computes logaddexp2(-2.336, -6.424), around -2.254;
        Viterbi takes their maximum, -2.336.

        For the derivation, see "i = 1, k = 3: A match or deletion at position 3" in
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

        if K >= 1:
            from_begin = previous_match[0] + trans[0, T_BM]
            from_insert_zero = previous_insert[0] + trans[0, T_IM]
            if max(from_begin, from_insert_zero) > NEG_INF / 2:
                match_scores[1] = match_odds[0] + np.logaddexp2(from_begin, from_insert_zero)

            from_match = previous_match[1:K] + trans[1:K, T_MM]
            from_insert = previous_insert[1:K] + trans[1:K, T_IM]
            from_delete = previous_delete[1:K] + trans[1:K, T_DM]
            best_match = np.maximum(np.maximum(from_match, from_insert), from_delete)
            total_match = np.logaddexp2(np.logaddexp2(from_match, from_insert), from_delete)
            # Test the original candidates, as log2sumexp2 does before summing.
            match_scores[2:] = np.where(
                best_match > NEG_INF / 2, match_odds[1:] + total_match, NEG_INF
            )

        from_match = previous_match + trans[:, T_MI]
        from_insert = previous_insert + trans[:, T_II]
        best_insert = np.maximum(from_match, from_insert)
        total_insert = np.logaddexp2(from_match, from_insert)
        insert_scores[:] = np.where(best_insert > NEG_INF / 2, insert_odds + total_insert, NEG_INF)

        for k in range(2, K + 1):
            from_match = match_scores[k - 1] + trans[k - 1, T_MD]
            from_delete = delete_scores[k - 1] + trans[k - 1, T_DD]
            if max(from_match, from_delete) > NEG_INF / 2:
                delete_scores[k] = np.logaddexp2(from_match, from_delete)

        return match_scores, insert_scores, delete_scores

    def viterbi(self, seq: str) -> float:
        """Return the best path's log-odds bit score using O(K) DP storage."""
        x, match, insert, delete = self._init_score_rows(seq)
        previous = (match, insert, delete)
        for aa in x:
            previous = self._viterbi_row(aa, previous)
        return float(max(scores[self.K] for scores in previous))

    def forward(self, seq: str) -> float:
        """Return the summed paths' log-odds bit score using O(K) DP storage."""
        x, match, insert, delete = self._init_score_rows(seq)
        previous = (match, insert, delete)
        for aa in x:
            previous = self._forward_row(aa, previous)
        return log2sumexp2(scores[self.K] for scores in previous)
