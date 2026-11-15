"""Training traces, raw profile counts, and numerical helpers for profile HMMs."""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import numpy.typing as npt

from .amino_acids import N_AMINO_ACIDS, alignment_width, encode_rows

NEG_INF: float = -np.inf


class Transition(IntEnum):
    """Name the state changes stored in each column of the transition array.

    Each name pairs a source and destination: M is Match (emit a residue at
    a consensus position), I is Insert (emit an extra residue), and D is
    Delete (skip a consensus position without emitting). B and E denote the
    silent Begin and End states. Examples are:

    - MM: M_k -> M_{k+1}, advance to the next match position.
    - II: I_k -> I_k, emit another inserted residue at the same slot.
    - MD: M_k -> D_{k+1}, skip the next consensus position.
    - BM: Begin -> M1, enter the first match position (row 0 only).
    - ME: M_K -> End, finish from the last match position (row K only).

    At row 0, MI represents Begin -> I0, allowing leading insertions.
    """

    MM = 0
    MI = 1
    MD = 2
    IM = 3
    II = 4
    DM = 5
    DD = 6
    BM = 7
    BD = 8
    ME = 9
    IE = 10
    DE = 11


N_TRANS = len(Transition)

# Limit event/index temporaries to one batch; a single wider row is still allowed.
_MAX_BATCH_CELLS = 250_000
# Source rows and destination columns are B, M, I, D; edges into B are excluded.
_TRANSITION_TYPES = np.array(
    [
        [-1, Transition.BM, Transition.MI, Transition.BD],
        [-1, Transition.MM, Transition.MI, Transition.MD],
        [-1, Transition.IM, Transition.II, -1],  # I->D must be repaired before counting.
        [-1, Transition.DM, -1, Transition.DD],  # D->I must be repaired before counting.
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

    B->I0 uses Transition.MI. Unsupported edges, including unrepaired I/D pairs,
    raise ValueError. At least one match position is required. End transitions
    are counted from final events without adding End events to the trace.

    For example, select both columns of ["AC", "A."] as match positions:

        encoded = encode_rows(["AC", "A."])
        trace = _trace_alignment(encoded, np.array([True, True]))
        counts = _count_transitions(trace, n_slots=3)

    Corresponding to the following paths:

        Row 1: AC -> B -> M1(A) -> M2(C)
        Row 2: A. -> B -> M1(A) -> D2

    This function counts the transitions between consecutive states in each path.
    The result has shape (3, N_TRANS). Rows identify source positions
    0, 1, and 2; columns identify transition types. Five entries are nonzero:

        counts[0, Transition.BM] == 2  # B -> M1 in both rows.
        counts[1, Transition.MM] == 1  # M1 -> M2 in Row 1.
        counts[1, Transition.MD] == 1  # M1 -> D2 in Row 2.

        counts[2, Transition.ME] == 1  # M2 -> End in Row 1.
        counts[2, Transition.DE] == 1  # D2 -> End in Row 2.
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
    exit_types = np.array([-1, Transition.ME, Transition.IE, Transition.DE])[
        trace.state[final_indices]
    ]
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
        trans[0, Transition.BM] == 2          # B -> M1 in both rows.
        trans[1, Transition.MM] == 1          # M1 -> M2 in Row 1.
        trans[1, Transition.MD] == 1          # M1 -> D2 in Row 2.
        trans[2, Transition.ME] == 1          # M2 -> End in Row 1.
        trans[2, Transition.DE] == 1          # D2 -> End in Row 2.

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
    sums = counts.sum(axis=1, keepdims=True)
    probs = np.full_like(counts, 1.0 / counts.shape[1])
    return np.divide(counts, sums, out=probs, where=sums > 0)


def _normalise_transitions(
    counts: npt.NDArray[np.float64], pseudocount: float = 0.0
) -> npt.NDArray[np.float64]:
    """Smooth and normalize legal outgoing alternatives; unused cells stay zero."""
    probs = np.zeros_like(counts, dtype=float)
    groups = (
        (slice(0, 1), [Transition.BM, Transition.MI, Transition.BD]),
        (slice(0, 1), [Transition.IM, Transition.II]),
        (slice(1, -1), [Transition.MM, Transition.MI, Transition.MD]),
        (slice(1, -1), [Transition.IM, Transition.II]),
        (slice(1, -1), [Transition.DM, Transition.DD]),
        (slice(-1, None), [Transition.MI, Transition.ME]),
        (slice(-1, None), [Transition.II, Transition.IE]),
        (slice(-1, None), [Transition.DE]),
    )
    for slots, transitions in groups:
        probs[slots, transitions] = _normalise_rows(counts[slots, transitions] + pseudocount)
    return probs


def _log_probabilities(probs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert exact zero probabilities to negative infinity without warnings."""
    logs = np.full_like(probs, NEG_INF)
    return np.log2(probs, out=logs, where=probs > 0)
