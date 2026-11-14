"""Tests for training traces and raw profile counts."""

from itertools import pairwise

import numpy as np
import pytest

import protein_profile_hmm.traces as traces_module
from protein_profile_hmm.amino_acids import AA_INDEX, encode_rows
from protein_profile_hmm.traces import (
    N_TRANS,
    NEG_INF,
    AlignmentTrace,
    State,
    Transition,
    _count_alignment,
    _count_emissions,
    _count_profile,
    _count_transitions,
    _normalise_rows,
    _normalise_transitions,
    _repair_trace,
    _trace_alignment,
    log2sumexp2,
)
from tests.profile_helpers import allowed_transition_groups


def test_log2sumexp2_matches_direct_for_finite_values() -> None:
    assert abs(log2sumexp2([1.0, 2.0, 3.0]) - np.log2(14.0)) < 1e-9


def test_log2sumexp2_handles_neg_inf_mixed() -> None:
    assert abs(log2sumexp2([NEG_INF, 5.0, NEG_INF]) - 5.0) < 1e-9


def test_log2sumexp2_all_neg_inf_returns_neg_inf() -> None:
    assert log2sumexp2([NEG_INF, NEG_INF]) == NEG_INF


def test_log2sumexp2_empty_returns_neg_inf() -> None:
    assert log2sumexp2([]) == NEG_INF


def test_trace_alignment_records_states_positions_and_emitted_residues() -> None:
    """Record each row's state path, omitting the walkthrough's final M3(E).

    Row 1: AGGC -> B -> M1(A) -> I1(G) -> I1(G) -> M2(C)
    Row 2: A..C -> B -> M1(A) --------------------> M2(C)
    Row 3: A... -> B -> M1(A) --------------------> D2

    Each expected tuple is (zero-based row index, state, profile position,
    encoded residue). Begin uses position 0; Match and Delete use positions
    1 and 2. Both G residues visit insert slot 1, producing separate events.
    Gaps in insert columns produce no events, while the final match-column
    gap in row 3 produces D2. Match and Insert emit AA_INDEX residues;
    Begin and Delete emit nothing and carry residue -1.
    """
    trace = _trace_alignment(
        encode_rows(["AGGC", "A..C", "A..."]), np.array([True, False, False, True])
    )
    assert list(zip(trace.sequence, trace.state, trace.position, trace.residue, strict=True)) == [
        (0, State.BEGIN, 0, -1),
        (0, State.MATCH, 1, AA_INDEX["A"]),
        (0, State.INSERT, 1, AA_INDEX["G"]),
        (0, State.INSERT, 1, AA_INDEX["G"]),
        (0, State.MATCH, 2, AA_INDEX["C"]),
        (1, State.BEGIN, 0, -1),
        (1, State.MATCH, 1, AA_INDEX["A"]),
        (1, State.MATCH, 2, AA_INDEX["C"]),
        (2, State.BEGIN, 0, -1),
        (2, State.MATCH, 1, AA_INDEX["A"]),
        (2, State.DELETE, 2, -1),
    ]


def test_trace_alignment_keeps_begin_for_rows_with_no_state_events() -> None:
    trace = _trace_alignment(encode_rows(["..", "--"]), np.array([False, False]))
    assert trace.sequence.tolist() == [0, 1]
    assert trace.state.tolist() == [State.BEGIN, State.BEGIN]
    assert trace.position.tolist() == [0, 0]
    assert trace.residue.tolist() == [-1, -1]


def _trace_events(trace: AlignmentTrace) -> list[tuple[int, State, int, int]]:
    return [
        (int(row), State(int(state)), int(position), int(residue))
        for row, state, position, residue in zip(
            trace.sequence, trace.state, trace.position, trace.residue, strict=True
        )
    ]


def _sequential_repair(trace: AlignmentTrace) -> list[tuple[int, State, int, int]]:
    """Reference the left-to-right pair rule independently of array run selection."""
    events = _trace_events(trace)
    repaired = []
    index = 0
    while index < len(events):
        left = events[index]
        if index + 1 < len(events):
            right = events[index + 1]
            if left[0] == right[0] and (left[1], right[1]) in (
                (State.INSERT, State.DELETE),
                (State.DELETE, State.INSERT),
            ):
                deletion = left if left[1] == State.DELETE else right
                insertion = left if left[1] == State.INSERT else right
                repaired.append((left[0], State.MATCH, deletion[2], insertion[3]))
                index += 2
                continue
        repaired.append(left)
        index += 1
    return repaired


def _assert_trace_repair(row: str, columns: list[bool], path: list[tuple[State, int, str]]) -> None:
    """Check the path, input preservation, idempotence, and array dtypes."""
    trace = _trace_alignment(encode_rows([row]), np.array(columns, dtype=bool))
    original = _trace_events(trace)
    repaired = _repair_trace(trace)
    expected = [(0, State.BEGIN, 0, -1)] + [
        (0, state, position, AA_INDEX[residue] if residue else -1)
        for state, position, residue in path
    ]
    assert _trace_events(repaired) == expected
    assert _trace_events(trace) == original
    assert _trace_events(_repair_trace(repaired)) == expected
    assert repaired.sequence.dtype == repaired.position.dtype == np.dtype(np.intp)
    assert repaired.state.dtype == repaired.residue.dtype == np.dtype(np.int8)


def test_repair_trace_uses_leftmost_pair_in_delete_insert_delete() -> None:
    # D1 -> I1(G) -> D2 becomes M1(G) -> D2, not D1 -> M2(G).
    _assert_trace_repair(
        row=".G.",
        columns=[True, False, True],
        path=[(State.MATCH, 1, "G"), (State.DELETE, 2, "")],
    )


def test_repair_trace_uses_leftmost_pair_in_insert_delete_insert() -> None:
    # I0(G) -> D1 -> I1(W) becomes M1(G) -> I1(W), not I0(G) -> M1(W).
    _assert_trace_repair(
        row="G.W",
        columns=[False, True, False],
        path=[(State.MATCH, 1, "G"), (State.INSERT, 1, "W")],
    )


def test_repair_trace_repairs_disjoint_pairs_in_an_alternating_group() -> None:
    # D1 -> I1(G) -> D2 -> I2(W) becomes M1(G) -> M2(W).
    _assert_trace_repair(
        row=".G.W",
        columns=[True, False, True, False],
        path=[(State.MATCH, 1, "G"), (State.MATCH, 2, "W")],
    )


def test_repair_trace_keeps_unpaired_insertions_and_deletions() -> None:
    # I0(G) -> I0(W) -> D1 -> D2 becomes I0(G) -> M1(W) -> D2.
    _assert_trace_repair(
        row="GW..",
        columns=[False, False, True, True],
        path=[(State.INSERT, 0, "G"), (State.MATCH, 1, "W"), (State.DELETE, 2, "")],
    )


def test_repair_trace_moves_a_leading_insert_into_the_first_match() -> None:
    # B -> I0(G) -> D1 becomes B -> M1(G), including the last event in the pair.
    _assert_trace_repair(row="G.", columns=[False, True], path=[(State.MATCH, 1, "G")])


def test_repair_trace_keeps_the_remaining_trailing_insert() -> None:
    # D1 -> I1(G) -> I1(W) becomes M1(G) -> I1(W).
    _assert_trace_repair(
        row=".GW",
        columns=[True, False, False],
        path=[(State.MATCH, 1, "G"), (State.INSERT, 1, "W")],
    )


def test_repair_trace_restarts_pair_selection_after_a_match() -> None:
    # D1 I1(G) D2 I2(W) | M3(A) | D4 I4(Y) becomes M1(G) M2(W) M3(A) M4(Y).
    # The first group has three candidates; the next group's first must still be selected.
    _assert_trace_repair(
        row=".G.WA.Y",
        columns=[True, False, True, False, True, True, False],
        path=[
            (State.MATCH, 1, "G"),
            (State.MATCH, 2, "W"),
            (State.MATCH, 3, "A"),
            (State.MATCH, 4, "Y"),
        ],
    )


def test_repair_trace_leaves_an_already_valid_path_unchanged() -> None:
    # Insert-column gaps produce no events: the path is already B -> M1(A) -> M2(C).
    _assert_trace_repair(
        row="A..C",
        columns=[True, False, False, True],
        path=[(State.MATCH, 1, "A"), (State.MATCH, 2, "C")],
    )


def test_repair_trace_leaves_an_all_delete_path_unchanged() -> None:
    # B -> D1 -> D2 has no residue available for repair.
    _assert_trace_repair(
        row="..",
        columns=[True, True],
        path=[(State.DELETE, 1, ""), (State.DELETE, 2, "")],
    )


def test_repair_trace_handles_a_begin_only_path() -> None:
    # Both gaps are in insert columns, leaving only B and no adjacent event pair.
    _assert_trace_repair(row="..", columns=[False, False], path=[])


def test_repair_trace_matches_sequential_rule_across_sequences() -> None:
    """Check complete ordered traces over varied combinations of repair groups.

    The hand-written trace examples each cover one row. The other multi-row
    repair tests check aggregate counts/models, which cannot detect exchanging
    whole paths between sequence IDs. Here every event must match a sequential
    reference, including its row, state, position, and residue, across five-row
    alignments of varying widths and column masks. This also exercises combinations of
    group lengths and spacing beyond the fixed examples.

    The conservation and topology assertions overlap the reference comparison:
    they check its result against model constraints as an additional safeguard.
    The fixed seed makes failures reproducible; this is not exhaustive coverage.
    """
    rng = np.random.default_rng(20261108)
    allowed = {
        State.BEGIN: {State.MATCH, State.INSERT, State.DELETE},
        State.MATCH: {State.MATCH, State.INSERT, State.DELETE},
        State.INSERT: {State.MATCH, State.INSERT},
        State.DELETE: {State.MATCH, State.DELETE},
    }
    for _ in range(150):
        width = int(rng.integers(0, 40))
        rows = ["".join(row) for row in rng.choice(list("ACGW.-"), (5, width))]
        encoded = encode_rows(rows)
        columns = rng.random(width) < 0.5
        trace = _trace_alignment(encoded, columns)
        repaired = _repair_trace(trace)
        repaired_events = _trace_events(repaired)
        assert repaired_events == _sequential_repair(trace)
        pairs = trace.state.size - repaired.state.size
        for state, change in [(State.MATCH, pairs), (State.INSERT, -pairs), (State.DELETE, -pairs)]:
            assert (
                np.count_nonzero(repaired.state == state)
                == np.count_nonzero(trace.state == state) + change
            )
        for row in range(len(rows)):
            events = [event for event in repaired_events if event[0] == row]
            assert [event[3] for event in events if event[3] >= 0] == encoded[row][
                encoded[row] >= 0
            ].tolist()
            assert [
                event[2] for event in events if event[1] in (State.MATCH, State.DELETE)
            ] == list(range(1, int(columns.sum()) + 1))
            for source, destination in pairwise(events):
                assert destination[1] in allowed[source[1]]
                assert destination[2] == source[2] + (destination[1] != State.INSERT)


def test_boundary_count_transitions_rejects_profiles_without_match_positions() -> None:
    trace = _trace_alignment(encode_rows(["..", "--"]), np.array([False, False]))
    with pytest.raises(ValueError, match="at least one match position"):
        _count_transitions(trace, n_slots=1)


@pytest.mark.parametrize(
    ("row", "columns", "edge"),
    [
        ("G.", [False, True], "INSERT at 0 -> DELETE at 1"),
        (".G", [True, False], "DELETE at 1 -> INSERT at 1"),
    ],
)
def test_count_transitions_rejects_unrepaired_pairs(
    row: str, columns: list[bool], edge: str
) -> None:
    trace = _trace_alignment(encode_rows(["..", row]), np.array(columns))
    with pytest.raises(ValueError, match=f"sequence 1: {edge}; repair the trace"):
        _count_transitions(trace, n_slots=2)


def test_count_emissions_counts_two_match_positions() -> None:
    # AC and AA: the first column uses slot 0, the second uses slot 1.
    encoded = encode_rows(["AC", "AA"])
    is_match = np.array([True, True])
    trace = _trace_alignment(encoded, is_match)

    counts = _count_emissions(trace, State.MATCH, n_slots=2)

    expected = np.zeros((2, 20), dtype=np.int64)
    expected[0, AA_INDEX["A"]] = 2
    expected[1, [AA_INDEX["A"], AA_INDEX["C"]]] = 1
    np.testing.assert_array_equal(counts, expected)
    assert np.issubdtype(counts.dtype, np.integer)


def test_count_emissions_ignores_unselected_columns_and_gaps() -> None:
    encoded = encode_rows(["AGC", "C-."])
    is_match = np.array([True, False, True])
    trace = _trace_alignment(encoded, is_match)

    counts = _count_emissions(trace, State.MATCH, n_slots=2)

    # G is in an unselected column; the final dot contributes no residue.
    expected = np.zeros((2, 20), dtype=np.int64)
    expected[0, [AA_INDEX["A"], AA_INDEX["C"]]] = 1
    expected[1, AA_INDEX["C"]] = 1
    np.testing.assert_array_equal(counts, expected)


def test_count_emissions_combines_insert_columns_in_one_slot() -> None:
    encoded = encode_rows(["AGGC", "A..C", "A..C"])
    is_match = np.array([True, False, False, True])
    trace = _trace_alignment(encoded, is_match)

    counts = _count_emissions(trace, State.INSERT, n_slots=3)

    # Both G residues belong to I1; unused boundary slots (I0, I2) remain zero.
    expected = np.zeros((3, 20), dtype=np.int64)
    expected[1, AA_INDEX["G"]] = 2
    np.testing.assert_array_equal(counts, expected)


@pytest.mark.parametrize(
    ("rows", "columns"),
    [(["A-", "-C"], [False, False]), (["--", ".."], [True, True])],
    ids=["no-selected-columns", "only-gaps"],
)
def test_count_emissions_returns_zero_counts_without_selected_residues(
    rows: list[str], columns: list[bool]
) -> None:
    trace = _trace_alignment(encode_rows(rows), np.array(columns))
    counts = _count_emissions(trace, State.MATCH, n_slots=2)
    np.testing.assert_array_equal(counts, np.zeros((2, 20), dtype=np.int64))


def test_count_emissions_handles_slots_beyond_the_residue_dtype_range() -> None:
    # A is at M1 and Y is at M130, with deletions between them.
    encoded = encode_rows(["A" + "." * 128 + "Y"] * 2)
    trace = _trace_alignment(encoded, np.ones(130, dtype=bool))

    counts = _count_emissions(trace, State.MATCH, n_slots=130)

    expected = np.zeros((130, 20), dtype=np.int64)
    expected[0, AA_INDEX["A"]] = 2
    expected[129, AA_INDEX["Y"]] = 2
    np.testing.assert_array_equal(counts, expected)


def test_count_transitions_counts_matches_and_deletions_within_each_row() -> None:
    # AC: B -> M1 -> M2; A.: B -> M1 -> D2.
    encoded = encode_rows(["AC", "A."])
    is_match = np.array([True, True])
    trace = _trace_alignment(encoded, is_match)

    counts = _count_transitions(trace, n_slots=3)

    expected = np.zeros((3, N_TRANS), dtype=np.int64)
    expected[0, Transition.BM] = 2
    expected[1, [Transition.MM, Transition.MD]] = 1
    expected[2, [Transition.ME, Transition.DE]] = 1
    np.testing.assert_array_equal(counts, expected)


def test_count_transitions_skips_insert_gaps_and_counts_insert_loops() -> None:
    # AGGC: B -> M1 -> I1 -> I1 -> M2.
    # A..C: B -> M1 -> M2; A...: B -> M1 -> D2.
    encoded = encode_rows(["AGGC", "A..C", "A..."])
    is_match = np.array([True, False, False, True])
    trace = _trace_alignment(encoded, is_match)

    counts = _count_transitions(trace, n_slots=3)

    expected = np.zeros((3, N_TRANS), dtype=np.int64)
    expected[0, Transition.BM] = 3
    expected[1, [Transition.MI, Transition.II, Transition.IM, Transition.MM, Transition.MD]] = 1
    expected[2, [Transition.ME, Transition.DE]] = [2, 1]
    np.testing.assert_array_equal(counts, expected)


def test_count_transitions_after_delete_insert_repair() -> None:
    # A.GC: B -> M1(A) -> D2 -> I2(G) -> M3(C) repairs to M1(A) -> M2(G) -> M3(C).
    # Both AC.C rows: B -> M1 -> M2 -> M3.
    encoded = encode_rows(["A.GC", "AC.C", "AC.C"])
    is_match = np.array([True, True, False, True])
    trace = _repair_trace(_trace_alignment(encoded, is_match))

    counts = _count_transitions(trace, n_slots=4)

    expected = np.zeros((4, N_TRANS), dtype=np.int64)
    expected[0, Transition.BM] = 3
    expected[1:3, Transition.MM] = 3
    expected[3, Transition.ME] = 3
    np.testing.assert_array_equal(counts, expected)


def test_count_transitions_counts_leading_and_trailing_insert_loops() -> None:
    # GGA..CYY: B -> I0 -> I0 -> M1 -> M2 -> I2 -> I2.
    # ..A..C..: B -> M1 -> M2; .....C..: B -> D1 -> M2.
    encoded = encode_rows(["GGA..CYY", "..A..C..", ".....C.."])
    is_match = np.array([False, False, True, False, False, True, False, False])
    trace = _trace_alignment(encoded, is_match)

    counts = _count_transitions(trace, n_slots=3)

    expected = np.zeros((3, N_TRANS), dtype=np.int64)
    expected[0, [Transition.MI, Transition.II, Transition.IM, Transition.BM, Transition.BD]] = 1
    expected[1, Transition.MM] = 2
    expected[1, Transition.DM] = 1
    expected[2, [Transition.MI, Transition.II]] = 1
    expected[2, [Transition.ME, Transition.IE]] = [2, 1]
    np.testing.assert_array_equal(counts, expected)


def test_count_transitions_after_insert_delete_repair() -> None:
    # AGG..E repairs to B -> M1(A) -> I1(G) -> M2(G) -> D3 -> M4(E).
    # A..CGE: B -> M1 -> M2 -> M3 -> M4.
    encoded = encode_rows(["AGG..E", "A..CGE"])
    is_match = np.array([True, False, False, True, True, True])
    trace = _repair_trace(_trace_alignment(encoded, is_match))

    counts = _count_transitions(trace, n_slots=5)

    expected = np.zeros((5, N_TRANS), dtype=np.int64)
    expected[0, Transition.BM] = 2
    expected[1, [Transition.MI, Transition.IM, Transition.MM]] = 1
    expected[2, [Transition.MD, Transition.MM]] = 1
    expected[3, [Transition.DM, Transition.MM]] = 1
    expected[4, Transition.ME] = 2
    np.testing.assert_array_equal(counts, expected)


def test_count_transitions_after_terminal_delete_insert_repair() -> None:
    # A.GG repairs to B -> M1(A) -> M2(G) -> I2(G); AC..: B -> M1 -> M2.
    encoded = encode_rows(["A.GG", "AC.."])
    is_match = np.array([True, True, False, False])
    trace = _repair_trace(_trace_alignment(encoded, is_match))

    counts = _count_transitions(trace, n_slots=3)

    expected = np.zeros((3, N_TRANS), dtype=np.int64)
    expected[0, Transition.BM] = 2
    expected[1, Transition.MM] = 2
    expected[2, Transition.MI] = 1
    expected[2, [Transition.ME, Transition.IE]] = 1
    np.testing.assert_array_equal(counts, expected)


def test_count_transitions_handles_a_single_match_position() -> None:
    # Two rows take B -> D1; the middle row takes B -> M1.
    encoded = encode_rows([".", "A", "-"])
    is_match = np.array([True])
    trace = _trace_alignment(encoded, is_match)

    counts = _count_transitions(trace, n_slots=2)

    expected = np.zeros((2, N_TRANS), dtype=np.int64)
    expected[0, Transition.BM] = 1
    expected[0, Transition.BD] = 2
    expected[1, [Transition.ME, Transition.DE]] = [1, 2]
    np.testing.assert_array_equal(counts, expected)


def test_count_profile_accumulates_counts_beyond_the_residue_dtype_range() -> None:
    # 150 paths insert two G residues at I1; another 150 go straight from M1 to M2.
    encoded = encode_rows(["AGGC"] * 150 + ["A..C"] * 150)
    match, insert, trans = _count_profile(encoded, np.array([True, False, False, True]))

    expected_match = np.zeros((2, 20), dtype=np.int64)
    expected_match[0, AA_INDEX["A"]] = 300
    expected_match[1, AA_INDEX["C"]] = 300
    expected_insert = np.zeros((3, 20), dtype=np.int64)
    expected_insert[1, AA_INDEX["G"]] = 300
    expected_trans = np.zeros((3, N_TRANS), dtype=np.int64)
    expected_trans[0, Transition.BM] = 300
    expected_trans[1, [Transition.MM, Transition.MI, Transition.II, Transition.IM]] = 150
    expected_trans[2, Transition.ME] = 300
    np.testing.assert_array_equal(match, expected_match)
    np.testing.assert_array_equal(insert, expected_insert)
    np.testing.assert_array_equal(trans, expected_trans)


@pytest.mark.parametrize("batch_cells", [1, 6, 250_000])
def test_count_alignment_uses_one_global_mask_and_accumulates_raw_counts(
    monkeypatch: pytest.MonkeyPatch, batch_cells: int
) -> None:
    # Global mask: I0, M1, I1. The first two rows alone would select all columns.
    monkeypatch.setattr(traces_module, "_MAX_BATCH_CELLS", batch_cells)
    match, insert, trans = _count_alignment(["GAQ", "...", ".A."], symfrac=0.5)
    expected_match = np.zeros((1, 20), dtype=np.int64)
    expected_match[0, AA_INDEX["A"]] = 2
    expected_insert = np.zeros((2, 20), dtype=np.int64)
    expected_insert[0, AA_INDEX["G"]] = 1
    expected_insert[1, AA_INDEX["Q"]] = 1
    expected_trans = np.zeros((2, N_TRANS), dtype=np.int64)
    expected_trans[0, [Transition.MI, Transition.IM, Transition.BD, Transition.BM]] = 1
    expected_trans[1, Transition.MI] = 1
    expected_trans[1, [Transition.ME, Transition.IE, Transition.DE]] = 1
    np.testing.assert_array_equal(match, expected_match)
    np.testing.assert_array_equal(insert, expected_insert)
    np.testing.assert_array_equal(trans, expected_trans)


@pytest.mark.parametrize(
    "rows",
    [["AG.E", "A.CE", "A.CE"], ["A.GE", "AC.E", "AC.E"]],
    ids=["insert-delete", "delete-insert"],
)
def test_count_alignment_repairs_emissions_and_transitions_together(rows: list[str]) -> None:
    # Both alignments repair to AGE / ACE / ACE: G must move from I1/I2 into M2.
    match, insert, trans = _count_alignment(rows, symfrac=0.5)
    expected_match = np.zeros((3, 20), dtype=np.int64)
    expected_match[0, AA_INDEX["A"]] = 3
    expected_match[1, [AA_INDEX["C"], AA_INDEX["G"]]] = [2, 1]
    expected_match[2, AA_INDEX["E"]] = 3
    expected_trans = np.zeros((4, N_TRANS), dtype=np.int64)
    expected_trans[0, Transition.BM] = 3
    expected_trans[1:3, Transition.MM] = 3
    expected_trans[3, Transition.ME] = 3
    np.testing.assert_array_equal(match, expected_match)
    np.testing.assert_array_equal(insert, np.zeros((4, 20), dtype=np.int64))
    np.testing.assert_array_equal(trans, expected_trans)


def test_repair_counts_preserve_large_positions_and_observation_totals() -> None:
    # I129(G) -> D130 becomes M130(G) in all 300 rows.
    rows = ["A" * 129 + "G."] * 300
    columns = np.array([True] * 129 + [False, True])
    match, insert, trans = _count_profile(encode_rows(rows), columns)
    expected_match = np.zeros((130, 20), dtype=np.int64)
    expected_match[:129, AA_INDEX["A"]] = 300
    expected_match[129, AA_INDEX["G"]] = 300
    expected_trans = np.zeros((131, N_TRANS), dtype=np.int64)
    expected_trans[0, Transition.BM] = 300
    expected_trans[1:130, Transition.MM] = 300
    expected_trans[130, Transition.ME] = 300
    np.testing.assert_array_equal(match, expected_match)
    np.testing.assert_array_equal(insert, np.zeros((131, 20), dtype=np.int64))
    np.testing.assert_array_equal(trans, expected_trans)


@pytest.mark.parametrize("batch_cells", [1, 16, 250_000])
def test_boundary_counts_one_repaired_exit_per_alignment_row(
    monkeypatch: pytest.MonkeyPatch, batch_cells: int
) -> None:
    monkeypatch.setattr(traces_module, "_MAX_BATCH_CELLS", batch_cells)
    # A.G. repairs terminal D2/I2 to M2; A.GG retains one I2; .... ends at D2.
    rows = ["A.G.", "A.GG", "....", "AC..", "AC..", "AC.."]
    _, _, counts = _count_alignment(rows, symfrac=0.5)
    np.testing.assert_array_equal(
        counts[-1, [Transition.ME, Transition.IE, Transition.DE]], [4, 1, 1]
    )
    assert counts[:, [Transition.ME, Transition.IE, Transition.DE]].sum() == len(rows)
    assert not counts[:-1, [Transition.ME, Transition.IE, Transition.DE]].any()


@pytest.mark.parametrize("empty", [False, True])
def test_normalise_rows_handles_zero_and_small_counts_without_mutating_input(empty: bool) -> None:
    counts = np.array([[1.0, 3.0], [0.0, 0.0], [1e-200, 3e-200]])
    expected = np.array([[0.25, 0.75], [0.5, 0.5], [0.25, 0.75]])
    if empty:
        counts, expected = counts[:0], expected[:0]
    original = counts.copy()
    actual = _normalise_rows(counts)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(counts, original)
    assert actual.dtype == np.float64


@pytest.mark.parametrize("K", [1, 3])
@pytest.mark.parametrize("scale", [0.0, 1.0])
@pytest.mark.parametrize("pseudocount", [0.0, 0.1, 1.0])
def test_normalise_transitions_normalises_only_allowed_groups(
    K: int, scale: float, pseudocount: float
) -> None:
    counts = np.tile(np.arange(1, N_TRANS + 1, dtype=float), (K + 1, 1)) * scale
    original = counts.copy()
    expected = np.zeros_like(counts)
    for k, transitions in allowed_transition_groups(K):
        values = counts[k, transitions] + pseudocount
        expected[k, transitions] = values / values.sum() if values.sum() else 1 / len(values)
    actual = _normalise_transitions(counts, pseudocount)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(counts, original)
