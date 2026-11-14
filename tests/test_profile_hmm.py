"""Tests for ProfileHMM."""

from io import StringIO
from itertools import pairwise

import numpy as np
import pytest

import protein_profile_hmm.protein_profile_hmm as profile_module
from protein_profile_hmm.amino_acids import AA_INDEX, BACKGROUND_FREQ, encode_rows
from protein_profile_hmm.msa import MSA, match_column_mask, parse_stockholm
from protein_profile_hmm.protein_profile_hmm import (
    N_TRANS,
    NEG_INF,
    T_BD,
    T_BM,
    T_DD,
    T_DE,
    T_DM,
    T_IE,
    T_II,
    T_IM,
    T_MD,
    T_ME,
    T_MI,
    T_MM,
    AlignmentTrace,
    ProfileHMM,
    State,
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


def _build(rows: list[str], **kwargs: float) -> ProfileHMM:
    """Build a ``ProfileHMM`` from an inline Stockholm MSA over ``rows``."""
    body = "\n".join(f"s{i + 1} {r}" for i, r in enumerate(rows))
    data = f"# STOCKHOLM 1.0\n{body}\n//\n"
    msa = parse_stockholm(StringIO(data))
    return ProfileHMM.build_from_msa(msa, **kwargs)


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
    expected[0, T_BM] = 2
    expected[1, [T_MM, T_MD]] = 1
    expected[2, [T_ME, T_DE]] = 1
    np.testing.assert_array_equal(counts, expected)


def test_count_transitions_skips_insert_gaps_and_counts_insert_loops() -> None:
    # AGGC: B -> M1 -> I1 -> I1 -> M2.
    # A..C: B -> M1 -> M2; A...: B -> M1 -> D2.
    encoded = encode_rows(["AGGC", "A..C", "A..."])
    is_match = np.array([True, False, False, True])
    trace = _trace_alignment(encoded, is_match)

    counts = _count_transitions(trace, n_slots=3)

    expected = np.zeros((3, N_TRANS), dtype=np.int64)
    expected[0, T_BM] = 3
    expected[1, [T_MI, T_II, T_IM, T_MM, T_MD]] = 1
    expected[2, [T_ME, T_DE]] = [2, 1]
    np.testing.assert_array_equal(counts, expected)


def test_count_transitions_after_delete_insert_repair() -> None:
    # A.GC: B -> M1(A) -> D2 -> I2(G) -> M3(C) repairs to M1(A) -> M2(G) -> M3(C).
    # Both AC.C rows: B -> M1 -> M2 -> M3.
    encoded = encode_rows(["A.GC", "AC.C", "AC.C"])
    is_match = np.array([True, True, False, True])
    trace = _repair_trace(_trace_alignment(encoded, is_match))

    counts = _count_transitions(trace, n_slots=4)

    expected = np.zeros((4, N_TRANS), dtype=np.int64)
    expected[0, T_BM] = 3
    expected[1:3, T_MM] = 3
    expected[3, T_ME] = 3
    np.testing.assert_array_equal(counts, expected)


def test_count_transitions_counts_leading_and_trailing_insert_loops() -> None:
    # GGA..CYY: B -> I0 -> I0 -> M1 -> M2 -> I2 -> I2.
    # ..A..C..: B -> M1 -> M2; .....C..: B -> D1 -> M2.
    encoded = encode_rows(["GGA..CYY", "..A..C..", ".....C.."])
    is_match = np.array([False, False, True, False, False, True, False, False])
    trace = _trace_alignment(encoded, is_match)

    counts = _count_transitions(trace, n_slots=3)

    expected = np.zeros((3, N_TRANS), dtype=np.int64)
    expected[0, [T_MI, T_II, T_IM, T_BM, T_BD]] = 1
    expected[1, T_MM] = 2
    expected[1, T_DM] = 1
    expected[2, [T_MI, T_II]] = 1
    expected[2, [T_ME, T_IE]] = [2, 1]
    np.testing.assert_array_equal(counts, expected)


def test_count_transitions_after_insert_delete_repair() -> None:
    # AGG..E repairs to B -> M1(A) -> I1(G) -> M2(G) -> D3 -> M4(E).
    # A..CGE: B -> M1 -> M2 -> M3 -> M4.
    encoded = encode_rows(["AGG..E", "A..CGE"])
    is_match = np.array([True, False, False, True, True, True])
    trace = _repair_trace(_trace_alignment(encoded, is_match))

    counts = _count_transitions(trace, n_slots=5)

    expected = np.zeros((5, N_TRANS), dtype=np.int64)
    expected[0, T_BM] = 2
    expected[1, [T_MI, T_IM, T_MM]] = 1
    expected[2, [T_MD, T_MM]] = 1
    expected[3, [T_DM, T_MM]] = 1
    expected[4, T_ME] = 2
    np.testing.assert_array_equal(counts, expected)


def test_count_transitions_after_terminal_delete_insert_repair() -> None:
    # A.GG repairs to B -> M1(A) -> M2(G) -> I2(G); AC..: B -> M1 -> M2.
    encoded = encode_rows(["A.GG", "AC.."])
    is_match = np.array([True, True, False, False])
    trace = _repair_trace(_trace_alignment(encoded, is_match))

    counts = _count_transitions(trace, n_slots=3)

    expected = np.zeros((3, N_TRANS), dtype=np.int64)
    expected[0, T_BM] = 2
    expected[1, T_MM] = 2
    expected[2, T_MI] = 1
    expected[2, [T_ME, T_IE]] = 1
    np.testing.assert_array_equal(counts, expected)


def test_count_transitions_handles_a_single_match_position() -> None:
    # Two rows take B -> D1; the middle row takes B -> M1.
    encoded = encode_rows([".", "A", "-"])
    is_match = np.array([True])
    trace = _trace_alignment(encoded, is_match)

    counts = _count_transitions(trace, n_slots=2)

    expected = np.zeros((2, N_TRANS), dtype=np.int64)
    expected[0, T_BM] = 1
    expected[0, T_BD] = 2
    expected[1, [T_ME, T_DE]] = [1, 2]
    np.testing.assert_array_equal(counts, expected)


def test_count_profile_assigns_match_positions_and_boundary_insert_slots() -> None:
    # Paths: I0(G) -> M1(A) -> I1(Y); D1; M1(A).
    encoded = encode_rows(["GAQ", "...", ".A."])
    match, insert, trans = _count_profile(encoded, np.array([False, True, False]))
    expected_match = np.zeros((1, 20), dtype=np.int64)
    expected_match[0, AA_INDEX["A"]] = 2
    expected_insert = np.zeros((2, 20), dtype=np.int64)
    expected_insert[0, AA_INDEX["G"]] = 1
    expected_insert[1, AA_INDEX["Q"]] = 1
    expected_trans = np.zeros((2, N_TRANS), dtype=np.int64)
    expected_trans[0, [T_MI, T_IM, T_BD, T_BM]] = 1
    expected_trans[1, T_MI] = 1
    expected_trans[1, [T_ME, T_IE, T_DE]] = 1
    np.testing.assert_array_equal(match, expected_match)
    np.testing.assert_array_equal(insert, expected_insert)
    np.testing.assert_array_equal(trans, expected_trans)


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
    expected_trans[0, T_BM] = 300
    expected_trans[1, [T_MM, T_MI, T_II, T_IM]] = 150
    expected_trans[2, T_ME] = 300
    np.testing.assert_array_equal(match, expected_match)
    np.testing.assert_array_equal(insert, expected_insert)
    np.testing.assert_array_equal(trans, expected_trans)


@pytest.mark.parametrize("batch_cells", [1, 6, 250_000])
def test_count_alignment_uses_one_global_mask_and_accumulates_raw_counts(
    monkeypatch: pytest.MonkeyPatch, batch_cells: int
) -> None:
    # Global mask: I0, M1, I1. The first two rows alone would select all columns.
    monkeypatch.setattr(profile_module, "_MAX_BATCH_CELLS", batch_cells)
    match, insert, trans = _count_alignment(["GAQ", "...", ".A."], symfrac=0.5)
    expected_match = np.zeros((1, 20), dtype=np.int64)
    expected_match[0, AA_INDEX["A"]] = 2
    expected_insert = np.zeros((2, 20), dtype=np.int64)
    expected_insert[0, AA_INDEX["G"]] = 1
    expected_insert[1, AA_INDEX["Q"]] = 1
    expected_trans = np.zeros((2, N_TRANS), dtype=np.int64)
    expected_trans[0, [T_MI, T_IM, T_BD, T_BM]] = 1
    expected_trans[1, T_MI] = 1
    expected_trans[1, [T_ME, T_IE, T_DE]] = 1
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
    expected_trans[0, T_BM] = 3
    expected_trans[1:3, T_MM] = 3
    expected_trans[3, T_ME] = 3
    np.testing.assert_array_equal(match, expected_match)
    np.testing.assert_array_equal(insert, np.zeros((4, 20), dtype=np.int64))
    np.testing.assert_array_equal(trans, expected_trans)


@pytest.mark.parametrize(
    "rows",
    [["AG.E", "A.CE", "A.CE"], ["A.GE", "AC.E", "AC.E"]],
    ids=["insert-delete", "delete-insert"],
)
def test_build_from_msa_matches_manually_repaired_alignment(rows: list[str]) -> None:
    hmm = _build(rows)
    # Align the same residues directly to the repaired Match positions.
    expected = _build(["AGE", "ACE", "ACE"])
    assert hmm.K == expected.K == 3
    for name in ("log_match_emit", "log_insert_emit", "log_trans", "log_bg"):
        np.testing.assert_allclose(getattr(hmm, name), getattr(expected, name), atol=1e-12, rtol=0)
    np.testing.assert_allclose(2**hmm.log_insert_emit, np.tile(BACKGROUND_FREQ, (4, 1)))
    for seq in ("", "A", "AGE", "ACE", "GWAGEG"):
        np.testing.assert_allclose(hmm.viterbi(seq), expected.viterbi(seq), atol=1e-12, rtol=0)
        np.testing.assert_allclose(hmm.forward(seq), expected.forward(seq), atol=1e-12, rtol=0)
        assert hmm.forward(seq) >= hmm.viterbi(seq) - 1e-12


@pytest.mark.parametrize("pseudocount", [0.0, 1.0], ids=["observed-only", "smoothed"])
def test_build_from_msa_smooths_repaired_match_counts(pseudocount: float) -> None:
    # Repair gives M2 two C observations and one G, with no remaining insert emissions.
    # Smoothing must use these repaired counts: (count + p) / (3 + 20*p).
    hmm = _build(["AG.E", "A.CE", "A.CE"], pseudocount=pseudocount)
    np.testing.assert_allclose(
        2 ** hmm.log_match_emit[1, [AA_INDEX["C"], AA_INDEX["G"]]],
        (np.array([2, 1]) + pseudocount) / (3 + 20 * pseudocount),
    )
    np.testing.assert_allclose(2**hmm.log_insert_emit, np.tile(BACKGROUND_FREQ, (4, 1)))


@pytest.mark.parametrize("batch_cells", [1, 14, 250_000])
def test_build_from_msa_repairs_batches_with_one_global_mask(
    monkeypatch: pytest.MonkeyPatch, batch_cells: int
) -> None:
    rows = ["G.W.Y.G", "..G.W..", ".A.C..W"] + [".A.C.E."] * 4
    # The first two rows alone would select the insert columns as matches.
    assert match_column_mask(rows).tolist() == [False, True, False, True, False, True, False]
    corrected_rows = ["GWYG", "GW..", "ACW."] + ["ACE."] * 4
    expected_counts = _count_alignment(corrected_rows, symfrac=0.5)
    expected = _build(corrected_rows)
    monkeypatch.setattr(profile_module, "_MAX_BATCH_CELLS", batch_cells)
    actual_counts = _count_alignment(rows, symfrac=0.5)
    for actual, counts in zip(actual_counts, expected_counts, strict=True):
        np.testing.assert_array_equal(actual, counts)
    hmm = _build(rows)
    assert hmm.K == expected.K == 3
    for name in ("log_match_emit", "log_insert_emit", "log_trans", "log_bg"):
        np.testing.assert_allclose(getattr(hmm, name), getattr(expected, name), atol=1e-12, rtol=0)


def test_repair_counts_preserve_large_positions_and_observation_totals() -> None:
    # I129(G) -> D130 becomes M130(G) in all 300 rows.
    rows = ["A" * 129 + "G."] * 300
    columns = np.array([True] * 129 + [False, True])
    match, insert, trans = _count_profile(encode_rows(rows), columns)
    expected_match = np.zeros((130, 20), dtype=np.int64)
    expected_match[:129, AA_INDEX["A"]] = 300
    expected_match[129, AA_INDEX["G"]] = 300
    expected_trans = np.zeros((131, N_TRANS), dtype=np.int64)
    expected_trans[0, T_BM] = 300
    expected_trans[1:130, T_MM] = 300
    expected_trans[130, T_ME] = 300
    np.testing.assert_array_equal(match, expected_match)
    np.testing.assert_array_equal(insert, np.zeros((131, 20), dtype=np.int64))
    np.testing.assert_array_equal(trans, expected_trans)


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


def _allowed_transition_groups(K: int) -> list[tuple[int, list[int]]]:
    groups = [(0, [T_BM, T_MI, T_BD]), (0, [T_IM, T_II])]
    for k in range(1, K):
        groups.extend([(k, [T_MM, T_MI, T_MD]), (k, [T_IM, T_II]), (k, [T_DM, T_DD])])
    return [*groups, (K, [T_MI, T_ME]), (K, [T_II, T_IE]), (K, [T_DE])]


@pytest.mark.parametrize("K", [1, 3])
@pytest.mark.parametrize("scale", [0.0, 1.0])
@pytest.mark.parametrize("pseudocount", [0.0, 0.1, 1.0])
def test_normalise_transitions_normalises_only_allowed_groups(
    K: int, scale: float, pseudocount: float
) -> None:
    counts = np.tile(np.arange(1, N_TRANS + 1, dtype=float), (K + 1, 1)) * scale
    original = counts.copy()
    expected = np.zeros_like(counts)
    for k, transitions in _allowed_transition_groups(K):
        values = counts[k, transitions] + pseudocount
        expected[k, transitions] = values / values.sum() if values.sum() else 1 / len(values)
    actual = _normalise_transitions(counts, pseudocount)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(counts, original)


def test_build_from_msa_picks_correct_number_of_match_columns() -> None:
    hmm = _build(["ACDE", "ACDE"], symfrac=0.5, pseudocount=1.0)
    assert hmm.K == 4


def test_build_from_msa_treats_low_residue_columns_as_insert() -> None:
    hmm = _build(["AC.DE", "AC.DE", "AC.DE"], symfrac=0.5, pseudocount=1.0)
    assert hmm.K == 4


@pytest.mark.parametrize("rows", [[], [""], ["...", "---"]])
def test_build_from_msa_rejects_empty_or_no_match_alignment(rows: list[str]) -> None:
    with pytest.raises(ValueError, match=r"empty MSA|no match columns"):
        ProfileHMM.build_from_msa(MSA([], rows))


def test_build_from_msa_batches_reject_no_match_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profile_module, "_MAX_BATCH_CELLS", 1)
    with pytest.raises(ValueError, match="no match columns"):
        _build(["...", "---"])


@pytest.mark.parametrize("batch_cells", [1, 250_000])
def test_build_from_msa_handles_all_delete_paths_without_pseudocounts(
    monkeypatch: pytest.MonkeyPatch, batch_cells: int
) -> None:
    monkeypatch.setattr(profile_module, "_MAX_BATCH_CELLS", batch_cells)
    # At symfrac=0 even empty columns are match positions: both paths are B -> D1 -> D2.
    hmm = _build(["..", "--"], symfrac=0.0, pseudocount=0.0)
    assert hmm.K == 2
    np.testing.assert_allclose(hmm.log_match_emit, np.full((2, 20), np.log2(1 / 20)))
    np.testing.assert_allclose(hmm.log_insert_emit, np.tile(np.log2(BACKGROUND_FREQ), (3, 1)))
    assert hmm.log_trans[0, T_BD] == 0
    assert hmm.log_trans[1, T_DD] == 0
    assert hmm.log_trans[2, T_DE] == 0
    for method in (hmm.viterbi, hmm.forward):
        assert method("") == 0
        for seq in ("A", "AA", "C"):
            assert method(seq) == NEG_INF


@pytest.mark.parametrize("rows", [["AC", "A"], ["A", "AC"], ["AC", "G", "EYY"]])
def test_build_from_msa_rejects_ragged_rows_before_batching(
    rows: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(profile_module, "_MAX_BATCH_CELLS", 1)
    with pytest.raises(ValueError, match="equal lengths"):
        ProfileHMM.build_from_msa(MSA([], rows))


def test_build_from_msa_uses_occupancy_not_identity_for_match_columns() -> None:
    """Different residues still share match positions when no rows have gaps."""
    rows = [
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD",
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
    ]

    assert match_column_mask(rows).tolist() == [True] * 44
    hmm = _build(rows, symfrac=0.5, pseudocount=1.0)
    assert hmm.K == 44

    # All paths are B -> M1 -> ... -> M44, emitting A, D, or C throughout.
    # Each match state observes A=1, D=1, C=1. Adding one pseudocount
    # per amino acid gives 23 total counts: 2 for A/C/D, 1 for each other.
    expected_match = np.full(20, 1 / 23)
    expected_match[[AA_INDEX[aa] for aa in "ACD"]] = 2 / 23
    np.testing.assert_allclose(2**hmm.log_match_emit, np.tile(expected_match, (44, 1)))

    # No insert residues were observed; all 45 slots retain the background.
    np.testing.assert_allclose(2**hmm.log_insert_emit, np.tile(BACKGROUND_FREQ, (45, 1)))
    # Each internal match state observes 3 M->M and no M->I or M->D.
    # With pseudocounts, outgoing counts are [4, 1, 1], summing to 6.
    np.testing.assert_allclose(
        2 ** hmm.log_trans[1:44, [T_MM, T_MI, T_MD]],
        np.tile([4 / 6, 1 / 6, 1 / 6], (43, 1)),
    )


def test_build_from_msa_distinguishes_insert_columns_from_deleted_positions() -> None:
    """A gap skips a match position, but does nothing in an insert column."""
    rows = [
        "AAAAAAAAAAAAAAAAAA.AAAAAAAAA.AAAAAAAAAA.AAAA",
        "DDD.DDDDDDDDDDDDDD.DDDDDDDDDDDDDDDDDDDD.DDDD",
        "CCC.CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC.CCCC",
    ]
    # Alignment columns (1-based):
    #   column       4       19       29       40
    #   residues    A..     ..C      .DC      ...
    #   occupancy   1/3     1/3      2/3      0/3
    #   assignment  I3      I17      M27      I37 (empty)
    mask = match_column_mask(rows, symfrac=0.5)
    assert (np.flatnonzero(~mask) + 1).tolist() == [4, 19, 40]
    assert mask[28]  # Column 29 remains a match column despite A's gap.
    hmm = _build(rows, symfrac=0.5, pseudocount=1.0)
    assert hmm.K == 41

    # A: B -> ... -> M3(A) -> I3(A) -> ... -> M26(A) -> D27 -> M28(A) -> ...
    # D: B -> M1(D) -> ... -> M41(D), ignoring gaps in insert columns.
    # C: B -> ... -> M17(C) -> I17(C) -> M18(C) -> ... -> M41(C).
    # M27 observes only C and D: 2 observations + 20 pseudocounts = 22.
    # Match emission arrays use index k-1, so M27 is at index 26.
    expected_match = np.full(20, 1 / 22)
    expected_match[[AA_INDEX[aa] for aa in "CD"]] = 2 / 22
    np.testing.assert_allclose(2 ** hmm.log_match_emit[26], expected_match)

    # Insert slots start with 20 background-weighted counts. I3 observes
    # one A and I17 one C, so each now has 21 total counts.
    for slot, residue in [(3, "A"), (17, "C")]:
        expected_insert = 20 * BACKGROUND_FREQ.copy()
        expected_insert[AA_INDEX[residue]] += 1
        np.testing.assert_allclose(2 ** hmm.log_insert_emit[slot], expected_insert / 21)
        # Two rows go straight to the next match; one takes the insert.
        np.testing.assert_allclose(
            2 ** hmm.log_trans[slot, [T_MM, T_MI, T_MD]], [3 / 6, 2 / 6, 1 / 6]
        )
        np.testing.assert_allclose(2 ** hmm.log_trans[slot, [T_IM, T_II]], [2 / 3, 1 / 3])

    # At M26, two rows continue to M27; A takes D27, then returns to M28.
    # Transition arrays use source position k directly, without subtracting 1.
    np.testing.assert_allclose(2 ** hmm.log_trans[26, [T_MM, T_MI, T_MD]], [3 / 6, 1 / 6, 2 / 6])
    np.testing.assert_allclose(2 ** hmm.log_trans[27, [T_DM, T_DD]], [2 / 3, 1 / 3])

    # Column 40 contains only gaps: no emissions or transitions are counted.
    # Removing that column entirely leaves the learned profile unchanged.
    without_empty_column = _build([row[:39] + row[40:] for row in rows])
    assert without_empty_column.K == hmm.K
    np.testing.assert_allclose(hmm.log_match_emit, without_empty_column.log_match_emit)
    np.testing.assert_allclose(hmm.log_insert_emit, without_empty_column.log_insert_emit)
    np.testing.assert_allclose(hmm.log_trans, without_empty_column.log_trans)


@pytest.mark.parametrize("batch_cells", [1, 6, 250_000])
def test_build_from_msa_handles_leading_insert_match_and_delete_paths(
    monkeypatch: pytest.MonkeyPatch, batch_cells: int
) -> None:
    """At the start, rows can insert a residue, match M1, or skip it."""
    monkeypatch.setattr(profile_module, "_MAX_BATCH_CELLS", batch_cells)
    rows = ["GAC", ".AC", "..C"]
    # Columns: G.. -> I0, AA. -> M1, CCC -> M2.
    assert match_column_mask(rows).tolist() == [False, True, True]
    hmm = _build(rows, symfrac=0.5, pseudocount=1.0)
    assert hmm.K == 2

    # GAC: virtual M0 -> I0(G) -> M1(A) -> M2(C)
    # .AC: B -> M1(A) -> M2(C)
    # ..C: B -> D1 -> M2(C), emitting nothing at D1.
    expected_insert = 20 * BACKGROUND_FREQ.copy()
    expected_insert[AA_INDEX["G"]] += 1
    np.testing.assert_allclose(2 ** hmm.log_insert_emit[0], expected_insert / 21)
    # Each match starts with 20 pseudocounts (one per amino acid); gaps add nothing.
    # M1 observes 2 A: total 22, A count 3, C count 1.
    # M2 observes 3 C: total 23, A count 1, C count 4.
    np.testing.assert_allclose(2 ** hmm.log_match_emit[:, AA_INDEX["A"]], [3 / 22, 1 / 23])
    np.testing.assert_allclose(2 ** hmm.log_match_emit[:, AA_INDEX["C"]], [1 / 22, 4 / 23])

    # Begin has one observation per alternative and one pseudocount each.
    np.testing.assert_allclose(2 ** hmm.log_trans[0, [T_BM, T_MI, T_BD]], [1 / 3] * 3)
    # I0->M1 and D1->M2 each occur once; their alternative loops occur zero times.
    np.testing.assert_allclose(2 ** hmm.log_trans[0, [T_IM, T_II]], [2 / 3, 1 / 3])
    np.testing.assert_allclose(2 ** hmm.log_trans[1, [T_DM, T_DD]], [2 / 3, 1 / 3])


def test_build_from_msa_handles_internal_match_insert_and_delete_paths() -> None:
    """Inside the profile, rows can continue, insert, or skip a match position."""
    rows = ["AGGCE", "A..CE", "A...E"]
    # Columns: AAA -> M1, G.. -> I1, G.. -> I1, CC. -> M2, EEE -> M3.
    assert match_column_mask(rows).tolist() == [True, False, False, True, True]
    hmm = _build(rows, symfrac=0.5, pseudocount=1.0)
    assert hmm.K == 3

    # AGGCE: M1(A) -> I1(G) -> I1(G) -> M2(C) -> M3(E)
    # A..CE: M1(A) -> M2(C) -> M3(E)
    # A...E: M1(A) -> D2 -> M3(E), emitting nothing at D2.
    expected_insert = 20 * BACKGROUND_FREQ.copy()
    expected_insert[AA_INDEX["G"]] += 2
    np.testing.assert_allclose(2 ** hmm.log_insert_emit[1], expected_insert / 22)
    # M2 observes 2 C; M3 observes 3 E. Include one pseudocount per amino acid.
    np.testing.assert_allclose(2 ** hmm.log_match_emit[1, AA_INDEX["C"]], 3 / 22)
    np.testing.assert_allclose(2 ** hmm.log_match_emit[2, AA_INDEX["E"]], 4 / 23)

    # M1: [MM, MI, MD] observed [1, 1, 1] + one pseudocount each = [2, 2, 2].
    # divided by the outgoing transition counts (plus pseudocount) from M1; 2 (MI) + 2 (MM) + 2 (MD) = 6.
    np.testing.assert_allclose(2 ** hmm.log_trans[1, [T_MM, T_MI, T_MD]], [1 / 3, 1 / 3, 1 / 3])
    # I1: one return to M2 (IM), one loop for the second G (II).
    # divded by the outgoing transition counts (plus pseudocount) from I1; 2 (II) + 2 (IM) = 4.
    np.testing.assert_allclose(2 ** hmm.log_trans[1, [T_IM, T_II]], [1 / 2, 1 / 2])
    # M2: two rows take MM; none take MI or MD. [2, 0, 0] + one pseudocount each = [3, 1, 1].
    # divided by the outgoing transition counts (plus pseudocount) from M2; 3 (MM) + 1 (MI) + 1 (MD) = 5.
    np.testing.assert_allclose(2 ** hmm.log_trans[2, [T_MM, T_MI, T_MD]], [3 / 5, 1 / 5, 1 / 5])
    # D2: one row resumes at M3 (DM); none extend the deletion (DD).
    # Observed [1, 0] + one pseudocount each = [2, 1].
    # divided by the outgoing transition counts (plus pseudocount) from D2 ; 2 (DM) + 1 (DD) = 3.
    np.testing.assert_allclose(2 ** hmm.log_trans[2, [T_DM, T_DD]], [2 / 3, 1 / 3])


def test_emission_rows_are_normalised_probabilities_in_log_space() -> None:
    hmm = _build(["ACDE", "ACDE"], symfrac=0.5, pseudocount=1.0)
    for k in range(hmm.K):
        probs = 2.0 ** hmm.log_match_emit[k]
        assert abs(probs.sum() - 1.0) < 1e-9
    for k in range(hmm.K + 1):
        probs = 2.0 ** hmm.log_insert_emit[k]
        assert abs(probs.sum() - 1.0) < 1e-6


def test_transition_distributions_normalised_per_source_state() -> None:
    hmm = _build(["ACDE", "ACDE"], symfrac=0.5, pseudocount=1.0)
    for k in range(1, hmm.K):
        m_sum = sum(2.0 ** hmm.log_trans[k, t] for t in (T_MM, T_MI, T_MD))
        i_sum = sum(2.0 ** hmm.log_trans[k, t] for t in (T_IM, T_II))
        d_sum = sum(2.0 ** hmm.log_trans[k, t] for t in (T_DM, T_DD))
        assert abs(m_sum - 1.0) < 1e-6
        assert abs(i_sum - 1.0) < 1e-6
        assert abs(d_sum - 1.0) < 1e-6


def test_emission_odds_use_the_requested_state_residue_and_background() -> None:
    hmm = ProfileHMM(
        K=2,
        log_match_emit=np.full((2, 20), -9.0),
        log_insert_emit=np.full((3, 20), -9.0),
        log_trans=np.zeros((3, N_TRANS)),
        log_bg=np.full(20, -4.0),
    )
    a, c = AA_INDEX["A"], AA_INDEX["C"]
    hmm.log_bg[c] = -3.0
    hmm.log_match_emit[0, a] = -5.0
    hmm.log_match_emit[1, c] = -1.0
    hmm.log_insert_emit[0, a] = -4.0
    hmm.log_insert_emit[2, c] = -6.0
    assert hmm._match_odds(0, a) == -1.0
    assert hmm._match_odds(1, c) == 2.0
    assert hmm._insert_odds(0, a) == 0.0
    assert hmm._insert_odds(2, c) == -3.0


@pytest.mark.parametrize(("seq", "expected"), [("", []), ("x-b.z", []), ("aXbzsi", [0, 15, 7])])
def test_init_score_rows_handles_filtered_queries_and_single_match_profiles(
    seq: str, expected: list[int]
) -> None:
    hmm = _build(["A", "A"])
    encoded, match, insert, delete = hmm._init_score_rows(seq)
    assert encoded == expected
    np.testing.assert_array_equal(match, [0.0, NEG_INF])
    np.testing.assert_array_equal(insert, [NEG_INF, NEG_INF])
    # Begin->D1 has probability 1/5, regardless of query length.
    np.testing.assert_array_equal(delete, [NEG_INF, np.log2(1 / 5)])


def test_init_score_rows_encodes_query_and_initialises_scores() -> None:
    """Encode five query residues and initialize scores before consuming any."""
    hmm = _build(["AGGCE", "A..CE", "A...E"])
    encoded, match_scores, insert_scores, delete_scores = hmm._init_score_rows("AAAAA")

    assert encoded == [0, 0, 0, 0, 0]  # A is encoded as 0.
    assert hmm.K == 3
    # Each row indexes profile positions 0..3, independently of query length.
    for scores in (match_scores, insert_scores, delete_scores):
        assert scores.shape == (4,)

    np.testing.assert_array_equal(match_scores, [0.0, NEG_INF, NEG_INF, NEG_INF])
    np.testing.assert_array_equal(insert_scores, np.full(4, NEG_INF))

    # Without consuming residues, B->D1->D2->D3 uses probabilities 1/6, 1/2, 1/3.
    assert delete_scores[0] == NEG_INF
    np.testing.assert_allclose(delete_scores[1:], np.log2([1 / 6, 1 / 12, 1 / 36]))


@pytest.mark.parametrize("method", ["viterbi", "forward"])
def test_scoring_uses_begin_probability_for_first_match(method: str) -> None:
    hmm = _build(["A"] * 3)
    # The only complete path is B->M1(A)->End, including entry and exit.
    expected = np.log2((4 / 6) * (4 / 23) * (4 / 5) / BACKGROUND_FREQ[AA_INDEX["A"]])
    np.testing.assert_allclose(getattr(hmm, method)("A"), expected)


@pytest.mark.parametrize("method", ["viterbi", "forward"])
def test_scoring_preserves_initial_deletion_path(method: str) -> None:
    hmm = _build(["AGGCE", "A..CE", "A...E"])
    # An empty query has exactly one path: B->D1->D2->D3, counted once.
    np.testing.assert_allclose(getattr(hmm, method)(""), np.log2((1 / 6) * (1 / 2) * (1 / 3)))


@pytest.mark.parametrize("method", ["viterbi", "forward"])
def test_scoring_one_residue_includes_leading_deletion_costs(method: str) -> None:
    hmm = _build(["AGGCE", "A..CE", "A...E"])
    background_a = BACKGROUND_FREQ[AA_INDEX["A"]]
    # All complete paths for A: match at one position and delete the other two.
    path_odds = [
        (4 / 6) * (4 / 23 / background_a) * (1 / 3) * (1 / 3),  # B->M1->D2->D3
        (1 / 6) * (1 / 2) * (1 / 22 / background_a) * (1 / 5),  # B->D1->M2->D3
        (1 / 6) * (1 / 2) * (2 / 3) * (1 / 23 / background_a) * (4 / 5),  # B->D1->D2->M3->End
    ]
    expected = np.log2(max(path_odds) if method == "viterbi" else sum(path_odds))
    np.testing.assert_allclose(getattr(hmm, method)("A"), expected)


@pytest.mark.parametrize("method", ["viterbi", "forward"])
def test_scoring_retains_leading_insert_paths(method: str) -> None:
    hmm = _build(["A"] * 3)
    match_g = (1 / 23) / BACKGROUND_FREQ[AA_INDEX["G"]]
    match_a = (4 / 23) / BACKGROUND_FREQ[AA_INDEX["A"]]
    # Inserts emit at background frequencies. Match each of GGA's residues in turn.
    path_odds = [
        (4 / 6) * match_g * (1 / 5) * (1 / 2) * (1 / 2),  # B->M1->I1->I1->End
        (1 / 6) * (1 / 2) * match_g * (1 / 5) * (1 / 2),  # B->I0->M1->I1->End
        (1 / 6) * (1 / 2) * (1 / 2) * match_a * (4 / 5),  # B->I0->I0->M1->End
    ]
    expected = np.log2(max(path_odds) if method == "viterbi" else sum(path_odds))
    np.testing.assert_allclose(getattr(hmm, method)("GGA"), expected)


def test_viterbi_on_training_sequence_is_positive() -> None:
    hmm = _build(["ACDEFGHIK"] * 3)
    score = hmm.viterbi("ACDEFGHIK")
    assert score > 5.0


def test_viterbi_on_random_sequence_is_lower() -> None:
    hmm = _build(["ACDEFGHIK"] * 3)
    good = hmm.viterbi("ACDEFGHIK")
    bad = hmm.viterbi("WWWWWWWWW")
    assert good > bad


def test_viterbi_score_is_finite() -> None:
    hmm = _build(["ACDEFGHIK"] * 2)
    score = hmm.viterbi("ACDE")
    assert np.isfinite(score)


def test_forward_geq_viterbi_on_training_sequence() -> None:
    hmm = _build(["ACDEFGHIK"] * 3)
    v = hmm.viterbi("ACDEFGHIK")
    f = hmm.forward("ACDEFGHIK")
    assert f >= v - 1e-6


def test_forward_geq_viterbi_on_unrelated_sequence() -> None:
    hmm = _build(["ACDEFGHIK"] * 2)
    v = hmm.viterbi("WWWWWWWWW")
    f = hmm.forward("WWWWWWWWW")
    assert f >= v - 1e-6


def test_viterbi_row_consumes_first_residue() -> None:
    hmm = _build(["AGGCE", "A..CE", "A...E"])
    encoded, match, insert, delete = hmm._init_score_rows("A")
    match, insert, delete = hmm._viterbi_row(encoded[0], (match, insert, delete))

    background_a = BACKGROUND_FREQ[AA_INDEX["A"]]
    # Each Match has one reachable path after consuming the first A.
    match_weights = [
        (4 / 6) * (4 / 23 / background_a),  # B -> M1(A)
        (1 / 6) * (1 / 2) * (1 / 22 / background_a),  # B -> D1 -> M2(A)
        (1 / 6) * (1 / 2) * (2 / 3) * (1 / 23 / background_a),  # B -> D1 -> D2 -> M3(A)
    ]
    assert match[0] == NEG_INF
    np.testing.assert_allclose(match[1:], np.log2(match_weights), atol=1e-12, rtol=0)

    # Only I0 can emit before a Match; Begin->I0 contributes 1/6.
    np.testing.assert_allclose(
        insert, [np.log2(1 / 6), NEG_INF, NEG_INF, NEG_INF], atol=1e-12, rtol=0
    )

    # The best deletion paths extend M1(A) -> D2 -> D3, with probability 1/3 per edge.
    np.testing.assert_array_equal(delete[:2], [NEG_INF, NEG_INF])
    expected_delete = np.log2([match_weights[0] / 3, match_weights[0] / 9])
    np.testing.assert_allclose(delete[2:], expected_delete, atol=1e-12, rtol=0)


def _background_profile(K: int) -> ProfileHMM:
    """A profile with background emissions and all transitions initially blocked.
    Each test then enables only the transitions needed for its example."""
    log_bg = np.log2(BACKGROUND_FREQ)
    return ProfileHMM(
        K=K,
        log_match_emit=np.tile(log_bg, (K, 1)),
        log_insert_emit=np.tile(log_bg, (K + 1, 1)),
        log_trans=np.full((K + 1, N_TRANS), NEG_INF),
        log_bg=log_bg,
    )


@pytest.mark.parametrize("method", ["viterbi", "forward"])
def test_scoring_rows_return_new_arrays_and_preserve_previous_scores(method: str) -> None:
    hmm = _background_profile(1)
    hmm.log_trans[0, T_BM] = -1.0
    x, match, insert, delete = hmm._init_score_rows("A")
    previous = (match, insert, delete)
    originals = [row.copy() for row in previous]
    current = getattr(hmm, f"_{method}_row")(x[0], previous)
    # Only B -> M1(A) is possible; all other states remain unreachable.
    np.testing.assert_array_equal(current[0], [NEG_INF, -1.0])
    np.testing.assert_array_equal(current[1:], np.full((2, 2), NEG_INF))
    for row in current:
        row.fill(99.0)
    for actual, original in zip(previous, originals, strict=True):
        np.testing.assert_array_equal(actual, original)


def test_init_score_rows_preserves_blocked_deletion_prefix() -> None:
    hmm = _background_profile(3)
    _, match, insert, delete = hmm._init_score_rows("")
    np.testing.assert_array_equal(match, [0, NEG_INF, NEG_INF, NEG_INF])
    np.testing.assert_array_equal(insert, np.full(4, NEG_INF))
    # Initialization adds transitions without applying a floor or threshold.
    np.testing.assert_array_equal(delete, [NEG_INF, NEG_INF, NEG_INF, NEG_INF])


@pytest.mark.parametrize("K", [0, -1])
def test_boundary_constructor_rejects_profiles_without_match_positions(K: int) -> None:
    with pytest.raises(ValueError, match="at least one match position"):
        ProfileHMM(K, np.empty((0, 20)), np.empty((1, 20)), np.empty((1, N_TRANS)), np.zeros(20))


@pytest.mark.parametrize("method", ["viterbi", "forward"])
def test_scoring_combines_two_tied_complete_paths(method: str) -> None:
    hmm = _background_profile(1)
    hmm.log_trans[0, [T_BM, T_MI, T_IM]] = -1.0
    hmm.log_trans[1, T_MI] = -1.0
    hmm.log_trans[1, [T_ME, T_IE]] = 0.0
    # AA has exactly two complete paths, each with weight 1/4:
    # B -> M1(A) -> I1(A), and B -> I0(A) -> M1(A).
    expected = -2.0 if method == "viterbi" else -1.0
    assert getattr(hmm, method)("AA") == expected


@pytest.mark.parametrize("method", ["viterbi", "forward"])
def test_scoring_merges_match_insert_and_delete_predecessors(method: str) -> None:
    hmm = _background_profile(3)
    hmm.log_trans[0, [T_BM, T_BD]] = -1.0
    hmm.log_trans[0, [T_MI, T_IM]] = 0.0
    hmm.log_trans[1, [T_MM, T_MD]] = -1.0
    hmm.log_trans[1, T_DM] = 0.0
    hmm.log_trans[2, [T_MM, T_MI]] = -1.0
    hmm.log_trans[2, [T_IM, T_DM]] = 0.0
    hmm.log_trans[3, T_ME] = 0.0
    # AAA has exactly three complete paths, all ending at M3:
    # B -> M1(A) -> M2(A) -> M3(A):       1/2 * 1/2 * 1/2 = 1/8
    # B -> D1 -> M2(A) -> I2(A) -> M3(A): 1/2 * 1 * 1/2 * 1 = 1/4
    # B -> I0(A) -> M1(A) -> D2 -> M3(A): 1 * 1 * 1/2 * 1 = 1/2
    expected = -1.0 if method == "viterbi" else np.log2(1 / 8 + 1 / 4 + 1 / 2)
    np.testing.assert_allclose(getattr(hmm, method)("AAA"), expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("method", ["viterbi", "forward"])
def test_scoring_blocked_paths_remain_unreachable(method: str) -> None:
    hmm = _background_profile(3)
    for seq in ("", "A", "AAAA"):
        assert getattr(hmm, method)(seq) == NEG_INF
    # Open only B -> M1 -> D2 -> D3. Other predecessors remain blocked.
    hmm.log_trans[0, T_BM] = -1.0
    hmm.log_trans[1, T_MD] = -2.0
    hmm.log_trans[2, T_DD] = -3.0
    # A path cannot finish until its explicit terminal exit is opened.
    assert getattr(hmm, method)("A") == NEG_INF
    hmm.log_trans[3, T_DE] = 0.0
    assert getattr(hmm, method)("A") == -6.0
    assert getattr(hmm, method)("AA") == NEG_INF


@pytest.mark.parametrize("method", ["viterbi", "forward"])
@pytest.mark.parametrize("entry", [NEG_INF, -1e30, -2.0])
def test_scoring_preserves_first_match_reachability(method: str, entry: float) -> None:
    hmm = _background_profile(1)
    hmm.log_trans[0, T_BM] = entry
    hmm.log_trans[1, T_ME] = 0.0
    assert getattr(hmm, method)("A") == entry


@pytest.mark.parametrize("method", ["viterbi", "forward"])
@pytest.mark.parametrize("transition_score", [NEG_INF, -1e30, -2.0])
def test_scoring_preserves_internal_match_reachability(
    method: str, transition_score: float
) -> None:
    hmm = _background_profile(2)
    # The only complete path is B -> M1(A) -> M2(A) -> End.
    hmm.log_trans[0, T_BM] = 0.0
    hmm.log_trans[1, T_MM] = transition_score
    hmm.log_trans[2, T_ME] = 0.0
    assert getattr(hmm, method)("AA") == transition_score


@pytest.mark.parametrize("method", ["viterbi", "forward"])
@pytest.mark.parametrize("transition_score", [NEG_INF, -1e30, -2.0])
def test_scoring_preserves_insert_reachability(method: str, transition_score: float) -> None:
    hmm = _background_profile(1)
    # The only complete path is B -> M1(A) -> I1(A) -> End.
    hmm.log_trans[0, T_BM] = 0.0
    hmm.log_trans[1, T_MI] = transition_score
    hmm.log_trans[1, T_IE] = 0.0
    assert getattr(hmm, method)("AA") == transition_score


@pytest.mark.parametrize("method", ["viterbi", "forward"])
def test_scoring_without_pseudocounts_preserves_observed_match_path(method: str) -> None:
    hmm = _build(["AC", "AC"], pseudocount=0.0)
    expected = -np.log2(BACKGROUND_FREQ[AA_INDEX["A"]] * BACKGROUND_FREQ[AA_INDEX["C"]])
    np.testing.assert_allclose(getattr(hmm, method)("AC"), expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("pseudocount", [-1.0, np.nan, np.inf, -np.inf])
def test_boundary_rejects_invalid_pseudocount(pseudocount: float) -> None:
    with pytest.raises(ValueError, match="pseudocount must be finite and nonnegative"):
        _build(["A"], pseudocount=pseudocount)


@pytest.mark.parametrize("pseudocount", [0.0, 0.1, 1.0])
def test_boundary_emissions_normalise_with_unobserved_states(pseudocount: float) -> None:
    with np.errstate(all="raise"):
        hmm = _build(["A.", "A."], symfrac=0.0, pseudocount=pseudocount)
    for emissions in (hmm.log_match_emit, hmm.log_insert_emit):
        np.testing.assert_allclose((2**emissions).sum(axis=1), 1)


@pytest.mark.parametrize("pseudocount", [0.0, 0.1, 1.0])
def test_boundary_transitions_normalise_only_allowed_groups(pseudocount: float) -> None:
    with np.errstate(all="raise"):
        hmm = _build(["A.", "A."], symfrac=0.0, pseudocount=pseudocount)
    allowed = np.zeros_like(hmm.log_trans, dtype=bool)
    for k, transitions in _allowed_transition_groups(hmm.K):
        np.testing.assert_allclose((2 ** hmm.log_trans[k, transitions]).sum(), 1)
        allowed[k, transitions] = True
    assert np.all(hmm.log_trans[~allowed] == NEG_INF)
    assert hmm.log_trans[hmm.K, T_DE] == 0


def test_boundary_zero_pseudocount_uses_uniform_fallback_and_exact_zeros() -> None:
    hmm = _build(["A.", "A."], symfrac=0.0, pseudocount=0)
    # The unvisited terminal Match and I0 use uniform distributions.
    np.testing.assert_allclose(2 ** hmm.log_match_emit[1], 1 / 20)
    np.testing.assert_allclose(2 ** hmm.log_trans[0, [T_IM, T_II]], [0.5, 0.5])
    assert hmm.log_match_emit[0, AA_INDEX["C"]] == NEG_INF
    assert hmm.log_trans[0, T_MI] == NEG_INF


@pytest.mark.parametrize("batch_cells", [1, 16, 250_000])
def test_boundary_counts_one_repaired_exit_per_alignment_row(
    monkeypatch: pytest.MonkeyPatch, batch_cells: int
) -> None:
    monkeypatch.setattr(profile_module, "_MAX_BATCH_CELLS", batch_cells)
    # A.G. repairs terminal D2/I2 to M2; A.GG retains one I2; .... ends at D2.
    rows = ["A.G.", "A.GG", "....", "AC..", "AC..", "AC.."]
    _, _, counts = _count_alignment(rows, symfrac=0.5)
    np.testing.assert_array_equal(counts[-1, [T_ME, T_IE, T_DE]], [4, 1, 1])
    assert counts[:, [T_ME, T_IE, T_DE]].sum() == len(rows)
    assert not counts[:-1, [T_ME, T_IE, T_DE]].any()


@pytest.mark.parametrize("method", ["viterbi", "forward"])
def test_boundary_zero_emissions_and_exits_are_impossible_without_warnings(method: str) -> None:
    with np.errstate(all="raise"):
        hmm = _build(["A"], pseudocount=0)
        assert getattr(hmm, method)("C") == NEG_INF
        hmm.log_trans[1, T_ME] = NEG_INF
        assert getattr(hmm, method)("A") == NEG_INF
        assert getattr(hmm, method)("") == NEG_INF


@pytest.mark.parametrize("method", ["viterbi", "forward"])
def test_boundary_filtered_query_uses_complete_empty_path(method: str) -> None:
    hmm = _build(["AGGCE", "A..CE", "A...E"])
    assert getattr(hmm, method)("x-b.z") == pytest.approx(np.log2(1 / 36))


@pytest.mark.parametrize("method", ["viterbi", "forward"])
@pytest.mark.parametrize("exit_probability", [0.1, 0.9])
def test_boundary_exit_weights_change_winning_endpoint(
    method: str, exit_probability: float
) -> None:
    hmm = _background_profile(2)
    hmm.log_trans[0, [T_BM, T_MI, T_BD]] = np.log2([0.3, 0.1, 0.6])
    hmm.log_trans[1, [T_MM, T_MI, T_MD]] = np.log2([0.1, 0.1, 0.8])
    hmm.log_trans[1, [T_DM, T_DD]] = np.log2([0.8, 0.2])
    hmm.log_trans[2, [T_MI, T_ME]] = np.log2([1 - exit_probability, exit_probability])
    hmm.log_trans[2, T_DE] = 0
    # B->M1(A)->D2->End has weight .24. B->D1->M2(A)->End has weight .48*p.
    # Without the exit, M2 always wins; with it, the winner depends on p.
    paths = [0.24, 0.48 * exit_probability]
    expected = max(paths) if method == "viterbi" else sum(paths)
    assert getattr(hmm, method)("A") == pytest.approx(np.log2(expected))


def test_boundary_total_sequence_mass_has_geometric_insertion_tail() -> None:
    """One match followed by optional inserts gives P(A**n) = 2**(-n), n >= 1."""
    hmm = _background_profile(1)
    hmm.log_match_emit[:] = NEG_INF
    hmm.log_insert_emit[:] = NEG_INF
    hmm.log_match_emit[:, AA_INDEX["A"]] = 0
    hmm.log_insert_emit[:, AA_INDEX["A"]] = 0
    hmm.log_trans[0, T_BM] = 0  # Always start at M1, emitting A.
    # At M1 and I1, exit or emit another A with equal probability.
    hmm.log_trans[1, [T_MI, T_ME, T_II, T_IE]] = np.log2(0.5)

    limit = 8
    log_background_a = np.log2(BACKGROUND_FREQ[AA_INDEX["A"]])
    mass = sum(2 ** (hmm.forward("A" * n) + n * log_background_a) for n in range(limit + 1))
    # Undo the background odds above. The omitted geometric tail is exactly 2**(-limit).
    assert mass == pytest.approx(1 - 2 ** (-limit), rel=0, abs=1e-14)
    assert hmm.forward("AACAA") == NEG_INF
