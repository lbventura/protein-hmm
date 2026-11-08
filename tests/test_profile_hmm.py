"""Tests for ProfileHMM."""

from io import StringIO

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
    T_DM,
    T_II,
    T_IM,
    T_MD,
    T_MI,
    T_MM,
    ProfileHMM,
    State,
    _count_alignment,
    _count_emissions,
    _count_profile,
    _count_transitions,
    _normalise_rows,
    _normalise_transitions,
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
    np.testing.assert_array_equal(_count_transitions(trace, n_slots=1), np.zeros((1, N_TRANS)))


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
    np.testing.assert_array_equal(counts, expected)


def test_count_transitions_folds_delete_to_insert_into_delete_to_match() -> None:
    # A.GC: B -> M1 -> D2 -> I2 -> M3.
    # Both AC.C rows: B -> M1 -> M2 -> M3.
    encoded = encode_rows(["A.GC", "AC.C", "AC.C"])
    is_match = np.array([True, True, False, True])
    trace = _trace_alignment(encoded, is_match)

    counts = _count_transitions(trace, n_slots=4)

    expected = np.zeros((4, N_TRANS), dtype=np.int64)
    expected[0, T_BM] = 3
    expected[1, T_MD] = 1
    expected[1, T_MM] = 2
    expected[2, T_DM] = 1  # D2 -> I2 uses the existing DM counting rule.
    expected[2, T_IM] = 1
    expected[2, T_MM] = 2
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
    np.testing.assert_array_equal(counts, expected)


def test_count_transitions_folds_insert_to_delete_and_counts_deletion_chains() -> None:
    # AGG..E: B -> M1 -> I1 -> I1 -> D2 -> D3 -> M4.
    # A..CGE: B -> M1 -> M2 -> M3 -> M4.
    encoded = encode_rows(["AGG..E", "A..CGE"])
    is_match = np.array([True, False, False, True, True, True])
    trace = _trace_alignment(encoded, is_match)

    counts = _count_transitions(trace, n_slots=5)

    expected = np.zeros((5, N_TRANS), dtype=np.int64)
    expected[0, T_BM] = 2
    expected[1, [T_MI, T_II, T_IM, T_MM]] = 1  # I1 -> D2 counts as IM.
    expected[2, [T_DD, T_MM]] = 1
    expected[3, [T_DM, T_MM]] = 1
    np.testing.assert_array_equal(counts, expected)


def test_count_transitions_counts_inserts_after_the_last_match_is_deleted() -> None:
    # A.GG: B -> M1 -> D2 -> I2 -> I2; AC..: B -> M1 -> M2.
    encoded = encode_rows(["A.GG", "AC.."])
    is_match = np.array([True, True, False, False])
    trace = _trace_alignment(encoded, is_match)

    counts = _count_transitions(trace, n_slots=3)

    expected = np.zeros((3, N_TRANS), dtype=np.int64)
    expected[0, T_BM] = 2
    expected[1, [T_MD, T_MM]] = 1
    expected[2, [T_DM, T_II]] = 1
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
    np.testing.assert_array_equal(match, expected_match)
    np.testing.assert_array_equal(insert, expected_insert)
    np.testing.assert_array_equal(trans, expected_trans)


@pytest.mark.parametrize("empty", [False, True])
def test_normalise_rows_handles_zero_and_small_counts_without_mutating_input(empty: bool) -> None:
    counts = np.array([[1.0, 3.0], [0.0, 0.0], [1e-200, 3e-200]])
    expected = np.array([[0.25, 0.75], [0.0, 0.0], [0.25, 0.75]])
    if empty:
        counts, expected = counts[:0], expected[:0]
    original = counts.copy()
    actual = _normalise_rows(counts)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(counts, original)
    assert actual.dtype == np.float64


@pytest.mark.parametrize("K", [1, 3])
@pytest.mark.parametrize("scale", [0.0, 1.0])
def test_normalise_transitions_preserves_boundary_rules_and_unused_cells(
    K: int, scale: float
) -> None:
    counts = np.tile([2.0, 3, 5, 7, 11, 13, 17, 19, 23], (K + 1, 1)) * scale
    original = counts.copy()
    expected = (
        np.array(
            [[2, 1, 5, 7 / 18, 11 / 18, 13, 17, 19 / 42, 23 / 42]]
            + [[0.2, 0.3, 0.5, 7 / 18, 11 / 18, 13 / 30, 17 / 30, 19, 23]] * (K - 1)
            + [[2, 1, 5, 7, 1, 13, 17, 19, 23]]
        )
        * scale
    )
    actual = _normalise_transitions(counts)
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
    with pytest.raises(ValueError, match="empty MSA|no match columns"):
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
    log_floor = np.log2(1e-300)
    np.testing.assert_array_equal(hmm.log_match_emit, np.full((2, 20), log_floor))
    np.testing.assert_allclose(hmm.log_insert_emit, np.tile(np.log2(BACKGROUND_FREQ), (3, 1)))
    expected_trans = np.full((3, N_TRANS), log_floor)
    expected_trans[0, T_BD] = 0.0
    expected_trans[1, T_DD] = 0.0
    np.testing.assert_array_equal(hmm.log_trans, expected_trans)


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

    # One direct B->M1 and one B->D1; the leading insert is counted separately.
    # With one pseudocount each, Begin's counts [2, 2] give [1/2, 1/2].
    np.testing.assert_allclose(2 ** hmm.log_trans[0, [T_BM, T_BD]], [1 / 2, 1 / 2])
    np.testing.assert_allclose(2 ** hmm.log_trans[0, T_MI], 1)  # Singleton group.
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


@pytest.mark.parametrize("seq, expected", [("", []), ("x-b.z", []), ("aXbzsi", [0, 15, 7])])
def test_init_score_rows_handles_filtered_queries_and_single_match_profiles(
    seq: str, expected: list[int]
) -> None:
    hmm = _build(["A", "A"])
    encoded, match, insert, delete = hmm._init_score_rows(seq)
    assert encoded == expected
    np.testing.assert_array_equal(match, [0.0, NEG_INF])
    np.testing.assert_array_equal(insert, [NEG_INF, NEG_INF])
    # Begin->D1 has probability 1/4, regardless of query length.
    np.testing.assert_array_equal(delete, [NEG_INF, -2.0])


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

    # Without consuming residues, B->D1->D2->D3 uses probabilities 1/5, 1/2, 1/3.
    assert delete_scores[0] == NEG_INF
    np.testing.assert_allclose(delete_scores[1:], np.log2([1 / 5, 1 / 10, 1 / 30]))


@pytest.mark.parametrize("method", ["viterbi", "forward"])
def test_scoring_uses_begin_probability_for_first_match(method: str) -> None:
    hmm = _build(["A"] * 3)
    # The only complete path for one residue is B->M1(A): Begin contributes 4/5.
    expected = np.log2((4 / 5) * (4 / 23) / BACKGROUND_FREQ[AA_INDEX["A"]])
    np.testing.assert_allclose(getattr(hmm, method)("A"), expected)


@pytest.mark.parametrize("method", ["viterbi", "forward"])
def test_scoring_preserves_initial_deletion_path(method: str) -> None:
    hmm = _build(["AGGCE", "A..CE", "A...E"])
    # An empty query has exactly one path: B->D1->D2->D3, counted once.
    np.testing.assert_allclose(getattr(hmm, method)(""), np.log2((1 / 5) * (1 / 2) * (1 / 3)))


@pytest.mark.parametrize("method", ["viterbi", "forward"])
def test_scoring_one_residue_includes_leading_deletion_costs(method: str) -> None:
    hmm = _build(["AGGCE", "A..CE", "A...E"])
    background_a = BACKGROUND_FREQ[AA_INDEX["A"]]
    # All complete paths for A: match at one position and delete the other two.
    path_odds = [
        (4 / 5) * (4 / 23 / background_a) * (1 / 3) * (1 / 3),  # B->M1->D2->D3
        (1 / 5) * (1 / 2) * (1 / 22 / background_a) * (1 / 5),  # B->D1->M2->D3
        (1 / 5) * (1 / 2) * (2 / 3) * (1 / 23 / background_a),  # B->D1->D2->M3
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
        (4 / 5) * match_g,  # B->M1(G)->I1(G)->I1(A)
        (1 / 2) * match_g,  # B->I0(G)->M1(G)->I1(A)
        (1 / 2) * (1 / 2) * match_a,  # B->I0(G)->I0(G)->M1(A)
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
        (4 / 5) * (4 / 23 / background_a),  # B -> M1(A)
        (1 / 5) * (1 / 2) * (1 / 22 / background_a),  # B -> D1 -> M2(A)
        (1 / 5) * (1 / 2) * (2 / 3) * (1 / 23 / background_a),  # B -> D1 -> D2 -> M3(A)
    ]
    assert match[0] == NEG_INF
    np.testing.assert_allclose(match[1:], np.log2(match_weights), atol=1e-12, rtol=0)

    # Only I0 can emit before a Match; its transition and emission odds are 1.
    np.testing.assert_array_equal(insert, [0.0, NEG_INF, NEG_INF, NEG_INF])

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
    np.testing.assert_array_equal(delete, [NEG_INF, NEG_INF, 2 * NEG_INF, 3 * NEG_INF])


@pytest.mark.parametrize("method", ["viterbi", "forward"])
@pytest.mark.parametrize("seq, expected", [("", 0.0), ("A", -1.0), ("AA", -3.0)])
def test_scoring_zero_match_profile_uses_only_insert_zero(
    method: str, seq: str, expected: float
) -> None:
    hmm = _background_profile(0)
    # B -> I0 has weight 1/2, and each subsequent I0 -> I0 has weight 1/4.
    hmm.log_trans[0, T_MI] = -1.0
    hmm.log_trans[0, T_II] = -2.0
    assert getattr(hmm, method)(seq) == expected


@pytest.mark.parametrize("method", ["viterbi", "forward"])
def test_scoring_combines_two_tied_complete_paths(method: str) -> None:
    hmm = _background_profile(1)
    hmm.log_trans[0, [T_BM, T_MI, T_IM]] = -1.0
    hmm.log_trans[1, T_MI] = -1.0
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
    assert getattr(hmm, method)("A") == -6.0
    assert getattr(hmm, method)("AA") == NEG_INF


@pytest.mark.parametrize("method", ["viterbi", "forward"])
@pytest.mark.parametrize("entry", [NEG_INF, NEG_INF / 2, np.nextafter(NEG_INF / 2, np.inf)])
def test_scoring_applies_emitting_state_reachability_threshold(method: str, entry: float) -> None:
    hmm = _background_profile(1)
    hmm.log_trans[0, T_BM] = entry
    expected = entry if entry > NEG_INF / 2 else NEG_INF
    assert getattr(hmm, method)("A") == expected


@pytest.mark.parametrize("method", ["viterbi", "forward"])
def test_scoring_without_pseudocounts_preserves_observed_match_path(method: str) -> None:
    hmm = _build(["AC", "AC"], pseudocount=0.0)
    expected = -np.log2(BACKGROUND_FREQ[AA_INDEX["A"]] * BACKGROUND_FREQ[AA_INDEX["C"]])
    np.testing.assert_allclose(getattr(hmm, method)("AC"), expected, atol=1e-12, rtol=0)
