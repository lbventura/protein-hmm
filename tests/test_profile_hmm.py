"""Tests for ProfileHMM."""

from io import StringIO

import numpy as np
import pytest

import protein_profile_hmm.traces as traces_module
from protein_profile_hmm.amino_acids import AA_INDEX, BACKGROUND_FREQ
from protein_profile_hmm.msa import MSA, match_column_mask, parse_stockholm
from protein_profile_hmm.protein_profile_hmm import ProfileHMM
from protein_profile_hmm.traces import (
    N_TRANS,
    NEG_INF,
    Transition,
    _count_alignment,
)
from tests.profile_helpers import allowed_transition_groups


def _build(rows: list[str], **kwargs: float) -> ProfileHMM:
    """Build a ``ProfileHMM`` from an inline Stockholm MSA over ``rows``."""
    body = "\n".join(f"s{i + 1} {r}" for i, r in enumerate(rows))
    data = f"# STOCKHOLM 1.0\n{body}\n//\n"
    msa = parse_stockholm(StringIO(data))
    return ProfileHMM.build_from_msa(msa, **kwargs)


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
    monkeypatch.setattr(traces_module, "_MAX_BATCH_CELLS", batch_cells)
    actual_counts = _count_alignment(rows, symfrac=0.5)
    for actual, counts in zip(actual_counts, expected_counts, strict=True):
        np.testing.assert_array_equal(actual, counts)
    hmm = _build(rows)
    assert hmm.K == expected.K == 3
    for name in ("log_match_emit", "log_insert_emit", "log_trans", "log_bg"):
        np.testing.assert_allclose(getattr(hmm, name), getattr(expected, name), atol=1e-12, rtol=0)


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
    monkeypatch.setattr(traces_module, "_MAX_BATCH_CELLS", 1)
    with pytest.raises(ValueError, match="no match columns"):
        _build(["...", "---"])


@pytest.mark.parametrize("batch_cells", [1, 250_000])
def test_build_from_msa_handles_all_delete_paths_without_pseudocounts(
    monkeypatch: pytest.MonkeyPatch, batch_cells: int
) -> None:
    monkeypatch.setattr(traces_module, "_MAX_BATCH_CELLS", batch_cells)
    # At symfrac=0 even empty columns are match positions: both paths are B -> D1 -> D2.
    hmm = _build(["..", "--"], symfrac=0.0, pseudocount=0.0)
    assert hmm.K == 2
    np.testing.assert_allclose(hmm.log_match_emit, np.full((2, 20), np.log2(1 / 20)))
    np.testing.assert_allclose(hmm.log_insert_emit, np.tile(np.log2(BACKGROUND_FREQ), (3, 1)))
    assert hmm.log_trans[0, Transition.BD] == 0
    assert hmm.log_trans[1, Transition.DD] == 0
    assert hmm.log_trans[2, Transition.DE] == 0
    for method in (hmm.viterbi, hmm.forward):
        assert method("") == 0
        for seq in ("A", "AA", "C"):
            assert method(seq) == NEG_INF


@pytest.mark.parametrize("rows", [["AC", "A"], ["A", "AC"], ["AC", "G", "EYY"]])
def test_build_from_msa_rejects_ragged_rows_before_batching(
    rows: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(traces_module, "_MAX_BATCH_CELLS", 1)
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
        2 ** hmm.log_trans[1:44, [Transition.MM, Transition.MI, Transition.MD]],
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
            2 ** hmm.log_trans[slot, [Transition.MM, Transition.MI, Transition.MD]],
            [3 / 6, 2 / 6, 1 / 6],
        )
        np.testing.assert_allclose(
            2 ** hmm.log_trans[slot, [Transition.IM, Transition.II]], [2 / 3, 1 / 3]
        )

    # At M26, two rows continue to M27; A takes D27, then returns to M28.
    # Transition arrays use source position k directly, without subtracting 1.
    np.testing.assert_allclose(
        2 ** hmm.log_trans[26, [Transition.MM, Transition.MI, Transition.MD]], [3 / 6, 1 / 6, 2 / 6]
    )
    np.testing.assert_allclose(
        2 ** hmm.log_trans[27, [Transition.DM, Transition.DD]], [2 / 3, 1 / 3]
    )

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
    monkeypatch.setattr(traces_module, "_MAX_BATCH_CELLS", batch_cells)
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
    np.testing.assert_allclose(
        2 ** hmm.log_trans[0, [Transition.BM, Transition.MI, Transition.BD]], [1 / 3] * 3
    )
    # I0->M1 and D1->M2 each occur once; their alternative loops occur zero times.
    np.testing.assert_allclose(
        2 ** hmm.log_trans[0, [Transition.IM, Transition.II]], [2 / 3, 1 / 3]
    )
    np.testing.assert_allclose(
        2 ** hmm.log_trans[1, [Transition.DM, Transition.DD]], [2 / 3, 1 / 3]
    )


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
    np.testing.assert_allclose(
        2 ** hmm.log_trans[1, [Transition.MM, Transition.MI, Transition.MD]], [1 / 3, 1 / 3, 1 / 3]
    )
    # I1: one return to M2 (IM), one loop for the second G (II).
    # divded by the outgoing transition counts (plus pseudocount) from I1; 2 (II) + 2 (IM) = 4.
    np.testing.assert_allclose(
        2 ** hmm.log_trans[1, [Transition.IM, Transition.II]], [1 / 2, 1 / 2]
    )
    # M2: two rows take MM; none take MI or MD. [2, 0, 0] + one pseudocount each = [3, 1, 1].
    # divided by the outgoing transition counts (plus pseudocount) from M2; 3 (MM) + 1 (MI) + 1 (MD) = 5.
    np.testing.assert_allclose(
        2 ** hmm.log_trans[2, [Transition.MM, Transition.MI, Transition.MD]], [3 / 5, 1 / 5, 1 / 5]
    )
    # D2: one row resumes at M3 (DM); none extend the deletion (DD).
    # Observed [1, 0] + one pseudocount each = [2, 1].
    # divided by the outgoing transition counts (plus pseudocount) from D2 ; 2 (DM) + 1 (DD) = 3.
    np.testing.assert_allclose(
        2 ** hmm.log_trans[2, [Transition.DM, Transition.DD]], [2 / 3, 1 / 3]
    )


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
        m_sum = sum(
            2.0 ** hmm.log_trans[k, t] for t in (Transition.MM, Transition.MI, Transition.MD)
        )
        i_sum = sum(2.0 ** hmm.log_trans[k, t] for t in (Transition.IM, Transition.II))
        d_sum = sum(2.0 ** hmm.log_trans[k, t] for t in (Transition.DM, Transition.DD))
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
    hmm.log_trans[0, Transition.BM] = -1.0
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
    hmm.log_trans[0, [Transition.BM, Transition.MI, Transition.IM]] = -1.0
    hmm.log_trans[1, Transition.MI] = -1.0
    hmm.log_trans[1, [Transition.ME, Transition.IE]] = 0.0
    # AA has exactly two complete paths, each with weight 1/4:
    # B -> M1(A) -> I1(A), and B -> I0(A) -> M1(A).
    expected = -2.0 if method == "viterbi" else -1.0
    assert getattr(hmm, method)("AA") == expected


@pytest.mark.parametrize("method", ["viterbi", "forward"])
def test_scoring_merges_match_insert_and_delete_predecessors(method: str) -> None:
    hmm = _background_profile(3)
    hmm.log_trans[0, [Transition.BM, Transition.BD]] = -1.0
    hmm.log_trans[0, [Transition.MI, Transition.IM]] = 0.0
    hmm.log_trans[1, [Transition.MM, Transition.MD]] = -1.0
    hmm.log_trans[1, Transition.DM] = 0.0
    hmm.log_trans[2, [Transition.MM, Transition.MI]] = -1.0
    hmm.log_trans[2, [Transition.IM, Transition.DM]] = 0.0
    hmm.log_trans[3, Transition.ME] = 0.0
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
    hmm.log_trans[0, Transition.BM] = -1.0
    hmm.log_trans[1, Transition.MD] = -2.0
    hmm.log_trans[2, Transition.DD] = -3.0
    # A path cannot finish until its explicit terminal exit is opened.
    assert getattr(hmm, method)("A") == NEG_INF
    hmm.log_trans[3, Transition.DE] = 0.0
    assert getattr(hmm, method)("A") == -6.0
    assert getattr(hmm, method)("AA") == NEG_INF


@pytest.mark.parametrize("method", ["viterbi", "forward"])
@pytest.mark.parametrize("entry", [NEG_INF, -1e30, -2.0])
def test_scoring_preserves_first_match_reachability(method: str, entry: float) -> None:
    hmm = _background_profile(1)
    hmm.log_trans[0, Transition.BM] = entry
    hmm.log_trans[1, Transition.ME] = 0.0
    assert getattr(hmm, method)("A") == entry


@pytest.mark.parametrize("method", ["viterbi", "forward"])
@pytest.mark.parametrize("transition_score", [NEG_INF, -1e30, -2.0])
def test_scoring_preserves_internal_match_reachability(
    method: str, transition_score: float
) -> None:
    hmm = _background_profile(2)
    # The only complete path is B -> M1(A) -> M2(A) -> End.
    hmm.log_trans[0, Transition.BM] = 0.0
    hmm.log_trans[1, Transition.MM] = transition_score
    hmm.log_trans[2, Transition.ME] = 0.0
    assert getattr(hmm, method)("AA") == transition_score


@pytest.mark.parametrize("method", ["viterbi", "forward"])
@pytest.mark.parametrize("transition_score", [NEG_INF, -1e30, -2.0])
def test_scoring_preserves_insert_reachability(method: str, transition_score: float) -> None:
    hmm = _background_profile(1)
    # The only complete path is B -> M1(A) -> I1(A) -> End.
    hmm.log_trans[0, Transition.BM] = 0.0
    hmm.log_trans[1, Transition.MI] = transition_score
    hmm.log_trans[1, Transition.IE] = 0.0
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
    for k, transitions in allowed_transition_groups(hmm.K):
        np.testing.assert_allclose((2 ** hmm.log_trans[k, transitions]).sum(), 1)
        allowed[k, transitions] = True
    assert np.all(hmm.log_trans[~allowed] == NEG_INF)
    assert hmm.log_trans[hmm.K, Transition.DE] == 0


def test_boundary_zero_pseudocount_uses_uniform_fallback_and_exact_zeros() -> None:
    hmm = _build(["A.", "A."], symfrac=0.0, pseudocount=0)
    # The unvisited terminal Match and I0 use uniform distributions.
    np.testing.assert_allclose(2 ** hmm.log_match_emit[1], 1 / 20)
    np.testing.assert_allclose(2 ** hmm.log_trans[0, [Transition.IM, Transition.II]], [0.5, 0.5])
    assert hmm.log_match_emit[0, AA_INDEX["C"]] == NEG_INF
    assert hmm.log_trans[0, Transition.MI] == NEG_INF


@pytest.mark.parametrize("method", ["viterbi", "forward"])
def test_boundary_zero_emissions_and_exits_are_impossible_without_warnings(method: str) -> None:
    with np.errstate(all="raise"):
        hmm = _build(["A"], pseudocount=0)
        assert getattr(hmm, method)("C") == NEG_INF
        hmm.log_trans[1, Transition.ME] = NEG_INF
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
    hmm.log_trans[0, [Transition.BM, Transition.MI, Transition.BD]] = np.log2([0.3, 0.1, 0.6])
    hmm.log_trans[1, [Transition.MM, Transition.MI, Transition.MD]] = np.log2([0.1, 0.1, 0.8])
    hmm.log_trans[1, [Transition.DM, Transition.DD]] = np.log2([0.8, 0.2])
    hmm.log_trans[2, [Transition.MI, Transition.ME]] = np.log2(
        [1 - exit_probability, exit_probability]
    )
    hmm.log_trans[2, Transition.DE] = 0
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
    hmm.log_trans[0, Transition.BM] = 0  # Always start at M1, emitting A.
    # At M1 and I1, exit or emit another A with equal probability.
    hmm.log_trans[1, [Transition.MI, Transition.ME, Transition.II, Transition.IE]] = np.log2(0.5)

    limit = 8
    log_background_a = np.log2(BACKGROUND_FREQ[AA_INDEX["A"]])
    mass = sum(2 ** (hmm.forward("A" * n) + n * log_background_a) for n in range(limit + 1))
    # Undo the background odds above. The omitted geometric tail is exactly 2**(-limit).
    assert mass == pytest.approx(1 - 2 ** (-limit), rel=0, abs=1e-14)
    assert hmm.forward("AACAA") == NEG_INF
