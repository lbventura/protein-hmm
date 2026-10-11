"""Tests for ProfileHMM."""

from io import StringIO

import numpy as np
import pytest

from protein_profile_hmm.amino_acids import AA_INDEX, BACKGROUND_FREQ
from protein_profile_hmm.msa import match_column_mask, parse_stockholm
from protein_profile_hmm.protein_profile_hmm import (
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


def test_build_from_msa_picks_correct_number_of_match_columns() -> None:
    hmm = _build(["ACDE", "ACDE"], symfrac=0.5, pseudocount=1.0)
    assert hmm.K == 4


def test_build_from_msa_treats_low_residue_columns_as_insert() -> None:
    hmm = _build(["AC.DE", "AC.DE", "AC.DE"], symfrac=0.5, pseudocount=1.0)
    assert hmm.K == 4


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


def test_build_from_msa_handles_leading_insert_match_and_delete_paths() -> None:
    """At the start, rows can insert a residue, match M1, or skip it."""
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


def test_init_dp_encodes_query_and_initialises_score_tables() -> None:
    """Five query residues initialise score tables for a three-position profile."""
    hmm = _build(["AGGCE", "A..CE", "A...E"])
    encoded, query_length, match_scores, insert_scores, delete_scores = hmm._init_dp("AAAAA")

    assert encoded == [0, 0, 0, 0, 0]  # A is encoded as 0.
    assert query_length == 5
    assert hmm.K == 3
    # Rows count consumed residues (0..5); columns count profile positions (0..3).
    for scores in (match_scores, insert_scores, delete_scores):
        assert scores.shape == (6, 4)

    expected_match = np.full((6, 4), NEG_INF)
    expected_match[0, 0] = 0.0  # Start before consuming residues or profile positions.
    np.testing.assert_array_equal(match_scores, expected_match)
    np.testing.assert_array_equal(insert_scores, np.full((6, 4), NEG_INF))

    # Without consuming residues, B->D1->D2->D3 uses probabilities 1/5, 1/2, 1/3.
    expected_delete = np.full((6, 4), NEG_INF)
    expected_delete[0, 1:] = np.log2([1 / 5, 1 / 10, 1 / 30])
    np.testing.assert_allclose(delete_scores, expected_delete)


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
