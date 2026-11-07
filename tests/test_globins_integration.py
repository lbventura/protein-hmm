"""End-to-end test: profile HMM built from globins4.sto should rank globin
sequences higher than the non-globin 7LESS_DROME tyrosine kinase."""

from pathlib import Path

import pytest

from protein_profile_hmm.fasta_parser import parse_fasta, read_first
from protein_profile_hmm.msa import parse_stockholm
from protein_profile_hmm.protein_profile_hmm import ProfileHMM

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def globin_profile() -> ProfileHMM:
    with open(DATA / "globins4.sto") as f:
        msa = parse_stockholm(f)
    return ProfileHMM.build_from_msa(msa)


@pytest.mark.parametrize("method", ["viterbi", "forward"])
def test_HBB_HUMAN_scores_positively(globin_profile: ProfileHMM, method: str) -> None:
    _, seq = read_first(DATA / "HBB_HUMAN")
    score = getattr(globin_profile, method)(seq)
    assert score > 0, f"HBB_HUMAN {method} score should be positive, got {score:.2f}"


@pytest.mark.parametrize("method", ["viterbi", "forward"])
def test_7LESS_DROME_scores_lower_than_HBB_HUMAN(globin_profile: ProfileHMM, method: str) -> None:
    _, hbb = read_first(DATA / "HBB_HUMAN")
    _, seven = read_first(DATA / "7LESS_DROME.fa")
    score = getattr(globin_profile, method)
    pos = score(hbb)
    neg = score(seven)
    assert pos > neg, f"globin ({pos:.2f}) should outscore non-globin ({neg:.2f})"


@pytest.mark.parametrize("method", ["viterbi", "forward"])
def test_globin_family_median_outscores_negative_control(
    globin_profile: ProfileHMM, method: str
) -> None:
    with open(DATA / "globins45.fa") as f:
        globins = list(parse_fasta(f))
    _, neg_seq = read_first(DATA / "7LESS_DROME.fa")
    score = getattr(globin_profile, method)
    neg_score = score(neg_seq)
    pos_scores = [score(seq) for _, seq in globins]
    median = sorted(pos_scores)[len(pos_scores) // 2]
    assert median > neg_score, (
        f"median globin score {median:.2f} should exceed 7LESS_DROME {neg_score:.2f}"
    )
