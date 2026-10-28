"""Tests for the amino-acid alphabet and background frequencies."""

import numpy as np
import pytest

from protein_profile_hmm.amino_acids import (
    AA_ALPHABET,
    AA_INDEX,
    BACKGROUND_FREQ,
    _encode_rows,
    encode,
    is_residue,
)


def test_alphabet_has_20_canonical_residues() -> None:
    assert len(AA_ALPHABET) == 20
    assert set(AA_ALPHABET) == set("ACDEFGHIKLMNPQRSTVWY")


def test_index_is_inverse_of_alphabet() -> None:
    for i, aa in enumerate(AA_ALPHABET):
        assert AA_INDEX[aa] == i


def test_background_sums_to_one() -> None:
    assert abs(BACKGROUND_FREQ.sum() - 1.0) < 1e-9


def test_background_has_20_entries() -> None:
    assert BACKGROUND_FREQ.shape == (20,)


def test_background_all_positive() -> None:
    assert (BACKGROUND_FREQ > 0).all()


def test_encode_handles_canonical_residues() -> None:
    assert encode("A") == AA_INDEX["A"]
    assert encode("y") == AA_INDEX["Y"]


def test_encode_returns_minus_one_for_unknown_residue() -> None:
    assert encode("X") == -1
    assert encode("-") == -1
    assert encode(".") == -1


def test_is_residue_filters_alignment_chars() -> None:
    assert is_residue("A") is True
    assert is_residue("y") is True
    assert is_residue("-") is False
    assert is_residue(".") is False
    assert is_residue("X") is False


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        ([], []),
        (["", ""], [[], []]),
        (["ACy", "-.X"], [[0, 1, 19], [-1, -1, -1]]),
        (["aſßéıx", "ACG--."], [[0, 15, -1, -1, 7, -1], [0, 1, 5, -1, -1, -1]]),
    ],
)
def test_encode_rows_preserves_residues_and_coordinates(
    rows: list[str], expected: list[list[int]]
) -> None:
    encoded = _encode_rows(rows)
    assert encoded.dtype == np.int8
    assert encoded.shape == (len(rows), len(rows[0]) if rows else 0)
    assert encoded.tolist() == expected


@pytest.mark.parametrize("rows", [["AC", "A"], ["AC", "G", "EYY"]])
def test_encode_rows_rejects_ragged_alignment(rows: list[str]) -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        _encode_rows(rows)
