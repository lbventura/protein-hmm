"""Amino-acid alphabet, residue indexing, and background frequencies.

This module is the single source of truth for the 20-letter canonical
amino-acid alphabet and the background ("null model") frequencies used to
turn raw log-probabilities into log-odds bit scores.

## The 20 canonical amino acids

Each residue is identified by its IUPAC one-letter code:

    A  Alanine        C  Cysteine       D  Aspartate      E  Glutamate
    F  Phenylalanine  G  Glycine        H  Histidine      I  Isoleucine
    K  Lysine         L  Leucine        M  Methionine     N  Asparagine
    P  Proline        Q  Glutamine      R  Arginine       S  Serine
    T  Threonine      V  Valine         W  Tryptophan     Y  Tyrosine

## Robinson & Robinson background frequencies

Robinson & Robinson (1991) tabulated amino-acid frequencies across a large
curated protein dataset and reported the resulting compositional fractions.
These serve as the null ("background") model: the prior probability that a
random position in a generic protein is each amino acid.  Dividing
position-specific emission probabilities by these values produces log-odds
scores that are positive when a residue is over-represented relative to the
background.

## Swiss-Prot

Swiss-Prot is the manually annotated and reviewed section of UniProtKB (the
Universal Protein Resource Knowledgebase), maintained by the Swiss Institute
of Bioinformatics (SIB) and EMBL-EBI.  Robinson & Robinson derived their
frequencies from an early Swiss-Prot release, making Swiss-Prot composition
the conventional background for profile-HMM scoring.
"""

import numpy as np
import numpy.typing as npt

AA_ALPHABET: str = "ACDEFGHIKLMNPQRSTVWY"
N_AMINO_ACIDS: int = len(AA_ALPHABET)

AA_INDEX: dict[str, int] = {aa: i for i, aa in enumerate(AA_ALPHABET)}

_BG_RAW = {
    "A": 0.0780,  # Alanine
    "C": 0.0190,  # Cysteine
    "D": 0.0530,  # Aspartate
    "E": 0.0630,  # Glutamate
    "F": 0.0390,  # Phenylalanine
    "G": 0.0740,  # Glycine
    "H": 0.0220,  # Histidine
    "I": 0.0530,  # Isoleucine
    "K": 0.0590,  # Lysine
    "L": 0.0910,  # Leucine
    "M": 0.0220,  # Methionine
    "N": 0.0430,  # Asparagine
    "P": 0.0520,  # Proline
    "Q": 0.0420,  # Glutamine
    "R": 0.0510,  # Arginine
    "S": 0.0680,  # Serine
    "T": 0.0590,  # Threonine
    "V": 0.0660,  # Valine
    "W": 0.0130,  # Tryptophan
    "Y": 0.0320,  # Tyrosine
}
_bg = np.array([_BG_RAW[aa] for aa in AA_ALPHABET], dtype=float)
BACKGROUND_FREQ: npt.NDArray[np.float64] = _bg / _bg.sum()


def encode(residue: str) -> int:
    """Return the alphabet index for a single residue character.

    Returns -1 for anything that isn't one of the 20 canonical amino acids
    (gaps, ``X``, ``B``, ``Z``, ``U``, ``.``, ``-``, etc.).
    """
    if not residue or len(residue) != 1:
        return -1
    return AA_INDEX.get(residue.upper(), -1)


def is_residue(c: str) -> bool:
    """True iff ``c`` is one of the 20 canonical amino-acid letters."""
    return encode(c) >= 0


_ASCII_INDEX = np.array([encode(chr(i)) for i in range(128)], dtype=np.int8)


def _alignment_width(rows: list[str]) -> int:
    """Validate rectangular rows before encoding or dividing them into batches."""
    width = len(rows[0]) if rows else 0
    if any(len(row) != width for row in rows):
        raise ValueError("MSA rows must have equal lengths")
    return width


def _encode_rows(rows: list[str]) -> npt.NDArray[np.int8]:
    """Encode aligned rows once; preserve one column per input character."""
    width = _alignment_width(rows)
    text = "".join(rows)
    if text.isascii():
        encoded = _ASCII_INDEX[np.frombuffer(text.encode("ascii"), dtype=np.uint8)]
    else:
        # Whole-string uppercasing can expand Unicode characters (e.g. ß).
        encoded = np.fromiter((encode(ch) for ch in text), dtype=np.int8, count=len(text))
    return encoded.reshape(len(rows), width)
