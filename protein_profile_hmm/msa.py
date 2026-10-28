"""Parse Stockholm-format multiple sequence alignments (MSAs) and decide which
alignment columns become Match columns of a profile HMM.

## Stockholm format

Stockholm is a column-based MSA format used by HMMER and Pfam.  Each
sequence line has the form::

    <name>  <sequence_row>

Lines starting with ``#`` are annotation or metadata and are skipped;
``//`` terminates the alignment block.  Rows for the same name may be split
across several interleaved blocks; this parser concatenates them.

## Match vs. Insert columns

Profile HMMs represent a protein family with a consensus of **K** positions.
Each position corresponds to a **Match** (M) state with a position-specific
emission distribution.  Between consecutive Match states (and flanking the
first and last) are **Insert** (I) states that absorb extra residues a query
may carry relative to the consensus.

When building a profile from an MSA the column assignment is:

- **Match column** — at least ``symfrac`` (default 0.5) of the column's
  characters correspond to standard aminoacids (e.g,., canonical residues).
  The column represents a consensus position occupied by a canonical residue in at least symfrac of the sequences;
  the residues need not be identical or similar.
- **Insert column** — fewer than ``symfrac`` characters are canonical
  (the column is dominated by gaps).  Residues here are absorbed by the
  nearest preceding Insert state and do not consume a Match position.

This is HMMER's ``--symfrac`` convention.
"""

from dataclasses import dataclass
from typing import TextIO

import numpy as np

from .amino_acids import _encode_rows


@dataclass
class MSA:
    """A multiple sequence alignment: parallel lists of names and rows."""

    names: list[str]
    rows: list[str]

    @property
    def n_seqs(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return len(self.rows[0]) if self.rows else 0


def parse_stockholm(stream: TextIO) -> MSA:
    """Parse a Stockholm-format alignment, concatenating any blocked rows."""
    name_order: list[str] = []
    seqs: dict[str, list[str]] = {}

    for raw in stream:
        line = raw.rstrip("\n")
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("//"):
            break
        if stripped.startswith("#"):
            continue
        parts = stripped.split(None, 1)
        if len(parts) != 2:
            continue
        name, chunk = parts
        if name not in seqs:
            seqs[name] = []
            name_order.append(name)
        seqs[name].append(chunk)

    rows = ["".join(seqs[n]) for n in name_order]
    return MSA(names=name_order, rows=rows)


def match_column_mask(rows: list[str], symfrac: float = 0.5) -> np.ndarray:
    """Boolean mask: True at columns that should become Match columns.

    A column is a Match column iff at least ``symfrac`` of its characters are
    canonical residues (e.g., one of the 20 aminoacids).
    """
    if not rows:
        return np.zeros(0, dtype=bool)
    encoded = _encode_rows(rows)
    occupancy: np.ndarray = np.count_nonzero(encoded >= 0, axis=0)
    return occupancy / len(rows) >= symfrac
