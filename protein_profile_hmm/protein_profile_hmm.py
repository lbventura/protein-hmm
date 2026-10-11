"""Glocal profile Hidden Markov Model for protein sequence comparison.

This is a simple re-implementation of HMMER's core
profile-HMM idea for educational purposes: a position-specific HMM with Match/Insert/Delete states per
consensus column, trained from a multiple sequence alignment, and used to
score query sequences against the family.

All emission and transition probabilities are stored in log2 space. Scoring
returns log-odds bit scores relative to a background null model: a positive
bit score means the sequence is more likely under the family model than under
the null.

The allowed transitions are:

- M : Match state; a sequence residue is aligned to consensus position k.
- I : Insert state; an extra sequence residue occurs between consensus positions k and k+1.
- D: Delete state; consensus position k is skipped, with no corresponding sequence residue.
- B: Begin state; a silent (non-counting) entry state into the profile consensus.

Transition layout (slot index k = 0..K):

    index   meaning
      0     M_k -> M_{k+1}    (T_MM)
      1     M_k -> I_k        (T_MI)
      2     M_k -> D_{k+1}    (T_MD)
      3     I_k -> M_{k+1}    (T_IM)
      4     I_k -> I_k        (T_II)
      5     D_k -> M_{k+1}    (T_DM)
      6     D_k -> D_{k+1}    (T_DD)
      7     B   -> M_1        (T_BM, only at k=0)
      8     B   -> D_1        (T_BD, only at k=0)
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .amino_acids import BACKGROUND_FREQ, N_AMINO_ACIDS, encode
from .msa import MSA, match_column_mask

NEG_INF: float = -1e30


def log2sumexp2(values: Iterable[float]) -> float:
    """log2(sum(2**v for v in values)), numerically stable."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return NEG_INF
    m = float(arr.max())
    if m <= NEG_INF / 2:
        return NEG_INF
    return m + float(np.log2(np.sum(2.0 ** (arr - m))))


T_MM = 0
T_MI = 1
T_MD = 2
T_IM = 3
T_II = 4
T_DM = 5
T_DD = 6
T_BM = 7
T_BD = 8
N_TRANS = 9


@dataclass
class ProfileHMM:
    """A glocal profile HMM.

    The profile has ``K`` Match columns (consensus positions). Conceptually
    there are also ``K`` Delete states (silent, one per Match column) and
    ``K + 1`` Insert states (one before column 1, one between each pair of
    consecutive match columns, one after column K).

    All log-prob arrays are in log2 space. Bit scores are computed as
    log-odds relative to ``BACKGROUND_FREQ``.
    """

    K: int
    log_match_emit: np.ndarray
    log_insert_emit: np.ndarray
    log_trans: np.ndarray
    log_bg: np.ndarray

    @classmethod
    def build_from_msa(
        cls,
        msa: MSA,
        symfrac: float = 0.5,
        pseudocount: float = 1.0,
    ) -> ProfileHMM:
        """Build a profile HMM from an MSA by counting + Laplace smoothing.

        Match columns are chosen by the symfrac rule; the remaining columns are
        treated as inserts attached to the nearest preceding match column.
        Emissions and transitions are tallied across all aligned rows, smoothed
        with ``pseudocount`` per cell, normalised, and stored as log2 probs.
        """
        rows = msa.rows
        if not rows:
            raise ValueError("Cannot build profile from empty MSA")

        is_match = match_column_mask(rows, symfrac=symfrac)
        K = int(is_match.sum())
        if K == 0:
            raise ValueError("MSA has no match columns under symfrac")

        # For each alignment column, create the insert slot it belongs to
        # (slot = number of match columns at-or-before this column without it).
        insert_slot_of_col = np.zeros(len(is_match), dtype=int)
        match_index_of_col = np.full(len(is_match), -1, dtype=int)  # k-1 if match
        seen = 0
        for c, m in enumerate(is_match):
            insert_slot_of_col[c] = seen
            if m:
                match_index_of_col[c] = seen
                seen += 1

        match_counts = np.full((K, N_AMINO_ACIDS), pseudocount, dtype=float)
        # Inserts get a smaller pseudocount towards background to keep insert
        # emissions near background (so insert-emit log-odds is near 0).
        insert_counts = np.tile(BACKGROUND_FREQ * N_AMINO_ACIDS, (K + 1, 1)).astype(float)
        trans_counts = np.full((K + 1, N_TRANS), pseudocount, dtype=float)

        for row in rows:
            # Walk the row, building a path of (kind, idx) and bumping counts
            prev_kind = "B"
            prev_k = 0
            for c, ch in enumerate(row):
                aa = encode(ch)
                if is_match[c]:
                    k = match_index_of_col[c] + 1  # 1-based match column
                    if aa >= 0:
                        kind = "M"
                        match_counts[k - 1, aa] += 1.0
                    else:
                        kind = "D"
                    # transition prev -> this
                    if prev_kind == "B":
                        if kind == "M":
                            trans_counts[0, T_BM] += 1.0
                        else:  # D
                            trans_counts[0, T_BD] += 1.0
                    elif prev_kind == "M":
                        if kind == "M":
                            trans_counts[prev_k, T_MM] += 1.0
                        else:
                            trans_counts[prev_k, T_MD] += 1.0
                    elif prev_kind == "I":
                        # I -> M or I -> D (rare); lump D into M
                        trans_counts[prev_k, T_IM] += 1.0
                    elif prev_kind == "D":
                        if kind == "M":
                            trans_counts[prev_k, T_DM] += 1.0
                        else:
                            trans_counts[prev_k, T_DD] += 1.0
                    prev_kind = kind
                    prev_k = k
                else:
                    # Insert column
                    if aa < 0:
                        continue  # gap inside an insert column is a no-op
                    slot = int(insert_slot_of_col[c])
                    insert_counts[slot, aa] += 1.0
                    # transition prev -> I_slot
                    if prev_kind == "B":
                        # B -> I_0 modelled as M_0 -> I_0
                        trans_counts[0, T_MI] += 1.0
                    elif prev_kind == "M":
                        trans_counts[prev_k, T_MI] += 1.0
                    elif prev_kind == "I":
                        trans_counts[prev_k, T_II] += 1.0
                    elif prev_kind == "D":
                        # D -> I: lump into D -> M for simplicity
                        trans_counts[prev_k, T_DM] += 1.0
                    prev_kind = "I"
                    prev_k = slot

        def _normalise_rows(
            counts: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
            sums = counts.sum(axis=1, keepdims=True)
            sums = np.where(sums <= 0, 1.0, sums)
            return np.divide(counts, sums)

        match_probs = _normalise_rows(match_counts)
        insert_probs = _normalise_rows(insert_counts)

        # Transitions are grouped per source state.
        trans_probs = trans_counts.copy()
        for k in range(K + 1):
            groups: list[list[int]] = []
            if k == 0:
                groups.append([T_BM, T_BD])  # Begin's choice
                groups.append([T_MI])  # only M->I from "virtual M_0"
                groups.append([T_IM, T_II])
            elif k < K:
                groups.append([T_MM, T_MI, T_MD])
                groups.append([T_IM, T_II])
                groups.append([T_DM, T_DD])
            else:  # k == K (last column): only loops/self-inserts matter
                groups.append([T_MI])
                groups.append([T_II])
            for grp in groups:
                s = sum(trans_probs[k, t] for t in grp)
                if s > 0:
                    for t in grp:
                        trans_probs[k, t] /= s

        log_match_emit = np.log2(np.clip(match_probs, 1e-300, None))
        log_insert_emit = np.log2(np.clip(insert_probs, 1e-300, None))
        log_trans = np.log2(np.clip(trans_probs, 1e-300, None))
        log_bg = np.log2(np.clip(BACKGROUND_FREQ, 1e-300, None))

        return cls(
            K=K,
            log_match_emit=log_match_emit,
            log_insert_emit=log_insert_emit,
            log_trans=log_trans,
            log_bg=log_bg,
        )

    def _encoded(self, seq: str) -> list[int]:
        out = []
        for ch in seq:
            j = encode(ch)
            if j >= 0:
                out.append(j)
        return out

    def _match_odds(self, k: int, aa: int) -> float:
        return float(self.log_match_emit[k, aa] - self.log_bg[aa])

    def _insert_odds(self, k: int, aa: int) -> float:
        return float(self.log_insert_emit[k, aa] - self.log_bg[aa])

    def _init_dp(self, seq: str) -> tuple[list[int], int, np.ndarray, np.ndarray, np.ndarray]:
        """Encode ``seq`` and allocate the ``(M, Id, D)`` DP matrices.

        Initialises ``M[0, 0] = 0`` and fills the all-delete prefix
        ``D[0, k]`` for ``k ≥ 1``. All other cells start at ``NEG_INF``.

        This supplies scores for leading gaps, allowing the first query
        residue to align to a later profile match position. Pseudocount
        smoothing can give these deletion paths nonzero probabilities even
        when no training row follows them;

        Returns ``(x, L, M, Id, D)``.
        """
        x = self._encoded(seq)
        L = len(x)
        K = self.K

        M = np.full((L + 1, K + 1), NEG_INF)
        Id = np.full((L + 1, K + 1), NEG_INF)
        D = np.full((L + 1, K + 1), NEG_INF)

        M[0, 0] = 0.0
        if K >= 1:
            D[0, 1] = float(self.log_trans[0, T_BD])
            for k in range(2, K + 1):
                D[0, k] = D[0, k - 1] + float(self.log_trans[k - 1, T_DD])

        return x, L, M, Id, D

    def viterbi(self, seq: str) -> float:
        """Return the Viterbi log-odds bit score of ``seq`` under this profile."""
        x, L, M, Id, D = self._init_dp(seq)
        K = self.K
        NI = NEG_INF

        for i in range(0, L + 1):
            for k in range(0, K + 1):
                if i > 0:
                    aa = x[i - 1]
                    if k >= 1:
                        cands = [
                            M[i - 1, k - 1] + float(self.log_trans[k - 1, T_MM]),
                            Id[i - 1, k - 1] + float(self.log_trans[k - 1, T_IM]),
                            D[i - 1, k - 1] + float(self.log_trans[k - 1, T_DM]),
                        ]
                        if k == 1:
                            cands.append(M[i - 1, 0] + float(self.log_trans[0, T_BM]))
                        best = max(cands)
                        if best > NI / 2:
                            M[i, k] = self._match_odds(k - 1, aa) + best
                    cands_i = [
                        M[i - 1, k] + float(self.log_trans[k, T_MI]),
                        Id[i - 1, k] + float(self.log_trans[k, T_II]),
                    ]
                    best_i = max(cands_i)
                    if best_i > NI / 2:
                        Id[i, k] = self._insert_odds(k, aa) + best_i
                # Silent D update (also at i==0, but skip i==0 to preserve init)
                if k >= 1 and i >= 0:
                    cands_d = [
                        M[i, k - 1] + float(self.log_trans[k - 1, T_MD]),
                        D[i, k - 1] + float(self.log_trans[k - 1, T_DD]),
                    ]
                    if k == 1:
                        cands_d.append(M[i, 0] + float(self.log_trans[0, T_BD]))
                    best_d = max(cands_d)
                    if best_d > D[i, k]:
                        D[i, k] = best_d

        terminal = max(M[L, K], Id[L, K], D[L, K])
        return float(terminal)

    def forward(self, seq: str) -> float:
        """Return the Forward log-odds bit score of ``seq``."""
        x, L, M, Id, D = self._init_dp(seq)
        K = self.K
        NI = NEG_INF

        for i in range(0, L + 1):
            for k in range(0, K + 1):
                if i > 0:
                    aa = x[i - 1]
                    if k >= 1:
                        contribs = [
                            M[i - 1, k - 1] + float(self.log_trans[k - 1, T_MM]),
                            Id[i - 1, k - 1] + float(self.log_trans[k - 1, T_IM]),
                            D[i - 1, k - 1] + float(self.log_trans[k - 1, T_DM]),
                        ]
                        if k == 1:
                            contribs.append(M[i - 1, 0] + float(self.log_trans[0, T_BM]))
                        s = log2sumexp2(contribs)
                        if s > NI / 2:
                            M[i, k] = self._match_odds(k - 1, aa) + s
                    contribs_i = [
                        M[i - 1, k] + float(self.log_trans[k, T_MI]),
                        Id[i - 1, k] + float(self.log_trans[k, T_II]),
                    ]
                    s_i = log2sumexp2(contribs_i)
                    if s_i > NI / 2:
                        Id[i, k] = self._insert_odds(k, aa) + s_i
                if k >= 1:
                    contribs_d = [
                        M[i, k - 1] + float(self.log_trans[k - 1, T_MD]),
                        D[i, k - 1] + float(self.log_trans[k - 1, T_DD]),
                    ]
                    if k == 1:
                        contribs_d.append(M[i, 0] + float(self.log_trans[0, T_BD]))
                    s_d = log2sumexp2(contribs_d)
                    # combine with any pre-existing initial D[i,k] value
                    D[i, k] = log2sumexp2([D[i, k], s_d])

        return float(log2sumexp2([M[L, K], Id[L, K], D[L, K]]))
