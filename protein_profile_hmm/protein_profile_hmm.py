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

from .amino_acids import BACKGROUND_FREQ, N_AMINO_ACIDS, _alignment_width, _encode_rows, encode
from .msa import MSA

NEG_INF: float = -1e30
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

# Limit event/index temporaries to one batch; a single wider row is still allowed.
_MAX_BATCH_CELLS = 250_000
_BEGIN, _MATCH, _INSERT, _DELETE = range(4)
# Source rows and destination columns are B, M, I, D; edges into B are excluded.
_TRANSITION_TYPES = np.array(
    [
        [-1, T_BM, T_MI, T_BD],
        [-1, T_MM, T_MI, T_MD],
        [-1, T_IM, T_II, T_IM],  # Preserve I->D folding into I->M.
        [-1, T_DM, T_DM, T_DD],  # Preserve D->I folding into D->M.
    ],
    dtype=np.intp,
)


def log2sumexp2(values: Iterable[float]) -> float:
    """log2(sum(2**v for v in values)), numerically stable."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return NEG_INF
    m = float(arr.max())
    if m <= NEG_INF / 2:
        return NEG_INF
    return m + float(np.log2(np.sum(2.0 ** (arr - m))))


def _count_emissions(
    encoded: npt.NDArray[np.int8], columns: np.ndarray, slots: np.ndarray, n_slots: int
) -> npt.NDArray[np.int64]:
    """Histogram (profile slot, residue) pairs without a one-hot array."""
    rows, cols = np.nonzero((encoded >= 0) & columns)
    keys = N_AMINO_ACIDS * slots[cols] + encoded[rows, cols]
    return np.bincount(keys, minlength=n_slots * N_AMINO_ACIDS).reshape(n_slots, N_AMINO_ACIDS)


def _count_transitions(
    encoded: npt.NDArray[np.int8], is_match: np.ndarray, positions: np.ndarray
) -> npt.NDArray[np.int64]:
    """Count adjacent state events, skipping gaps only in insert columns."""
    n_rows, width = encoded.shape
    occupied = encoded >= 0
    # Each row starts with its own Begin; -1 marks a cell that emits no event.
    events = np.full((n_rows, width + 1), _BEGIN, dtype=np.int8)
    events[:, 1:] = np.where(
        is_match, np.where(occupied, _MATCH, _DELETE), np.where(occupied, _INSERT, -1)
    )
    flat = events.ravel()
    indices = np.flatnonzero(flat >= 0)
    kinds = flat[indices]
    slots = np.r_[0, positions][indices % (width + 1)]
    # Exclude the artificial edge from the end of one row to the next Begin.
    within_row = kinds[1:] != _BEGIN
    types = _TRANSITION_TYPES[kinds[:-1][within_row], kinds[1:][within_row]]
    keys = N_TRANS * slots[:-1][within_row] + types
    n_slots = int(is_match.sum()) + 1
    return np.bincount(keys, minlength=n_slots * N_TRANS).reshape(n_slots, N_TRANS)


def _count_profile(
    encoded: npt.NDArray[np.int8], is_match: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return observed match, insert, and transition counts without priors."""
    positions = np.cumsum(is_match, dtype=np.intp)
    K = int(is_match.sum())
    return (
        _count_emissions(encoded, is_match, positions - 1, K),
        _count_emissions(encoded, ~is_match, positions, K + 1),
        _count_transitions(encoded, is_match, positions),
    )


def _normalise_rows(counts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    sums = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, np.where(sums <= 0, 1.0, sums))


def _normalise_transitions(counts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Normalize each source's alternatives, retaining current boundary rules."""
    probs = counts.copy()
    groups = (
        probs[0:1, T_BM : T_BD + 1],  # Begin's M/D choice.
        probs[0:1, T_MI : T_MI + 1],  # Virtual M0->I0.
        probs[0:1, T_IM : T_II + 1],
        probs[1:-1, T_MM : T_MD + 1],  # Internal M, I, and D sources.
        probs[1:-1, T_IM : T_II + 1],
        probs[1:-1, T_DM : T_DD + 1],
        probs[-1:, T_MI : T_MI + 1],  # Terminal insertion and self-loop.
        probs[-1:, T_II : T_II + 1],
    )
    for group in groups:
        sums = group.sum(axis=1, keepdims=True)
        np.divide(group, sums, out=group, where=sums > 0)
    return probs


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
        Observations are counted with array histograms. Match emissions and
        transitions get ``pseudocount`` per cell; inserts get 20 background-
        weighted counts in total. Large alignments use row batches with one
        global match mask. Probabilities are stored in log2 space.
        """
        rows = msa.rows
        if not rows:
            raise ValueError("Cannot build profile from empty MSA")

        width = _alignment_width(rows)
        batch_rows = max(1, _MAX_BATCH_CELLS // max(width, 1))
        encoded = None
        if len(rows) <= batch_rows:
            encoded = _encode_rows(rows)
            occupancy = np.count_nonzero(encoded >= 0, axis=0)
        else:
            # Column assignments must use all rows, not a mask per batch.
            occupancy = np.zeros(width, dtype=np.int64)
            for start in range(0, len(rows), batch_rows):
                occupancy += np.count_nonzero(
                    _encode_rows(rows[start : start + batch_rows]) >= 0, axis=0
                )
        is_match = occupancy / len(rows) >= symfrac
        K = int(is_match.sum())
        if K == 0:
            raise ValueError("MSA has no match columns under symfrac")

        if encoded is not None:
            match_counts, insert_counts, trans_counts = _count_profile(encoded, is_match)
        else:
            match_counts = np.zeros((K, N_AMINO_ACIDS), dtype=np.int64)
            insert_counts = np.zeros((K + 1, N_AMINO_ACIDS), dtype=np.int64)
            trans_counts = np.zeros((K + 1, N_TRANS), dtype=np.int64)
            for start in range(0, len(rows), batch_rows):
                batch = _encode_rows(rows[start : start + batch_rows])
                match, insert, trans = _count_profile(batch, is_match)
                match_counts += match
                insert_counts += insert
                trans_counts += trans

        # Add priors once, after combining all batches. Inserts retain their
        # fixed total of 20 background-weighted counts, independent of pseudocount.
        match_probs = _normalise_rows(match_counts.astype(float) + pseudocount)
        insert_probs = _normalise_rows(insert_counts + N_AMINO_ACIDS * BACKGROUND_FREQ)
        trans_probs = _normalise_transitions(trans_counts.astype(float) + pseudocount)

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
        x = []
        for ch in seq:
            aa = encode(ch)
            if aa >= 0:
                x.append(aa)
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

        # Row zero already contains the complete leading deletion paths.
        for i in range(1, L + 1):
            aa = x[i - 1]
            for k in range(0, K + 1):
                if k >= 1:
                    if k == 1:
                        # M[0, 0] represents Begin; use BM, not an ordinary MM edge.
                        cands = [
                            M[i - 1, 0] + float(self.log_trans[0, T_BM]),
                            Id[i - 1, 0] + float(self.log_trans[0, T_IM]),
                        ]
                    else:
                        cands = [
                            M[i - 1, k - 1] + float(self.log_trans[k - 1, T_MM]),
                            Id[i - 1, k - 1] + float(self.log_trans[k - 1, T_IM]),
                            D[i - 1, k - 1] + float(self.log_trans[k - 1, T_DM]),
                        ]
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
                # D1 is reachable only before consuming residues, via Begin.
                if k >= 2:
                    cands_d = [
                        M[i, k - 1] + float(self.log_trans[k - 1, T_MD]),
                        D[i, k - 1] + float(self.log_trans[k - 1, T_DD]),
                    ]
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

        # Preserve the initialized deletion prefix without counting it again.
        for i in range(1, L + 1):
            aa = x[i - 1]
            for k in range(0, K + 1):
                if k >= 1:
                    if k == 1:
                        # Begin and I0 are the only predecessors of M1.
                        contribs = [
                            M[i - 1, 0] + float(self.log_trans[0, T_BM]),
                            Id[i - 1, 0] + float(self.log_trans[0, T_IM]),
                        ]
                    else:
                        contribs = [
                            M[i - 1, k - 1] + float(self.log_trans[k - 1, T_MM]),
                            Id[i - 1, k - 1] + float(self.log_trans[k - 1, T_IM]),
                            D[i - 1, k - 1] + float(self.log_trans[k - 1, T_DM]),
                        ]
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
                if k >= 2:
                    contribs_d = [
                        M[i, k - 1] + float(self.log_trans[k - 1, T_MD]),
                        D[i, k - 1] + float(self.log_trans[k - 1, T_DD]),
                    ]
                    D[i, k] = log2sumexp2(contribs_d)

        return float(log2sumexp2([M[L, K], Id[L, K], D[L, K]]))
