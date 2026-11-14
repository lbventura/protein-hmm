"""Glocal profile Hidden Markov Model for protein sequence comparison.

This is a simple re-implementation of HMMER's core
profile-HMM idea for educational purposes: a position-specific HMM with Match/Insert/Delete states per
consensus column, trained from a multiple sequence alignment, and used to
score query sequences against the family.

All emission and transition probabilities are stored in log2 space. Scoring
returns complete-path log-odds bit scores relative to the product of background
residue frequencies conditional on query length. Paths containing a zero-probability
transition or emission are impossible under the model. A query scores -inf when
no possible complete path can emit it.

The allowed transitions are:

- M : Match state; a sequence residue is aligned to consensus position k.
- I : Insert state; an extra sequence residue occurs between consensus positions k and k+1.
- D: Delete state; consensus position k is skipped, with no corresponding sequence residue.
- B: Begin state; a silent entry state into the profile consensus.
- End: Silent absorbing destination after the complete query has been consumed.

Transition layout (slot index k = 0..K):

    index   meaning
      0     M_k -> M_{k+1}    (Transition.MM)
      1     M_k -> I_k        (Transition.MI)
      2     M_k -> D_{k+1}    (Transition.MD)
      3     I_k -> M_{k+1}    (Transition.IM)
      4     I_k -> I_k        (Transition.II)
      5     D_k -> M_{k+1}    (Transition.DM)
      6     D_k -> D_{k+1}    (Transition.DD)
      7     B   -> M_1        (Transition.BM, only at k=0)
      8     B   -> D_1        (Transition.BD, only at k=0)
      9     M_K -> End        (Transition.ME, only at k=K)
     10     I_K -> End        (Transition.IE, only at k=K)
     11     D_K -> End        (Transition.DE, only at k=K)

At slot zero Transition.MI denotes B -> I0 and competes with Transition.BM and Transition.BD.
Terminal Match and Insert states choose between insertion and End; D_K
exits with probability one. End is silent and has no emission or DP row.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .amino_acids import BACKGROUND_FREQ, N_AMINO_ACIDS, encode
from .msa import MSA
from .traces import (
    NEG_INF,
    Transition,
    _count_alignment,
    _log_probabilities,
    _normalise_rows,
    _normalise_transitions,
    log2sumexp2,
)


@dataclass
class ProfileHMM:
    """A glocal profile HMM.

    The profile requires ``K >= 1`` Match columns (consensus positions). Conceptually
    there are also ``K`` Delete states (silent, one per Match column) and
    ``K + 1`` Insert states (one before column 1, one between each pair of
    consecutive match columns, one after column K).

    Begin chooses between M1, I0, and D1. Terminal M_K/I_K have explicit
    competing exits to silent End, while D_K exits with probability one.
    Direct construction must supply all 12 transition columns and exits.
    All log-probability arrays are in log2 space, with bit scores using the ``BACKGROUND_FREQ`` residue product conditional on length.
    """

    K: int
    log_match_emit: npt.NDArray[np.float64]
    log_insert_emit: npt.NDArray[np.float64]
    log_trans: npt.NDArray[np.float64]
    log_bg: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.K < 1:
            raise ValueError("ProfileHMM requires at least one match position (K >= 1)")

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
        Each aligned sequence defines a training path. Adjacent I/D events
        are repaired into Match events before array operations count residues
        and transitions, including one exit per row, from the shared paths.
        Match emissions and allowed outgoing alternatives get the same finite,
        nonnegative ``pseudocount`` per alternative; inserts get 20 background-
        weighted counts in total. With ``pseudocount=0``, match emission and
        outgoing transition distributions use observed frequencies. If a
        distribution has no observations, its allowed alternatives receive
        equal probabilities; otherwise, unobserved alternatives have probability
        zero. Disallowed transitions always have probability zero.
        Large alignments use row batches with one global match mask.
        Probabilities are stored in log2 space.
        """
        if not np.isfinite(pseudocount) or pseudocount < 0:
            raise ValueError("pseudocount must be finite and nonnegative")
        match_counts, insert_counts, trans_counts = _count_alignment(msa.rows, symfrac)
        K = len(match_counts)

        # Add priors once, after combining all batches. Inserts retain their
        # fixed total of 20 background-weighted counts, independent of pseudocount.
        match_probs = _normalise_rows(match_counts.astype(float) + pseudocount)
        insert_probs = _normalise_rows(insert_counts + N_AMINO_ACIDS * BACKGROUND_FREQ)
        trans_probs = _normalise_transitions(trans_counts.astype(float), pseudocount)

        log_match_emit = _log_probabilities(match_probs)
        log_insert_emit = _log_probabilities(insert_probs)
        log_trans = _log_probabilities(trans_probs)
        log_bg = _log_probabilities(BACKGROUND_FREQ)

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

    def _init_score_rows(
        self, seq: str
    ) -> tuple[
        list[int], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        """Encode the query and initialize Match/Insert/Delete rows of shape (K+1,).

        M0 represents Begin at score zero. The Delete row contains the complete
        leading deletion path; all other entries start at NEG_INF.

        For a profile built from ["AGGCE", "A..CE", "A...E"], the training paths are:

            Row 1: AGGCE → B → M1(A) → I1(G) → I1(G) → M2(C) → M3(E)
            Row 2: A..CE → B → M1(A) ────────────────→ M2(C) → M3(E)
            Row 3: A...E → B → M1(A) ────────────────→ D2    → M3(E)

        Query AAAA initializes these score rows before consuming any residues:

            encoded: [0, 0, 0, 0]
            Match:   [0, NEG_INF, NEG_INF, NEG_INF]
            Insert:  [NEG_INF, NEG_INF, NEG_INF, NEG_INF]
            Delete:  [NEG_INF, log2(1/6), log2(1/12), log2(1/36)]

        B -> D1: path probability of 1/6
        B -> D1 -> D2: 1/6 * 1/2 = 1/12
        B -> D1 -> D2 -> D3: 1/6 * 1/2 * 1/3 = 1/36

        For more details, see the "Transition Probabilities" section of protein_profile_hmm/viterbi_first_residue_walkthrough.md.
        """
        x = []
        for ch in seq:
            aa = encode(ch)
            if aa >= 0:
                x.append(aa)
        K = self.K
        match = np.full(K + 1, NEG_INF)
        insert = np.full(K + 1, NEG_INF)
        delete = np.full(K + 1, NEG_INF)
        match[0] = 0.0
        delete[1] = float(self.log_trans[0, Transition.BD])
        for k in range(2, K + 1):
            delete[k] = delete[k - 1] + float(self.log_trans[k - 1, Transition.DD])
        return x, match, insert, delete

    def _viterbi_row(
        self,
        aa: int,
        previous: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return one query residue's Match/Insert/Delete scores using best paths.

        Return three new arrays of shape (K+1,), indexed by profile position.

        Example: Training sequences

                ["AGGCE", "A..CE", "A...E"],

        with query AAAA. Before consuming the query's first A, previous contains the
        rows from _init_score_rows (M0 represents Begin):

            position:   0        1        2        3
            Match:      0     NEG_INF  NEG_INF  NEG_INF
            Insert:  NEG_INF  NEG_INF  NEG_INF  NEG_INF
            Delete:  NEG_INF   -2.585   -3.585   -5.170

        Calling _viterbi_row(AA_INDEX["A"], previous) returns the following
        rows:

            position:   0        1        2        3
            Match:   NEG_INF   0.570   -4.365   -5.015
            Insert:  -2.585   NEG_INF  NEG_INF  NEG_INF
            Delete:  NEG_INF NEG_INF  -1.015   -2.600

        Each candidate extends a predecessor's best path by adding the
        edge's log2 transition probability. Match and Insert also add
        log2(P(A | destination) / P(A | background)) for emitting this A.
        The choices follow the transition layout at the top of this module:

            destination    predecessors                 transitions
            Match[k>=2]    previous M/I/D at k-1        Transition.MM / Transition.IM / Transition.DM
            Insert[k]      previous M/I at k            Transition.MI / Transition.II
            Delete[k>=2]   new M/D at k-1               Transition.MD / Transition.DD

        In the first row of the matrix above, M1 is reached from Begin. M2 and M3 can only
        come from previous D1 and D2: previous M1/M2 and I1/I2 are unreachable.
        For example, with b = P(A | background), M2 is:

            B -> D1 -> M2(A): log2(1/6) + log2(1/2) + log2((1/22)/b)
                             = -4.365

        Among Insert states, only I0 is reachable, via the virtual M0 -> I0
        edge. Insert reads previous Match scores, so the new M1(A) cannot
        feed I1 until another residue is consumed. Delete emits nothing and uses the
        newly computed row: M1(A) -> D2 -> D3 gives -1.015 and -2.600.
        At D3, this beats the alternative new M2(A) -> D3 path (-6.687).
        For more details, see the "i = 1, k = 3" section of protein_profile_hmm/viterbi_first_residue_walkthrough.md.
        """
        previous_match, previous_insert, previous_delete = previous

        K = self.K
        match_scores = np.full(K + 1, NEG_INF)
        insert_scores = np.full(K + 1, NEG_INF)
        delete_scores = np.full(K + 1, NEG_INF)

        trans = self.log_trans
        match_odds = self.log_match_emit[:, aa] - self.log_bg[aa]
        insert_odds = self.log_insert_emit[:, aa] - self.log_bg[aa]

        # Begin and I0 are the only predecessors of M1.
        best = max(
            previous_match[0] + trans[0, Transition.BM],
            previous_insert[0] + trans[0, Transition.IM],
        )
        match_scores[1] = match_odds[0] + best

        # Source positions 1..K-1 feed destination matches 2..K.
        from_match = previous_match[1:K] + trans[1:K, Transition.MM]
        from_insert = previous_insert[1:K] + trans[1:K, Transition.IM]
        from_delete = previous_delete[1:K] + trans[1:K, Transition.DM]
        best_match = np.maximum(np.maximum(from_match, from_insert), from_delete)
        match_scores[2:] = match_odds[1:] + best_match

        from_match = previous_match + trans[:, Transition.MI]
        from_insert = previous_insert + trans[:, Transition.II]
        best_insert = np.maximum(from_match, from_insert)
        insert_scores[:] = insert_odds + best_insert

        # D1 is only reachable before consuming residues.
        # Computing Dk requires the update to D(k-1).
        for k in range(2, K + 1):
            from_match = match_scores[k - 1] + trans[k - 1, Transition.MD]
            from_delete = delete_scores[k - 1] + trans[k - 1, Transition.DD]
            delete_scores[k] = max(from_match, from_delete)

        return match_scores, insert_scores, delete_scores

    def _forward_row(
        self,
        aa: int,
        previous: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return one query residue's scores by summing path weights in log2 space.

        previous contains Match/Insert/Delete rows for the query prefix before
        this residue. Each entry is the log2 sum of the weights of all paths
        ending at that state. Return three new arrays of shape (K+1,), indexed
        by profile position.

        Each candidate adds an edge's log2 transition probability to its
        predecessor's score. The choices follow the transition layout at the
        top of this module, as in _viterbi_row:

            destination    predecessors                 transitions
            Match[k>=2]    previous M/I/D at k-1        Transition.MM / Transition.IM / Transition.DM
            Insert[k]      previous M/I at k            Transition.MI / Transition.II
            Delete[k>=2]   new M/D at k-1               Transition.MD / Transition.DD

        For Match, nested np.logaddexp2 calls sum the contributions
        from M, I, and D into total_match. Match and Insert then add the
        emission log-odds log2(P(aa | destination) / P(aa | background)).
        These operations run across profile positions using array slices.
        M1 is handled separately: its predecessors are Begin (stored in M0)
        via Transition.BM and previous I0 via Transition.IM.

        The score itself uses total_match or total_insert, summing all predecessor paths.

        Dk (k >= 2) can be reached from the Match or Delete state at the
        preceding profile position. Because neither transition consumes a residue,
        both predecessors belong to the same query row and no emission odds are added.

        Example: train on ["AGGCE", "A..CE", "A...E"] and score query AAAA.
        Before the first A, previous contains the rows from _init_score_rows:

            position:   0        1        2        3
            Match:      0     NEG_INF  NEG_INF  NEG_INF
            Insert:  NEG_INF  NEG_INF  NEG_INF  NEG_INF
            Delete:  NEG_INF   -2.585   -3.585   -5.170

        Calling _forward_row(AA_INDEX["A"], previous) returns:

            position:   0        1        2        3
            Match:   NEG_INF   0.570   -4.365   -5.015
            Insert:  -2.585   NEG_INF  NEG_INF  NEG_INF
            Delete:  NEG_INF NEG_INF  -1.015   -2.517

        Each reachable Match/Insert state and D2 has only one path at this step, so their scores equal
        Viterbi's. D3 combines two paths within the new row:

            via D2: B -> M1(A) -> D2 -> D3    score -2.600
            via M2: B -> D1 -> M2(A) -> D3    score -6.687

        Forward computes logaddexp2(-2.600, -6.687), around -2.517;
        Viterbi takes their maximum, -2.600.

        For the derivation, see "i = 1, k = 3: reach the last profile position" in
        protein_profile_hmm/viterbi_first_residue_walkthrough.md.
        """
        previous_match, previous_insert, previous_delete = previous

        K = self.K
        match_scores = np.full(K + 1, NEG_INF)
        insert_scores = np.full(K + 1, NEG_INF)
        delete_scores = np.full(K + 1, NEG_INF)

        trans = self.log_trans
        match_odds = self.log_match_emit[:, aa] - self.log_bg[aa]
        insert_odds = self.log_insert_emit[:, aa] - self.log_bg[aa]

        from_begin = previous_match[0] + trans[0, Transition.BM]
        from_insert_zero = previous_insert[0] + trans[0, Transition.IM]
        match_scores[1] = match_odds[0] + np.logaddexp2(from_begin, from_insert_zero)

        from_match = previous_match[1:K] + trans[1:K, Transition.MM]
        from_insert = previous_insert[1:K] + trans[1:K, Transition.IM]
        from_delete = previous_delete[1:K] + trans[1:K, Transition.DM]
        total_match = np.logaddexp2(np.logaddexp2(from_match, from_insert), from_delete)
        match_scores[2:] = match_odds[1:] + total_match

        from_match = previous_match + trans[:, Transition.MI]
        from_insert = previous_insert + trans[:, Transition.II]
        total_insert = np.logaddexp2(from_match, from_insert)
        insert_scores[:] = insert_odds + total_insert

        for k in range(2, K + 1):
            from_match = match_scores[k - 1] + trans[k - 1, Transition.MD]
            from_delete = delete_scores[k - 1] + trans[k - 1, Transition.DD]
            delete_scores[k] = np.logaddexp2(from_match, from_delete)

        return match_scores, insert_scores, delete_scores

    def _terminal_candidates(
        self,
        rows: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]],
    ) -> tuple[float, float, float]:
        """Extend the final Match/Insert/Delete scores through End exactly once."""
        match, insert, delete = rows
        return (
            float(match[self.K] + self.log_trans[self.K, Transition.ME]),
            float(insert[self.K] + self.log_trans[self.K, Transition.IE]),
            float(delete[self.K] + self.log_trans[self.K, Transition.DE]),
        )

    def viterbi(self, seq: str) -> float:
        """Return the best complete path's log-odds bit score using O(K) storage.

        A complete path consumes all encoded residues and reaches silent End.
        The denominator is the background residue product conditional on length.
        Impossible queries return negative infinity.
        """
        x, match, insert, delete = self._init_score_rows(seq)
        previous = (match, insert, delete)
        for aa in x:
            previous = self._viterbi_row(aa, previous)
        return max(self._terminal_candidates(previous))

    def forward(self, seq: str) -> float:
        """Return all complete paths' log-odds bit score using O(K) storage.

        Sum path probabilities through End, relative to the background residue
        product conditional on length. Impossible queries return negative infinity.
        """
        x, match, insert, delete = self._init_score_rows(seq)
        previous = (match, insert, delete)
        for aa in x:
            previous = self._forward_row(aa, previous)
        return log2sumexp2(self._terminal_candidates(previous))
