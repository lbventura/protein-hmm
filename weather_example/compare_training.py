"""Compare naive vs improved Baum-Welch training on the weather HMM.

Naive    : 1 sequence, 1 random init, no smoothing, unscaled forward/backward
           (identical to what weather_example.py does).
Improved : many sequences from the same generative model, multiple random
           restarts (pick highest log-likelihood), additive Dirichlet smoothing
           in the M-step, and log-space forward/backward to avoid underflow.

Both methods are evaluated by aligning labels (best row permutation) and
computing the L1 distance |A - Â| + |B - B̂| against the true parameters.
"""

import logging
from collections.abc import Sequence
from itertools import permutations

import numpy as np
import numpy.typing as npt

from . import weather_example as wm
from .hmm import HiddenMarkovModel

logger = logging.getLogger(__name__)


def generate_sequences(
    model: HiddenMarkovModel,
    n_seqs: int,
    length: int,
    seed: int,
) -> list[list[int]]:
    """Sample ``n_seqs`` observation sequences of length ``length`` from ``model``."""
    rng = np.random.default_rng(seed)
    return [model.sample(length, rng)[0] for _ in range(n_seqs)]


def _safe_log(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    with np.errstate(divide="ignore"):
        return np.log(x)


def _logsumexp(
    a: npt.NDArray[np.float64], axis: int | None = None
) -> float | npt.NDArray[np.float64]:
    a = np.asarray(a)
    if axis is None:
        flat = a.ravel()
        m = float(np.max(flat))
        if not np.isfinite(m):
            m = 0.0
        return m + float(np.log(np.sum(np.exp(flat - m))))
    a_max = np.max(a, axis=axis, keepdims=True)
    a_max_safe = np.where(np.isfinite(a_max), a_max, 0.0)
    s = np.sum(np.exp(a - a_max_safe), axis=axis, keepdims=True)
    return np.squeeze(np.log(s) + a_max_safe, axis=axis)


def _log_forward_backward(
    log_pi: npt.NDArray[np.float64],
    log_A: npt.NDArray[np.float64],
    log_B: npt.NDArray[np.float64],
    obs: Sequence[int],
) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Numerically stable log-space forward/backward.

    Returns (log_lik, log_alpha, log_beta).
    """
    T = len(obs)
    N = log_pi.shape[0]

    log_alpha = np.full((T, N), -np.inf)
    log_alpha[0] = log_pi + log_B[:, obs[0]]
    for t in range(1, T):
        # log_alpha[t, j] = log_B[j, obs[t]] + logsumexp_i(log_alpha[t-1, i] + log_A[i, j])
        log_alpha[t] = log_B[:, obs[t]] + _logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0)

    log_lik = float(_logsumexp(log_alpha[T - 1]))

    log_beta = np.full((T, N), -np.inf)
    log_beta[T - 1] = 0.0
    for t in range(T - 2, -1, -1):
        # log_beta[t, i] = logsumexp_j(log_A[i, j] + log_B[j, obs[t+1]] + log_beta[t+1, j])
        log_beta[t] = _logsumexp(log_A + (log_B[:, obs[t + 1]] + log_beta[t + 1])[None, :], axis=1)

    return log_lik, log_alpha, log_beta


def baum_welch_improved(
    sequences: Sequence[Sequence[int]],
    N: int,
    M: int,
    n_iter: int = 200,
    tol: float = 1e-5,
    pseudo: float = 0.5,
    init_seed: int = 0,
) -> tuple[HiddenMarkovModel, float]:
    """Multi-sequence Baum-Welch with log-space E-step and Dirichlet smoothing.

    `pseudo` is the per-cell pseudo-count added in the M-step (equivalent to
    a Dirichlet prior with concentration `pseudo + 1`). It keeps parameters
    strictly positive so EM can keep exploring.
    """
    rng = np.random.default_rng(init_seed)
    model = HiddenMarkovModel(
        pi=rng.dirichlet(np.ones(N)),
        A=rng.dirichlet(np.ones(N), size=N),
        B=rng.dirichlet(np.ones(M), size=N),
    )

    prev_total_ll = -np.inf
    for _ in range(n_iter):
        log_pi = _safe_log(model.pi)
        log_A = _safe_log(model.A)
        log_B = _safe_log(model.B)

        pi_num = np.zeros(N)
        A_num = np.zeros((N, N))
        A_den = np.zeros(N)
        B_num = np.zeros((N, M))
        B_den = np.zeros(N)
        total_ll = 0.0

        for obs in sequences:
            log_lik, log_alpha, log_beta = _log_forward_backward(log_pi, log_A, log_B, obs)
            total_ll += log_lik

            # gamma_t(i) = exp(log_alpha[t,i] + log_beta[t,i] - log_lik) ∈ [0, 1]
            gamma = np.exp(log_alpha + log_beta - log_lik)

            # ξ_t(i,j) = exp(log_alpha[t,i] + log_A[i,j]
            #               + log_B[j, o_{t+1}] + log_beta[t+1,j] - log_lik)
            obs_next = np.asarray(obs[1:])
            log_xi = (
                log_alpha[:-1, :, None]
                + log_A[None, :, :]
                + log_B[:, obs_next].T[:, None, :]
                + log_beta[1:, None, :]
                - log_lik
            )
            xi = np.exp(log_xi)

            obs_arr = np.asarray(obs)
            pi_num += gamma[0]
            A_num += xi.sum(axis=0)
            A_den += gamma[:-1].sum(axis=0)
            for k in range(M):
                B_num[:, k] += gamma[obs_arr == k].sum(axis=0)
            B_den += gamma.sum(axis=0)

        # Additive (Dirichlet) smoothing keeps params strictly positive.
        new_pi = pi_num + pseudo
        new_pi /= new_pi.sum()
        new_A = (A_num + pseudo) / (A_den[:, None] + pseudo * N)
        new_B = (B_num + pseudo) / (B_den[:, None] + pseudo * M)

        model = HiddenMarkovModel(new_pi, new_A, new_B)

        if abs(total_ll - prev_total_ll) < tol:
            break
        prev_total_ll = total_ll

    # Recompute log-likelihood under final parameters.
    final_ll = 0.0
    log_pi = _safe_log(model.pi)
    log_A = _safe_log(model.A)
    log_B = _safe_log(model.B)
    for obs in sequences:
        ll, _, _ = _log_forward_backward(log_pi, log_A, log_B, obs)
        final_ll += ll
    return model, final_ll


def best_permutation_distance(
    true_A: npt.NDArray[np.float64],
    true_B: npt.NDArray[np.float64],
    est_A: npt.NDArray[np.float64],
    est_B: npt.NDArray[np.float64],
) -> tuple[float, npt.NDArray[np.intp]]:
    """Find the row permutation of (est_A, est_B) closest to truth in L1."""
    N = true_A.shape[0]
    best: tuple[float, npt.NDArray[np.intp] | None] = (np.inf, None)
    for perm in permutations(range(N)):
        p = np.array(perm)
        pA = est_A[p][:, p]
        pB = est_B[p]
        d = float(np.abs(true_A - pA).sum() + np.abs(true_B - pB).sum())
        if d < best[0]:
            best = (d, p)
    return best  # type: ignore[return-value]


def fmt_matrix(M: npt.NDArray[np.float64], names: Sequence[str], label: str) -> str:
    out = f"  {label}:\n"
    for i, row in enumerate(M):
        out += f"    {names[i]:6s}: " + "  ".join(f"{v:.3f}" for v in row) + "\n"
    return out


def main() -> None:
    logger.info("  Naive vs improved Baum-Welch training on the weather HMM")

    true_model = HiddenMarkovModel(wm.PI_TRUE, wm.A_TRUE, wm.B_TRUE)
    true_A = np.array(wm.A_TRUE)
    true_B = np.array(wm.B_TRUE)

    DATA_SEED = 42
    LENGTH = 100
    N_SEQS_IMPROVED = 20
    N_RESTARTS = 5
    PSEUDO = 0.5

    logger.info(fmt_matrix(true_A, wm.STATE_NAMES, "True A"))
    logger.info(fmt_matrix(true_B, wm.STATE_NAMES, "True B (Walk / Shop / Clean)"))

    # Naive: identical to weather_example.py
    naive_seqs = generate_sequences(true_model, 1, LENGTH, DATA_SEED)
    init_rng = np.random.default_rng(7)
    naive_model = HiddenMarkovModel(
        pi=init_rng.dirichlet([1, 1]),
        A=init_rng.dirichlet([1, 1], size=2),
        B=init_rng.dirichlet([1, 1, 1], size=2),
    )
    naive_model.baum_welch(naive_seqs[0], n_iter=2000)
    d_naive, perm_naive = best_permutation_distance(true_A, true_B, naive_model.A, naive_model.B)

    # Improved: many sequences + restarts + smoothing + log-space scaling
    improved_seqs = generate_sequences(true_model, N_SEQS_IMPROVED, LENGTH, DATA_SEED)

    best_model: HiddenMarkovModel | None = None
    best_ll = -np.inf
    for r in range(N_RESTARTS):
        mdl, ll = baum_welch_improved(
            improved_seqs,
            N=2,
            M=3,
            n_iter=500,
            pseudo=PSEUDO,
            init_seed=100 + r,
        )
        if ll > best_ll:
            best_model, best_ll = mdl, ll
    assert best_model is not None
    d_improved, perm_improved = best_permutation_distance(
        true_A, true_B, best_model.A, best_model.B
    )

    logger.info(f"[Naive] 1 sequence (len {LENGTH}), 1 random init, no smoothing, unscaled F/B")
    p = perm_naive
    logger.info(fmt_matrix(naive_model.A[p][:, p], wm.STATE_NAMES, "Learned A (label-aligned)"))
    logger.info(fmt_matrix(naive_model.B[p], wm.STATE_NAMES, "Learned B (label-aligned)"))
    logger.info(f"  L1 distance |A - Â| + |B - B̂| = {d_naive:.4f}\n")

    logger.info(
        f"[Improved] {N_SEQS_IMPROVED} sequences (len {LENGTH}, same seed prefix), "
        f"{N_RESTARTS} restarts,\n"
        f"           pseudo-count={PSEUDO}, log-space forward/backward"
    )
    p = perm_improved
    logger.info(fmt_matrix(best_model.A[p][:, p], wm.STATE_NAMES, "Learned A (label-aligned)"))
    logger.info(fmt_matrix(best_model.B[p], wm.STATE_NAMES, "Learned B (label-aligned)"))
    logger.info(f"  L1 distance |A - Â| + |B - B̂| = {d_improved:.4f}\n")

    if d_improved > 0:
        logger.info(
            f"  Error reduction: {d_naive / d_improved:.1f}x lower L1 distance "
            f"(naive {d_naive:.3f} → improved {d_improved:.3f})"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
