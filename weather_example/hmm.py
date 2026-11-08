"""Hidden Markov Model implementation.

Uses the classic weather example:
  - Hidden states: Sunny (0), Rainy (1)
  - Observations: Walk (0), Shop (1), Clean (2)
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

SeedLike = np.random.Generator | int | None


class HiddenMarkovModel:
    """A discrete Hidden Markov Model.

    Parameters
    ----------
    pi : array-like, shape (N,)
        Initial state probability distribution.
    A : array-like, shape (N, N)
        State transition matrix. A[i, j] = P(s_t=j | s_{t-1}=i).
    B : array-like, shape (N, M)
        Emission matrix. B[i, k] = P(o_t=k | s_t=i).
    """

    pi: npt.NDArray[np.float64]
    A: npt.NDArray[np.float64]
    B: npt.NDArray[np.float64]
    N: int
    M: int

    def __init__(self, pi: npt.ArrayLike, A: npt.ArrayLike, B: npt.ArrayLike) -> None:
        self.pi = np.array(pi, dtype=float)
        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float)
        self.N = self.A.shape[0]
        self.M = self.B.shape[1]

    def _forward_pass(self, observations: Sequence[int]) -> npt.NDArray[np.float64]:
        """Compute alpha[t, i] = P(o_1..o_t, s_t=i) for all t, i."""
        T = len(observations)
        alpha = np.zeros((T, self.N))
        alpha[0] = self.pi * self.B[:, observations[0]]
        for t in range(1, T):
            for j in range(self.N):
                alpha[t, j] = self.B[j, observations[t]] * np.sum(alpha[t - 1] * self.A[:, j])
        return alpha

    def _backward_pass(self, observations: Sequence[int]) -> npt.NDArray[np.float64]:
        """Compute beta[t, i] = P(o_{t+1}..o_T | s_t=i) for all t, i."""
        T = len(observations)
        beta = np.zeros((T, self.N))
        beta[T - 1] = 1.0
        for t in range(T - 2, -1, -1):
            for i in range(self.N):
                beta[t, i] = np.sum(self.A[i] * self.B[:, observations[t + 1]] * beta[t + 1])
        return beta

    def forward(self, observations: Sequence[int]) -> float:
        """Compute the likelihood P(O | model) using the forward algorithm."""
        alpha = self._forward_pass(observations)
        return float(np.sum(alpha[-1]))

    def viterbi(self, observations: Sequence[int]) -> list[int]:
        """Find the most likely hidden state sequence using the Viterbi algorithm."""
        T = len(observations)
        delta = np.zeros((T, self.N))
        psi = np.zeros((T, self.N), dtype=int)

        delta[0] = self.pi * self.B[:, observations[0]]
        psi[0] = 0

        for t in range(1, T):
            for j in range(self.N):
                scores = delta[t - 1] * self.A[:, j]
                psi[t, j] = int(np.argmax(scores))
                delta[t, j] = self.B[j, observations[t]] * np.max(scores)

        states = [int(np.argmax(delta[T - 1]))]
        for t in range(T - 1, 0, -1):
            states.insert(0, int(psi[t, states[0]]))
        return states

    def baum_welch(
        self,
        observations: Sequence[int],
        n_iter: int = 100,
        tol: float = 1e-6,
    ) -> None:
        """Estimate model parameters using the Baum-Welch (EM) algorithm.

        Updates self.pi, self.A, self.B in-place. Maximum `n_iter` EM iterations,
        early-stop when |Δ log-likelihood| < `tol`.
        """
        obs = list(observations)
        obs_arr = np.asarray(obs)
        T = len(obs)
        prev_log_likelihood = -np.inf

        for _ in range(n_iter):
            alpha = self._forward_pass(obs)
            beta = self._backward_pass(obs)

            # gamma[t, i] = P(s_t=i | O, model)
            gamma = alpha * beta
            gamma_sum = gamma.sum(axis=1, keepdims=True)
            gamma_sum = np.where(gamma_sum == 0, 1e-300, gamma_sum)
            gamma /= gamma_sum

            # xi[t, i, j] = P(s_t=i, s_{t+1}=j | O, model)  for t < T-1
            xi = np.zeros((T - 1, self.N, self.N))
            for t in range(T - 1):
                denom = 0.0
                for i in range(self.N):
                    for j in range(self.N):
                        xi[t, i, j] = (
                            alpha[t, i] * self.A[i, j] * self.B[j, obs[t + 1]] * beta[t + 1, j]
                        )
                        denom += xi[t, i, j]
                if denom > 0:
                    xi[t] /= denom

            self.pi = gamma[0]
            for i in range(self.N):
                denom = gamma[:-1, i].sum()
                for j in range(self.N):
                    self.A[i, j] = xi[:, i, j].sum() / (denom if denom > 0 else 1e-300)
            for i in range(self.N):
                denom = gamma[:, i].sum()
                for k in range(self.M):
                    mask = obs_arr == k
                    self.B[i, k] = gamma[mask, i].sum() / (denom if denom > 0 else 1e-300)

            log_likelihood = float(np.log(np.sum(alpha[T - 1]) + 1e-300))
            if abs(log_likelihood - prev_log_likelihood) < tol:
                break
            prev_log_likelihood = log_likelihood

    def sample(
        self,
        length: int,
        rng: SeedLike = None,
    ) -> tuple[list[int], list[int]]:
        """Sample an observation sequence (and the underlying hidden states).

        `rng` may be an int seed, an ``np.random.Generator``, or ``None``.
        Returns ``(observations, states)`` of length ``length``.
        """
        rng = np.random.default_rng(rng)
        state = int(rng.choice(self.N, p=self.pi))
        observations: list[int] = []
        states: list[int] = []
        for _ in range(length):
            obs = int(rng.choice(self.M, p=self.B[state]))
            observations.append(obs)
            states.append(state)
            state = int(rng.choice(self.N, p=self.A[state]))
        return observations, states
