"""
models/jump_model.py
====================
Statistical Jump Model (Bemporad et al. 2018; Nystrup et al. 2020b).

Solves (equation 1 of Shu et al. 2024):

    min_{Θ, S}  Σ_t  l(x_t, θ_{s_t})  +  λ Σ_t  1{s_{t-1} ≠ s_t}

with  l(x, θ) = ½ ‖x − θ‖²  (scaled squared ℓ₂ distance).

Algorithm:  coordinate descent
  ─ Fix Θ  → optimise S via dynamic programming  (Viterbi-like)
  ─ Fix S  → optimise Θ  (cluster centroid update)

The implementation follows the scikit-learn API style referenced in
footnote 11 of the paper (github.com/Yizhan-Oliver-Shu/jump-models).

Single Responsibility : only the jump-model fitting / prediction logic.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config.settings import JM_MAX_ITER, JM_N_STATES, JM_TOL

logger = logging.getLogger(__name__)


class JumpModel:
    """
    K-state Statistical Jump Model with ℓ₂ loss.

    Parameters
    ----------
    n_states   : number of hidden states K  (paper: K=2)
    jump_pen   : jump penalty λ  ≥ 0
    max_iter   : maximum coordinate-descent iterations
    tol        : convergence tolerance (change in objective)
    random_state : seed for centroid initialisation
    """

    def __init__(
        self,
        n_states:     int   = JM_N_STATES,
        jump_pen:     float = 1.0,
        max_iter:     int   = JM_MAX_ITER,
        tol:          float = JM_TOL,
        random_state: int   = 42,
    ) -> None:
        self.n_states     = n_states
        self.jump_pen     = jump_pen
        self.max_iter     = max_iter
        self.tol          = tol
        self.random_state = random_state

        # Fitted attributes
        self.centroids_: Optional[np.ndarray] = None   # (K, D)
        self.labels_:    Optional[np.ndarray] = None   # (T,) optimal state sequence
        self.obj_:       float = np.inf
        self._n_iter:    int   = 0

    # ------------------------------------------------------------------
    # Public interface  (sklearn-compatible)
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "JumpModel":
        """
        Fit the JM to data matrix X  (T × D).

        Parameters
        ----------
        X : feature matrix, rows = time points, columns = features.
            Should be pre-standardised.
        """
        T, D = X.shape
        K    = self.n_states
        lam  = self.jump_pen

        # ── Initialisation: k-means++ centroids ──────────────────────
        rng = np.random.default_rng(self.random_state)
        centroids = self._init_centroids(X, K, rng)

        labels = self._assign_dp(X, centroids, lam)
        obj    = self._objective(X, labels, centroids, lam)

        for it in range(self.max_iter):
            # ── Update centroids (M-step) ─────────────────────────────
            centroids_new = self._update_centroids(X, labels, K)

            # ── Update labels (E-step via DP) ─────────────────────────
            labels_new = self._assign_dp(X, centroids_new, lam)

            obj_new = self._objective(X, labels_new, centroids_new, lam)

            if abs(obj - obj_new) < self.tol:
                centroids = centroids_new
                labels    = labels_new
                obj       = obj_new
                self._n_iter = it + 1
                break

            centroids = centroids_new
            labels    = labels_new
            obj       = obj_new
        else:
            self._n_iter = self.max_iter
            logger.debug("JM did not converge after %d iterations.", self.max_iter)

        self.centroids_ = centroids
        self.labels_    = labels
        self.obj_       = obj
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assign each row in X to its nearest centroid (no temporal penalty).
        Used to get the regime label for a *new* out-of-sample observation.
        """
        if self.centroids_ is None:
            raise RuntimeError("Call fit() before predict().")
        dists = self._distances(X, self.centroids_)   # (T, K)
        return np.argmin(dists, axis=1)

    def regime_stats(
        self,
        excess_returns: np.ndarray,
    ) -> dict[int, dict[str, float]]:
        """
        Compute per-regime mean excess return and annualised volatility.
        Used to label bullish (0) / bearish (1) regimes.
        """
        if self.labels_ is None:
            raise RuntimeError("Call fit() before regime_stats().")
        stats: dict[int, dict[str, float]] = {}
        for k in range(self.n_states):
            mask = self.labels_ == k
            r    = excess_returns[mask]
            stats[k] = {
                "mean_daily":  float(r.mean()) if len(r) > 0 else 0.0,
                "cum_return":  float((1 + r).prod() - 1) if len(r) > 0 else 0.0,
                "volatility":  float(r.std() * np.sqrt(252)) if len(r) > 0 else 0.0,
                "count":       int(mask.sum()),
            }
        return stats

    # ------------------------------------------------------------------
    # Private  –  Coordinate descent steps
    # ------------------------------------------------------------------

    @staticmethod
    def _distances(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Compute  ½ ‖x_t − θ_k‖²  for all (t, k).

        Returns (T, K) matrix.
        """
        # Broadcast: (T,1,D) - (1,K,D) → (T,K,D)
        diff = X[:, None, :] - centroids[None, :, :]  # (T,K,D)
        return 0.5 * (diff ** 2).sum(axis=2)           # (T,K)

    @staticmethod
    def _init_centroids(
        X: np.ndarray,
        K: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """K-means++ initialisation."""
        T = X.shape[0]
        idx = [rng.integers(T)]
        for _ in range(K - 1):
            dists = np.array([
                min(0.5 * np.sum((X[i] - X[c]) ** 2) for c in idx)
                for i in range(T)
            ])
            probs = dists / dists.sum()
            idx.append(rng.choice(T, p=probs))
        return X[idx]  # (K, D)

    def _assign_dp(
        self,
        X: np.ndarray,
        centroids: np.ndarray,
        lam: float,
    ) -> np.ndarray:
        """
        Dynamic programming to solve:
            min_S  Σ_t l(x_t, θ_{s_t}) + λ Σ_t 1{s_{t-1} ≠ s_t}

        Complexity: O(T · K²) → efficient for small K.
        """
        T, _ = X.shape
        K    = self.n_states
        loss = self._distances(X, centroids)  # (T, K)

        # V[t, k] = min cost to reach state k at time t
        V   = np.full((T, K), np.inf)
        ptr = np.zeros((T, K), dtype=int)

        # Initialise
        V[0] = loss[0]

        for t in range(1, T):
            for k in range(K):
                # cost of transitioning from each previous state j to k
                costs = V[t - 1] + lam * (np.arange(K) != k).astype(float)
                best_j      = int(np.argmin(costs))
                V[t, k]     = costs[best_j] + loss[t, k]
                ptr[t, k]   = best_j

        # Back-tracking
        labels    = np.zeros(T, dtype=int)
        labels[T - 1] = int(np.argmin(V[T - 1]))
        for t in range(T - 2, -1, -1):
            labels[t] = ptr[t + 1, labels[t + 1]]

        return labels

    @staticmethod
    def _update_centroids(
        X: np.ndarray,
        labels: np.ndarray,
        K: int,
    ) -> np.ndarray:
        """Recompute centroids as the mean of assigned observations."""
        D = X.shape[1]
        centroids = np.zeros((K, D))
        for k in range(K):
            mask = labels == k
            if mask.any():
                centroids[k] = X[mask].mean(axis=0)
        return centroids

    def _objective(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray,
        lam: float,
    ) -> float:
        """Evaluate the JM objective function."""
        loss_term = sum(
            self._distances(X[t : t + 1], centroids)[0, labels[t]]
            for t in range(len(X))
        )
        jump_term = lam * float((labels[1:] != labels[:-1]).sum())
        return float(loss_term + jump_term)
