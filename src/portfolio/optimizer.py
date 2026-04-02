"""
portfolio/optimizer.py
======================
Markowitz Mean-Variance Optimiser (equation 2-4, Shu et al. 2024).

    maximize   w^T μ − γ_risk · w^T Σ w − γ_trade · a · ‖w − w_pre‖₁
    subject to 0 ≤ w ≤ w_ub
               1^T w ≤ L

Uses CVXPY with the OSQP solver (open-source; drop-in for Gurobi / gurobipy
referenced in the paper).  Gurobi can be substituted by changing SOLVER.

Single Responsibility : only quadratic-programme formulation + solving.
Dependency Inversion  : callers supply μ, Σ, w_pre; no data fetching here.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import cvxpy as cp
import numpy as np

from src.config.settings import (
    LEVERAGE_MAX,
    TRANSACTION_COST,
    WEIGHT_UB,
)

logger = logging.getLogger(__name__)

# Default solver; change to cp.GUROBI if a Gurobi licence is available
SOLVER = cp.OSQP


class MVOptimizer:
    """
    Solve the enhanced Markowitz QP for a given risk/trade aversion pair.

    Parameters
    ----------
    gamma_risk  : risk-aversion coefficient γ^risk
    gamma_trade : trade-aversion coefficient γ^trade
    weight_ub   : per-asset upper bound  (default 40%)
    leverage    : maximum total leverage (default 1.0)
    cost_bps    : one-way transaction cost in basis points
    solver      : CVXPY solver (default OSQP)
    """

    def __init__(
        self,
        gamma_risk:  float = 10.0,
        gamma_trade: float = 1.0,
        weight_ub:   float = WEIGHT_UB,
        leverage:    float = LEVERAGE_MAX,
        cost_bps:    float = 5.0,
        solver:      str   = "OSQP",
    ) -> None:
        self.gamma_risk  = gamma_risk
        self.gamma_trade = gamma_trade
        self.weight_ub   = weight_ub
        self.leverage    = leverage
        self.cost        = cost_bps * 1e-4
        self.solver      = solver

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(
        self,
        mu:    np.ndarray,
        Sigma: np.ndarray,
        w_pre: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Solve the QP and return optimal weights.

        Parameters
        ----------
        mu    : (N,) expected excess return vector
        Sigma : (N, N) covariance matrix
        w_pre : (N,) pre-trade weights (used for trading-cost term);
                if None, assumed equal-weight / zero.

        Returns
        -------
        w_opt : (N,) optimal weights; sums to ≤ leverage.
        """
        N = len(mu)
        if w_pre is None:
            w_pre = np.zeros(N)

        w    = cp.Variable(N, name="w")
        ret  = mu @ w
        risk = cp.quad_form(w, cp.psd_wrap(Sigma))

        # L1 trading-cost term (linearised with auxiliary variable)
        trade_cost = self.gamma_trade * self.cost * cp.norm1(w - w_pre)

        objective = cp.Maximize(ret - self.gamma_risk * risk - trade_cost)
        constraints = [
            w >= 0,
            w <= self.weight_ub,
            cp.sum(w) <= self.leverage,
        ]

        prob = cp.Problem(objective, constraints)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                prob.solve(solver=self.solver, warm_start=True, verbose=False)
            except cp.error.SolverError:
                logger.warning("Primary solver failed; trying SCS fallback.")
                prob.solve(solver=cp.SCS, verbose=False)

        if w.value is None:
            logger.warning("Optimiser returned no solution; using pre-trade weights.")
            return np.clip(w_pre, 0, self.weight_ub)

        return np.clip(w.value, 0, self.weight_ub)
