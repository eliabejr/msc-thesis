"""
tests/test_smoke.py
===================
Fast smoke tests that verify the environment is correctly configured.
No external data downloads required — all tests use synthetic data.
Runtime: < 30 seconds.
"""

import numpy as np
import pandas as pd
import pytest


# ── 1. Imports ────────────────────────────────────────────────────────────────

def test_core_imports():
    from src.config.settings import ASSETS, LAMBDA_CANDIDATES, ASSET_TICKERS
    assert len(ASSETS) == 12
    assert len(LAMBDA_CANDIDATES) > 10
    assert "LargeCap" in ASSET_TICKERS


def test_feature_imports():
    from src.features.return_features import ReturnFeatureBuilder
    from src.features.macro_features import MacroFeatureBuilder
    assert ReturnFeatureBuilder is not None


def test_model_imports():
    from src.models.jump_model import JumpModel
    from src.models.xgb_classifier import RegimeForecaster
    assert JumpModel is not None


def test_portfolio_imports():
    from src.portfolio.optimizer import MVOptimizer
    from src.portfolio.strategies import STRATEGY_REGISTRY
    assert len(STRATEGY_REGISTRY) == 7


def test_time_series_stats_smoke():
    from src.analysis.time_series_stats import (
        hp_decompose,
        hac_ols,
        spearman_bootstrap_ci,
    )

    idx = pd.date_range("2020-01-01", periods=80, freq="B")
    x = pd.Series(np.random.default_rng(0).standard_normal(80), index=idx)
    trend, cycle = hp_decompose(x, lam=1_600_000.0)
    assert len(trend) == len(x)
    assert np.isfinite(trend.iloc[-1])
    y = np.random.default_rng(1).standard_normal(80)
    z = np.random.default_rng(2).standard_normal(80)
    fit = hac_ols(y, z)
    assert hasattr(fit, "pvalues")
    rho, lo, hi = spearman_bootstrap_ci(y, z, n_boot=100, seed=0)
    assert np.isfinite(rho)


def test_cluster_stability_cvi_smoke():
    from src.cluster_stability import internal_validity_scores

    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 4))
    labels = np.r_[np.zeros(40, dtype=int), np.ones(40, dtype=int)]
    scores = internal_validity_scores(X, labels)
    assert np.isfinite(scores["silhouette"])
    assert np.isfinite(scores["davies_bouldin"])
    assert np.isfinite(scores["calinski_harabasz"])


# ── 2. Jump Model ─────────────────────────────────────────────────────────────

def test_jump_model_two_clear_regimes():
    """JM should recover two perfectly separated synthetic regimes."""
    from src.models.jump_model import JumpModel

    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal(loc=[ 2,  2], scale=0.1, size=(200, 2)),
        rng.normal(loc=[-2, -2], scale=0.1, size=(200, 2)),
    ])
    jm = JumpModel(n_states=2, jump_pen=10.0, max_iter=100)
    labels = jm.fit_predict(X)

    # Both states must be present
    assert len(np.unique(labels)) == 2
    # Perfect recovery: one half should be one label, other half the other
    assert labels[:200].std() == 0
    assert labels[200:].std() == 0
    assert labels[0] != labels[200]


def test_jump_model_lambda_zero_is_kmeans():
    """With λ=0, JM reduces to k-means (no temporal penalty)."""
    from src.models.jump_model import JumpModel

    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, 4))
    jm = JumpModel(n_states=2, jump_pen=0.0, max_iter=50)
    jm.fit(X)
    assert jm.centroids_ is not None
    assert jm.centroids_.shape == (2, 4)


def test_jump_model_high_lambda_single_regime():
    """With very high λ, all points collapse into one regime."""
    from src.models.jump_model import JumpModel

    rng = np.random.default_rng(2)
    X = rng.standard_normal((100, 3))
    jm = JumpModel(n_states=2, jump_pen=1e6, max_iter=50)
    jm.fit(X)
    # Expect at most one jump in the entire sequence
    jumps = int((jm.labels_[1:] != jm.labels_[:-1]).sum())
    assert jumps <= 1


# ── 3. Return Features ────────────────────────────────────────────────────────

def test_return_features_shape():
    from src.features.return_features import ReturnFeatureBuilder

    rng = np.random.default_rng(3)
    ret = pd.Series(rng.normal(0, 0.01, 500),
                    index=pd.date_range("2010-01-01", periods=500, freq="B"))

    builder = ReturnFeatureBuilder()

    # JM features for a non-excluded asset (8 features)
    feats_jm = builder.build(ret, "LargeCap", for_jm=True)
    assert feats_jm.shape == (500, 8)

    # JM features for excluded asset (6 features — no DD)
    feats_no_dd = builder.build(ret, "AggBond", for_jm=True)
    assert feats_no_dd.shape == (500, 6)

    # XGBoost features always include all 8
    feats_xgb = builder.build(ret, "AggBond", for_jm=False)
    assert feats_xgb.shape == (500, 8)


# ── 4. Optimizer ─────────────────────────────────────────────────────────────

def test_optimizer_basic():
    from src.portfolio.optimizer import MVOptimizer

    N = 4
    rng = np.random.default_rng(4)
    mu = np.array([0.001, 0.002, -0.001, 0.0005])
    A  = rng.standard_normal((N, N))
    Sigma = A @ A.T / N + np.eye(N) * 1e-4

    opt = MVOptimizer(gamma_risk=10.0, gamma_trade=0.0)
    w   = opt.solve(mu, Sigma)

    assert w.shape == (N,)
    assert (w >= -1e-6).all(), "Weights must be non-negative"
    assert w.sum() <= 1.0 + 1e-6, "Leverage must be ≤ 1"
    assert (w <= 0.40 + 1e-6).all(), "Per-asset cap must be ≤ 40%"


def test_optimizer_all_negative_mu():
    """When all returns are negative, optimizer should allocate near-zero weights."""
    from src.portfolio.optimizer import MVOptimizer

    N = 4
    mu = np.full(N, -0.01)
    Sigma = np.eye(N) * 0.0001

    opt = MVOptimizer(gamma_risk=10.0, gamma_trade=0.0)
    w   = opt.solve(mu, Sigma)
    assert w.sum() < 0.05   # essentially all in risk-free


# ── 5. Strategies ─────────────────────────────────────────────────────────────

def test_ew_strategy():
    from src.portfolio.strategies import EWPortfolio
    from src.config.settings import ASSETS

    strat = EWPortfolio(ASSETS)
    rng   = np.random.default_rng(5)
    ret   = pd.DataFrame(
        rng.normal(0, 0.01, (100, 12)),
        columns=ASSETS,
        index=pd.date_range("2010-01-01", periods=100, freq="B"),
    )
    w = strat.weights(date=ret.index[-1], returns=ret, rf=pd.Series(dtype=float))
    assert len(w) == 12
    assert abs(w.sum() - 1.0) < 1e-6
    assert abs(w[0] - 1/12) < 1e-6


def test_ew_jmxgb_fewer_than_min_bullish():
    """EW(JM-XGB) must allocate 100% to risk-free when ≤ 3 assets are bullish."""
    from src.portfolio.strategies import EWJMXGBPortfolio
    from src.config.settings import ASSETS

    strat     = EWJMXGBPortfolio(ASSETS)
    regime_fc = pd.Series({a: 1 for a in ASSETS})   # all bearish
    regime_fc["LargeCap"] = 0                         # only 1 bullish

    w = strat.weights(
        date=pd.Timestamp("2015-01-01"),
        returns=pd.DataFrame(),
        rf=pd.Series(dtype=float),
        regime_fc=regime_fc,
    )
    assert w.sum() == pytest.approx(0.0)   # 100% risk-free


# ── 6. Metrics ────────────────────────────────────────────────────────────────

def test_metrics_smoke():
    from src.backtest.metrics import compute_metrics

    rng  = np.random.default_rng(6)
    idx  = pd.date_range("2010-01-01", periods=252 * 3, freq="B")
    ret  = pd.Series(rng.normal(0.0004, 0.01, len(idx)), index=idx)
    to   = pd.Series(rng.uniform(0, 0.05, len(idx)), index=idx)
    wts  = pd.DataFrame(
        np.ones((len(idx), 3)) / 3,
        index=idx, columns=["A", "B", "C"],
    )
    rf   = pd.Series(0.00015, index=idx)

    m = compute_metrics(ret, to, wts, rf)
    assert "Sharpe" in m
    assert "MDD" in m
    assert m["MDD"] <= 0
    assert m["Leverage"] == pytest.approx(1.0, abs=0.01)
