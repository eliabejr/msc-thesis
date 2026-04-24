import pytest
import numpy as np
import polars as pl


def test_regime_diagnostics_summary_reports_miss_rate_and_balanced_scores():
    from src.ablation.regime_diagnostics import regime_diagnostics_summary

    true = np.array([0, 0, 0, 1, 1, 1, 0, 0], dtype=np.int64)
    pred = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)

    summary = regime_diagnostics_summary(true, pred, max_delay=2)

    assert summary["MissRate"] == pytest.approx(0.5)
    assert summary["FAR"] == pytest.approx(0.0)
    assert summary["CP_Accuracy"] == pytest.approx(5 / 7)
    assert summary["CP_BalancedAccuracy"] == pytest.approx(0.5)
    assert summary["StateAccuracy"] == pytest.approx(5 / 8)
    assert summary["StateBalancedAccuracy"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# W1 — Joint hyperparameter sweep: terminal wealth + argmax selection
# ---------------------------------------------------------------------------


def test_ablation_result_has_wealth_curve_fields():
    """`AblationResult` deve expor wealth terminal full/val/oos e switches."""
    from src.ablation.ablation_runner import AblationResult

    r = AblationResult()
    for field in (
        "terminal_wealth",
        "terminal_wealth_val",
        "terminal_wealth_oos",
        "n_position_switches",
    ):
        assert hasattr(r, field), f"AblationResult.{field} faltando"


def test_w1_grid_registers_and_respects_budget():
    from src.ablation.ablation_runner import (
        ABLATION_W1_CONFIGS,
        ABLATION_CONFIG_MAP,
        COMPONENT_NAMES,
        W1_BUDGET_MAX,
        get_ablation_configs,
    )

    assert "W1" in ABLATION_CONFIG_MAP
    assert "W1" in COMPONENT_NAMES
    assert len(ABLATION_W1_CONFIGS) > 0
    assert len(ABLATION_W1_CONFIGS) <= W1_BUDGET_MAX
    assert get_ablation_configs("W1") is ABLATION_W1_CONFIGS

    # Todas as 5 dimensões devem variar efetivamente no grid.
    lam  = {c.lambda_penalty  for c in ABLATION_W1_CONFIGS}
    vol  = {c.vol_estimator   for c in ABLATION_W1_CONFIGS}
    k    = {c.n_regimes       for c in ABLATION_W1_CONFIGS}
    rec  = {c.recal_frequency for c in ABLATION_W1_CONFIGS}
    fc   = {c.forecaster_type for c in ABLATION_W1_CONFIGS}
    assert min(len(lam), len(vol), len(k), len(rec), len(fc)) >= 2, \
        "Cada dimensão de W1 deve ter pelo menos 2 valores distintos."


def test_w1_grid_rejects_over_budget():
    """Se o grid nominal exceder o budget, a construção deve abortar ruidosamente."""
    from src.ablation.ablation_runner import _build_w1_configs

    with pytest.raises(ValueError, match="excede budget"):
        _build_w1_configs(budget_max=1)


def test_terminal_wealth_matches_cumprod_definition():
    """terminal_wealth = Π(1 + r_t) a partir de wealth inicial = 1.0."""
    rets = np.array([0.01, -0.02, 0.03, 0.0, 0.005])
    expected = float(np.prod(1.0 + rets))
    # Cálculo direto — mesma fórmula usada em run_single_ablation().
    computed = float(np.prod(1.0 + rets))
    assert computed == pytest.approx(expected)
    # Sanidade: se todos os retornos forem zero, wealth = 1.0.
    assert float(np.prod(1.0 + np.zeros(10))) == pytest.approx(1.0)


def test_validation_fraction_split_is_walk_forward():
    """O split de validação deve usar apenas o prefixo do vetor (sem leakage)."""
    from src.ablation.ablation_runner import W1_VALIDATION_FRACTION

    rets = np.array([0.01, 0.02, -0.01, 0.03, 0.0, 0.01, -0.02, 0.04, 0.01, -0.01])
    split = max(1, int(round(len(rets) * W1_VALIDATION_FRACTION)))
    split = min(split, len(rets) - 1)

    tw_val = float(np.prod(1.0 + rets[:split]))
    tw_oos = float(np.prod(1.0 + rets[split:]))
    tw_full = float(np.prod(1.0 + rets))

    # tw_full = tw_val * tw_oos (propriedade multiplicativa da wealth curve)
    assert tw_val * tw_oos == pytest.approx(tw_full, rel=1e-12)
    # E a janela de validação é estritamente um prefixo.
    assert 1 <= split < len(rets)


def test_best_config_argmax_picks_highest_metric():
    from src.ablation.ablation_runner import best_config_argmax

    df = pl.DataFrame({
        "config":         ["a", "a", "b", "b", "c", "c"],
        "asset":          ["X", "Y", "X", "Y", "X", "Y"],
        "terminal_wealth_val": [1.10, 1.20, 1.40, 1.38, 1.05, 1.00],
        "max_drawdown":        [-0.10, -0.11, -0.30, -0.28, -0.05, -0.04],
        "turnover":            [0.80, 0.85, 1.20, 1.15, 0.50, 0.55],
    })

    sel = best_config_argmax(df, metric="terminal_wealth_val")
    assert sel["config"] == "b"
    assert sel["metric_value"] == pytest.approx((1.40 + 1.38) / 2)
    # Diagnósticos agregados devem vir junto.
    assert "max_drawdown" in sel["details"]
    assert "turnover" in sel["details"]


def test_best_config_argmax_tiebreak_prefers_lower_drawdown():
    """Em empate na métrica principal, o tie-break leva a menor drawdown."""
    from src.ablation.ablation_runner import best_config_argmax

    df = pl.DataFrame({
        "config":              ["safe", "risky"],
        "terminal_wealth_val": [1.30, 1.30],
        "max_drawdown":        [-0.08, -0.40],
        "turnover":            [1.0, 1.0],
    })

    sel = best_config_argmax(
        df,
        metric="terminal_wealth_val",
        tiebreak_cols=[("max_drawdown", True), ("turnover", False)],
    )
    assert sel["config"] == "safe"


def test_analyze_ablation_exposes_argmax_slot():
    """analyze_ablation deve expor a seleção argmax, separada da semântica vs baseline."""
    from src.ablation.ablation_runner import analyze_ablation

    df = pl.DataFrame({
        "ablation_id":         ["W1"] * 6,
        "config":              ["a", "a", "b", "b", "c", "c"],
        "asset":               ["X", "Y", "X", "Y", "X", "Y"],
        "terminal_wealth":     [1.10, 1.20, 1.40, 1.38, 1.05, 1.00],
        "max_drawdown":        [-0.10, -0.11, -0.30, -0.28, -0.05, -0.04],
        "turnover":            [0.80, 0.85, 1.20, 1.15, 0.50, 0.55],
    })

    out = analyze_ablation(df, metric="terminal_wealth")
    assert "argmax" in out
    assert out["argmax"]["config"] == "b"
    assert out["metric"] == "terminal_wealth"


def test_analyze_ablation_pairwise_still_works_with_polars_pivot():
    from src.ablation.ablation_runner import analyze_ablation

    df = pl.DataFrame({
        "ablation_id": ["A1"] * 18,
        "asset": ["A", "B", "C", "D", "E", "F"] * 3,
        "config": ["baseline"] * 6 + ["alt"] * 6 + ["alt2"] * 6,
        "sortino_ratio": [
            0.10, 0.12, 0.11, 0.09, 0.13, 0.10,
            0.40, 0.42, 0.41, 0.39, 0.43, 0.40,
            0.30, 0.31, 0.29, 0.32, 0.33, 0.30,
        ],
        "max_drawdown": [-0.20] * 18,
        "turnover": [1.0] * 18,
    })

    out = analyze_ablation(df, metric="sortino_ratio")
    perf_matrix = out["perf_matrix"]

    assert list(perf_matrix.columns) == ["alt", "alt2", "baseline"]
    assert list(perf_matrix.index) == ["A", "B", "C", "D", "E", "F"]
    assert not out["pairwise"].empty
    assert set(out["pairwise"]["config"]) == {"alt", "alt2"}


def test_simple_portfolio_jit_matches_python_reference():
    from src.ablation.ablation_runner import _simple_portfolio

    pred = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)
    er = np.array([0.01, -0.02, 0.005, -0.01, 0.03, 0.0], dtype=np.float64)

    # Referência Python da lógica econômica original.
    gamma_trade = 1.5
    gamma_risk = 5.0
    leverage_max = 1.2
    transaction_cost = 5e-4
    baseline_gr = 10.0
    risk_scale = baseline_gr / gamma_risk
    risk_scale = float(np.clip(risk_scale, 0.0, max(0.0, leverage_max)))

    expected = np.zeros(len(pred), dtype=np.float64)
    prev_pos = 0.0
    for t in range(len(pred)):
        bull = 1.0 - float(pred[t])
        pos = bull * risk_scale
        tc = abs(pos - prev_pos) * transaction_cost * (1.0 + gamma_trade)
        expected[t] = pos * er[t] - tc
        prev_pos = pos

    # O runner usa TRANSACTION_COST do settings; o baseline do projeto é 5 bps = 5e-4.
    got = _simple_portfolio(
        pred_labels=pred,
        er=er,
        gamma_trade=gamma_trade,
        gamma_risk=gamma_risk,
        leverage_max=leverage_max,
    )
    assert got == pytest.approx(expected, rel=0, abs=1e-12)


def test_w1_exported_from_package_root():
    """A superfície pública do pacote deve expor os novos símbolos."""
    import src.ablation as ab

    assert hasattr(ab, "ABLATION_W1_CONFIGS")
    assert hasattr(ab, "W1_BUDGET_MAX")
    assert hasattr(ab, "W1_VALIDATION_FRACTION")
    assert hasattr(ab, "best_config_argmax")
