"""
src/ablation
============
Módulos auxiliares para o Ablation Study do pipeline JM-XGB.

Organização:
  - volatility_estimators : estimadores de volatilidade JIT (CC, Parkinson, GK, RS, YZ)
  - jit_metrics           : métricas de performance compiladas com Numba (ADD, Sortino, MDD)
  - regime_diagnostics    : diagnósticos de regimes (changepoints, ARI, run-length)
  - polars_utils          : utilitários de dados e agregação usando Polars
  - statistical_tests     : testes estatísticos (Wilcoxon, Friedman, Holm, Cohen's d)
  - ablation_runner       : orquestrador para execução das experiências de ablation
"""

from src.ablation.volatility_estimators import (
    rolling_close_to_close,
    rolling_parkinson,
    rolling_garman_klass,
    rolling_rogers_satchell,
    rolling_yang_zhang,
    compute_all_estimators,
    ESTIMATOR_NAMES,
)
from src.ablation.jit_metrics import (
    sortino_ratio_jit,
    max_drawdown_jit,
    sharpe_ratio_jit,
    compute_add_jit,
    false_alarm_rate_jit,
    compute_metrics_array,
)
from src.ablation.regime_diagnostics import (
    detect_changepoints,
    compute_add,
    compute_ari,
    mean_run_length,
    regime_concordance,
    regime_diagnostics_summary,
)
from src.ablation.polars_utils import (
    series_to_polars,
    dataframe_to_polars,
    load_regime_forecasts,
    load_portfolio_results,
    build_ablation_summary,
    rolling_metrics_polars,
    float_nan_to_null,
)
from src.ablation.statistical_tests import (
    wilcoxon_test,
    friedman_test,
    holm_correction,
    cohens_d,
    deflated_sharpe_ratio,
    pairwise_comparison_table,
)
from src.ablation.ablation_runner import (
    AblationConfig,
    AblationResult,
    BASELINE_CONFIG,
    ABLATION_A1_CONFIGS,
    ABLATION_A2_CONFIGS,
    ABLATION_A3_CONFIGS,
    ABLATION_B1_CONFIGS,
    ABLATION_B2_CONFIGS,
    ABLATION_C1_CONFIGS,
    ABLATION_C2_CONFIGS,
    ABLATION_D1_CONFIGS,
    ABLATION_I1_CONFIGS,
    ABLATION_I2_CONFIGS,
    prepare_ablation_data,
    run_single_ablation,
    run_ablation_sweep,
    run_full_ablation_study,
    get_ablation_configs,
    get_component_name,
    analyze_ablation,
    compare_ablations,
)

__all__ = [
    # Volatility estimators
    "rolling_close_to_close",
    "rolling_parkinson",
    "rolling_garman_klass",
    "rolling_rogers_satchell",
    "rolling_yang_zhang",
    "compute_all_estimators",
    "ESTIMATOR_NAMES",
    # JIT metrics
    "sortino_ratio_jit",
    "max_drawdown_jit",
    "sharpe_ratio_jit",
    "compute_add_jit",
    "false_alarm_rate_jit",
    "compute_metrics_array",
    # Regime diagnostics
    "detect_changepoints",
    "compute_add",
    "compute_ari",
    "mean_run_length",
    "regime_concordance",
    "regime_diagnostics_summary",
    # Polars utils
    "series_to_polars",
    "dataframe_to_polars",
    "load_regime_forecasts",
    "load_portfolio_results",
    "build_ablation_summary",
    "rolling_metrics_polars",
    "float_nan_to_null",
    # Statistical tests
    "wilcoxon_test",
    "friedman_test",
    "holm_correction",
    "cohens_d",
    "deflated_sharpe_ratio",
    "pairwise_comparison_table",
    # Ablation runner
    "AblationConfig",
    "AblationResult",
    "BASELINE_CONFIG",
    "ABLATION_A1_CONFIGS",
    "ABLATION_A2_CONFIGS",
    "ABLATION_A3_CONFIGS",
    "ABLATION_B1_CONFIGS",
    "ABLATION_B2_CONFIGS",
    "ABLATION_C1_CONFIGS",
    "ABLATION_C2_CONFIGS",
    "ABLATION_D1_CONFIGS",
    "ABLATION_I1_CONFIGS",
    "ABLATION_I2_CONFIGS",
    "prepare_ablation_data",
    "run_single_ablation",
    "run_ablation_sweep",
    "run_full_ablation_study",
    "get_ablation_configs",
    "get_component_name",
    "analyze_ablation",
    "compare_ablations",
]
