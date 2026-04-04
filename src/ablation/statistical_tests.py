"""
ablation/statistical_tests.py
===============================
Testes estatísticos para comparação de configurações do ablation study.

Testes implementados:
  - wilcoxon_test          : Wilcoxon signed-rank (comparação pareada)
  - friedman_test          : Friedman test (comparação múltipla non-paramétrica)
  - holm_correction        : Correção Holm step-down para múltiplas comparações
  - cohens_d               : Tamanho de efeito Cohen's d
  - deflated_sharpe_ratio  : DSR (Bailey & López de Prado, 2014)
  - pairwise_comparison_table : tabela de comparações pareadas em Polars

Referências:
  - Wilcoxon (1945)
  - Friedman (1937)
  - Holm (1979)
  - Cohen (1988)
  - Bailey & López de Prado (2014) — Deflated Sharpe Ratio
  - Demšar (2006) — Comparação de classificadores
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
import polars as pl


# ---------------------------------------------------------------------------
# Wilcoxon Signed-Rank Test
# ---------------------------------------------------------------------------

def wilcoxon_test(
    a:         np.ndarray,
    b:         np.ndarray,
    alpha:     float = 0.05,
    zero_method: str = "wilcox",
) -> Dict[str, float]:
    """
    Wilcoxon signed-rank test para comparação pareada.

    H0: as distribuições de (a - b) são simétricas em torno de zero.

    Parameters
    ----------
    a, b        : arrays de mesma dimensão para comparação pareada
    alpha       : nível de significância
    zero_method : método para tratar zeros ('wilcox', 'pratt', 'zsplit')

    Returns
    -------
    dict com:
      - statistic : estatística W
      - p_value   : p-value bilateral
      - reject_h0 : True se p < alpha
      - effect_size: r = Z / sqrt(N)
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    mask = ~np.isnan(diff)
    diff = diff[mask]

    if len(diff) < 5:
        return {"statistic": np.nan, "p_value": np.nan, "reject_h0": False, "effect_size": 0.0}

    try:
        stat, p = stats.wilcoxon(diff, zero_method=zero_method, alternative="two-sided")
        z = stats.norm.ppf(1 - p / 2) * np.sign(np.median(diff))
        r = abs(z) / np.sqrt(len(diff))
    except ValueError:
        return {"statistic": np.nan, "p_value": 1.0, "reject_h0": False, "effect_size": 0.0}

    return {
        "statistic":   float(stat),
        "p_value":     float(p),
        "reject_h0":   bool(p < alpha),
        "effect_size": float(r),
    }


# ---------------------------------------------------------------------------
# Friedman Test
# ---------------------------------------------------------------------------

def friedman_test(
    *groups: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Friedman test para comparação não-paramétrica de k ≥ 3 grupos.

    H0: todas as k condições têm a mesma distribuição.

    Parameters
    ----------
    *groups : k arrays de mesma dimensão (blocos × condições)
    alpha   : nível de significância

    Returns
    -------
    dict com statistic, p_value, reject_h0
    """
    if len(groups) < 3:
        raise ValueError("Friedman test requer pelo menos 3 grupos.")

    arrays = [np.asarray(g, dtype=float) for g in groups]
    min_len = min(len(a) for a in arrays)
    arrays  = [a[:min_len] for a in arrays]

    stat, p = stats.friedmanchisquare(*arrays)

    return {
        "statistic": float(stat),
        "p_value":   float(p),
        "reject_h0": bool(p < alpha),
        "df":        float(len(groups) - 1),
    }


# ---------------------------------------------------------------------------
# Correção Holm Step-Down
# ---------------------------------------------------------------------------

def holm_correction(
    p_values:   List[float],
    names:      Optional[List[str]] = None,
    alpha:      float = 0.05,
) -> pl.DataFrame:
    """
    Correção Holm (1979) para múltiplas comparações.

    Controla o Family-Wise Error Rate (FWER) com menos conservadorismo
    que a correção de Bonferroni.

    Parameters
    ----------
    p_values : lista de p-values
    names    : rótulos para cada p-value
    alpha    : nível de significância

    Returns
    -------
    pl.DataFrame com colunas:
      name, p_raw, p_adjusted, reject_h0, rank
    """
    n = len(p_values)
    if names is None:
        names = [f"test_{i}" for i in range(n)]

    # Ordenar por p-value crescente
    order = np.argsort(p_values)
    sorted_p = [p_values[i] for i in order]
    sorted_names = [names[i] for i in order]

    # Threshold Holm: alpha / (n - rank + 1)
    adjusted_p = []
    reject     = []
    stop       = False
    for rank, p in enumerate(sorted_p, 1):
        threshold = alpha / (n - rank + 1)
        if stop or p > threshold:
            stop = True
            reject.append(False)
        else:
            reject.append(True)
        # p-value ajustado (Holm)
        p_adj = min(1.0, p * (n - rank + 1))
        adjusted_p.append(p_adj)

    # Reordenar para saída original
    result_rows = list(zip(sorted_names, sorted_p, adjusted_p, reject))

    return pl.DataFrame({
        "name":      [r[0] for r in result_rows],
        "p_raw":     [r[1] for r in result_rows],
        "p_adjusted":[r[2] for r in result_rows],
        "reject_h0": [r[3] for r in result_rows],
    }).sort("p_raw")


# ---------------------------------------------------------------------------
# Cohen's d
# ---------------------------------------------------------------------------

def cohens_d(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Cohen's d: tamanho de efeito para comparação entre dois grupos.

    d = (mean_a - mean_b) / pooled_std

    Interpretação:
      |d| < 0.2 : negligível
      |d| < 0.5 : pequeno
      |d| < 0.8 : médio
      |d| ≥ 0.8 : grande

    Returns
    -------
    float : d de Cohen
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]

    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0

    mean_diff  = a.mean() - b.mean()
    pooled_std = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))

    if pooled_std < 1e-12:
        return 0.0

    return float(mean_diff / pooled_std)


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio (Bailey & López de Prado, 2014)
# ---------------------------------------------------------------------------

def deflated_sharpe_ratio(
    sharpe_obs:  float,
    n_trials:    int,
    n_obs:       int,
    skewness:    float = 0.0,
    kurtosis:    float = 3.0,
) -> float:
    """
    Deflated Sharpe Ratio (DSR).

    Corrige o Sharpe observado pelo viés de seleção introduzido por múltiplos testes.

    DSR = P(SR* > 0 | SR_obs, N, T, skew, kurt)

    Onde SR* é o Sharpe "verdadeiro" e SR_obs é o Sharpe da melhor configuração.

    Parameters
    ----------
    sharpe_obs : Sharpe Ratio da melhor configuração testada
    n_trials   : número de configurações testadas
    n_obs      : número de observações (dias)
    skewness   : assimetria dos retornos
    kurtosis   : curtose dos retornos (3 = normal)

    Returns
    -------
    float : DSR ∈ [0, 1] (probabilidade do SR verdadeiro > 0)

    Referência:
      Bailey, D.H. & López de Prado, M. (2014). The Deflated Sharpe Ratio.
      Journal of Portfolio Management, 40(5), 94-107.
    """
    if n_trials < 1 or n_obs < 2:
        return 0.0

    # Esperança máxima do Sharpe com seleção de múltiplos estimadores
    # E[max SR] ≈ (1 - gamma * euler_mascheroni) * Z^-1(1 - 1/n_trials)
    #           + gamma * euler_mascheroni * Z^-1(1 - 1/e * n_trials)
    # Aproximação simplificada (Bailey & López de Prado, 2014, eq. 8):
    euler = 0.5772156649
    gamma_const = 1.0 - euler  # ≈ 0.4228

    if n_trials > 1:
        expected_max = (
            (1 - euler) * stats.norm.ppf(1 - 1.0 / n_trials) +
            euler       * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
        )
    else:
        expected_max = 0.0

    # Desvio padrão do Sharpe (considerando assimetria e curtose)
    # Var(SR) ≈ (1 + 0.5*SR² - skew*SR + (kurt-1)/4) / (n_obs - 1)
    sr_var = (
        1.0 + 0.5 * sharpe_obs ** 2
        - skewness * sharpe_obs
        + (kurtosis - 1) / 4.0
    ) / max(n_obs - 1, 1)
    sr_std = np.sqrt(sr_var)

    if sr_std < 1e-12:
        return 1.0 if sharpe_obs > expected_max else 0.0

    z = (sharpe_obs - expected_max) / sr_std
    return float(stats.norm.cdf(z))


# ---------------------------------------------------------------------------
# Tabela de comparações pareadas
# ---------------------------------------------------------------------------

def pairwise_comparison_table(
    series_dict: Dict[str, np.ndarray],
    metric_name: str = "metric",
    alpha:       float = 0.05,
) -> pl.DataFrame:
    """
    Gera tabela de comparações pareadas entre todas as configurações.

    Parameters
    ----------
    series_dict : dict {nome_config: array_de_valores}
    metric_name : nome da métrica para display
    alpha       : nível de significância

    Returns
    -------
    pl.DataFrame com colunas:
      config_a, config_b, mean_a, mean_b, delta, p_value, cohens_d,
      reject_h0, significance
    """
    names = list(series_dict.keys())
    rows  = []

    for a_name, b_name in combinations(names, 2):
        arr_a = np.asarray(series_dict[a_name], dtype=float)
        arr_b = np.asarray(series_dict[b_name], dtype=float)

        test    = wilcoxon_test(arr_a, arr_b, alpha=alpha)
        d       = cohens_d(arr_a, arr_b)
        mean_a  = float(np.nanmean(arr_a))
        mean_b  = float(np.nanmean(arr_b))

        # Nível de significância simbólico
        p = test["p_value"]
        if np.isnan(p):
            sig = "n.a."
        elif p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        else:
            sig = "ns"

        rows.append({
            "config_a":     a_name,
            "config_b":     b_name,
            f"mean_{metric_name}_a": round(mean_a, 4),
            f"mean_{metric_name}_b": round(mean_b, 4),
            "delta":        round(mean_a - mean_b, 4),
            "p_value":      round(float(p), 4) if not np.isnan(p) else None,
            "cohens_d":     round(d, 3),
            "reject_h0":    bool(test["reject_h0"]),
            "significance": sig,
        })

    return pl.DataFrame(rows)
