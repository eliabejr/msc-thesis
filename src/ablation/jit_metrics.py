"""
ablation/jit_metrics.py
========================
Métricas de performance compiladas com Numba JIT para máxima velocidade.

 Métricas implementadas:
  - Sortino Ratio                (JIT)
  - Maximum Drawdown             (JIT)
  - Sharpe Ratio                 (JIT)
  - Average Detection Delay / ADD(JIT)
  - False Alarm Rate             (JIT)
  - Miss Rate                    (JIT)
  - compute_metrics_array        : wrapper que retorna dict com todas as métricas

Uso:
  >>> import numpy as np
  >>> from src.ablation.jit_metrics import sortino_ratio_jit, compute_metrics_array
  >>> returns = np.random.normal(0.0004, 0.01, 1000)
  >>> sortino = sortino_ratio_jit(returns, rf_daily=0.0001)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import numba as nb

TRADING_DAYS = 252

# ---------------------------------------------------------------------------
# JIT – Sortino Ratio
# ---------------------------------------------------------------------------

@nb.njit(cache=True, fastmath=True)
def sortino_ratio_jit(
    returns:   np.ndarray,
    rf_daily:  float = 0.0,
    ann_factor: int  = 252,
) -> float:
    """
    Sortino Ratio anualizado.

    Sortino = (mean_excess_return / downside_deviation) * sqrt(ann_factor)

    Parameters
    ----------
    returns    : array de retornos diários do portfólio
    rf_daily   : taxa livre de risco diária
    ann_factor : fator de anualização (252 para dados diários)

    Returns
    -------
    float : Sortino Ratio anualizado
    """
    n = len(returns)
    if n < 2:
        return 0.0

    # Média dos excessos
    mean_exc = 0.0
    for i in range(n):
        mean_exc += returns[i] - rf_daily
    mean_exc /= n

    # Downside deviation (semi-desvio padrão dos excessos negativos)
    dd_sq = 0.0
    for i in range(n):
        excess = returns[i] - rf_daily
        if excess < 0.0:
            dd_sq += excess * excess
    dd = (dd_sq / n) ** 0.5

    if dd < 1e-12:
        return 0.0

    return mean_exc / dd * (ann_factor ** 0.5)


# ---------------------------------------------------------------------------
# JIT – Maximum Drawdown
# ---------------------------------------------------------------------------

@nb.njit(cache=True, fastmath=True)
def max_drawdown_jit(returns: np.ndarray) -> float:
    """
    Maximum Drawdown (MDD).

    MDD = min_t [ (W_t - peak_t) / peak_t ]

    Parameters
    ----------
    returns : array de retornos diários (total, não excessos)

    Returns
    -------
    float : MDD (negativo, e.g. -0.35 = -35%)
    """
    n = len(returns)
    if n == 0:
        return 0.0

    wealth = 1.0
    peak   = 1.0
    mdd    = 0.0

    for i in range(n):
        wealth *= (1.0 + returns[i])
        if wealth > peak:
            peak = wealth
        dd = (wealth - peak) / peak
        if dd < mdd:
            mdd = dd

    return mdd


# ---------------------------------------------------------------------------
# JIT – Sharpe Ratio
# ---------------------------------------------------------------------------

@nb.njit(cache=True, fastmath=True)
def sharpe_ratio_jit(
    returns:    np.ndarray,
    rf_daily:   float = 0.0,
    ann_factor: int   = 252,
) -> float:
    """
    Sharpe Ratio anualizado.

    Parameters
    ----------
    returns    : array de retornos diários
    rf_daily   : taxa livre de risco diária
    ann_factor : fator de anualização

    Returns
    -------
    float : Sharpe Ratio anualizado
    """
    n = len(returns)
    if n < 2:
        return 0.0

    mean_exc = 0.0
    for i in range(n):
        mean_exc += returns[i] - rf_daily
    mean_exc /= n

    var = 0.0
    for i in range(n):
        d = (returns[i] - rf_daily) - mean_exc
        var += d * d
    var /= (n - 1)
    std = var ** 0.5

    if std < 1e-12:
        return 0.0

    return mean_exc / std * (ann_factor ** 0.5)


# ---------------------------------------------------------------------------
# JIT – Average Detection Delay (ADD)
# ---------------------------------------------------------------------------

@nb.njit(cache=True, fastmath=True)
def compute_add_jit(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    max_delay:   int = 126,
) -> float:
    """
    Average Detection Delay (ADD).

    Para cada changepoint no sinal de referência (true_labels),
    mede quantos dias até pred_labels concordar com true_labels.

    ADD = E[ min(d : pred[cp+d] == true[cp]) ]

    Parameters
    ----------
    true_labels : array de labels de referência (0 ou 1)
    pred_labels : array de labels preditos
    max_delay   : penalidade máxima se detecção não ocorrer

    Returns
    -------
    float : ADD médio em dias (0 = detecção perfeita)
    """
    T = len(true_labels)
    if T != len(pred_labels) or T < 2:
        return 0.0

    total_delay = 0.0
    n_changes   = 0

    for t in range(1, T):
        if true_labels[t] != true_labels[t - 1]:
            # Changepoint em t
            detected = False
            for d in range(0, max_delay):
                if t + d >= T:
                    break
                if pred_labels[t + d] == true_labels[t]:
                    total_delay += float(d)
                    detected = True
                    break
            if not detected:
                total_delay += float(max_delay)
            n_changes += 1

    if n_changes == 0:
        return 0.0

    return total_delay / float(n_changes)


# ---------------------------------------------------------------------------
# JIT – False Alarm Rate
# ---------------------------------------------------------------------------

@nb.njit(cache=True, fastmath=True)
def false_alarm_rate_jit(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
) -> float:
    """
    Taxa de Alarmes Falsos.

    FAR = (# predições de mudança quando não há mudança real) / (# não-mudanças reais)

    Parameters
    ----------
    true_labels : labels de referência (0 ou 1)
    pred_labels : labels preditos

    Returns
    -------
    float : FAR ∈ [0, 1]
    """
    T = len(true_labels)
    if T < 2:
        return 0.0

    false_alarms   = 0
    non_changes    = 0

    for t in range(1, T):
        if true_labels[t] == true_labels[t - 1]:
            non_changes += 1
            if pred_labels[t] != pred_labels[t - 1]:
                false_alarms += 1

    if non_changes == 0:
        return 0.0

    return float(false_alarms) / float(non_changes)


# ---------------------------------------------------------------------------
# JIT – Miss Rate
# ---------------------------------------------------------------------------

@nb.njit(cache=True, fastmath=True)
def miss_rate_jit(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    max_delay:   int = 126,
) -> float:
    """
    Taxa de mudanças reais não detectadas na janela de tolerância.

    Para cada changepoint em ``true_labels`` no instante ``t``, a mudança é
    considerada detectada se existir ``d`` em ``[0, max_delay)`` tal que
    ``pred_labels[t + d] == true_labels[t]``. Caso contrário, conta como miss.

    Parameters
    ----------
    true_labels : labels de referência (0 ou 1)
    pred_labels : labels preditos
    max_delay   : janela máxima para considerar uma detecção válida

    Returns
    -------
    float : Miss Rate ∈ [0, 1]
    """
    T = len(true_labels)
    if T != len(pred_labels) or T < 2:
        return 0.0

    misses = 0
    n_changes = 0

    for t in range(1, T):
        if true_labels[t] != true_labels[t - 1]:
            detected = False
            for d in range(0, max_delay):
                if t + d >= T:
                    break
                if pred_labels[t + d] == true_labels[t]:
                    detected = True
                    break
            if not detected:
                misses += 1
            n_changes += 1

    if n_changes == 0:
        return 0.0

    return float(misses) / float(n_changes)


# ---------------------------------------------------------------------------
# JIT – Calmar Ratio
# ---------------------------------------------------------------------------

@nb.njit(cache=True, fastmath=True)
def calmar_ratio_jit(
    returns:    np.ndarray,
    rf_daily:   float = 0.0,
    ann_factor: int   = 252,
) -> float:
    """
    Calmar Ratio = |Ann. Return| / |MDD|.
    """
    ann_ret = 0.0
    for i in range(len(returns)):
        ann_ret += returns[i] - rf_daily
    if len(returns) > 0:
        ann_ret = ann_ret / len(returns) * ann_factor

    mdd = max_drawdown_jit(returns)

    if abs(mdd) < 1e-10:
        return 0.0

    return abs(ann_ret) / abs(mdd)


# ---------------------------------------------------------------------------
# JIT – Turnover anualizado
# ---------------------------------------------------------------------------

@nb.njit(cache=True, fastmath=True)
def annualized_turnover_jit(
    weights_diff: np.ndarray,
    ann_factor:   int = 252,
) -> float:
    """
    Turnover anualizado = mean(Σ_i |Δw_i|) × 252.

    Parameters
    ----------
    weights_diff : array 2D (T × N_assets) de variações de peso diárias
    ann_factor   : fator de anualização
    """
    T, N = weights_diff.shape
    if T == 0:
        return 0.0

    total = 0.0
    for t in range(T):
        day_to = 0.0
        for i in range(N):
            d = weights_diff[t, i]
            if d < 0.0:
                day_to -= d
            else:
                day_to += d
        total += day_to

    return total / T * ann_factor


# ---------------------------------------------------------------------------
# JIT – Stage 3 single-asset portfolio
# ---------------------------------------------------------------------------

@nb.njit(cache=True, fastmath=True)
def simple_portfolio_jit(
    pred_labels: np.ndarray,
    er:          np.ndarray,
    gamma_trade: float,
    gamma_risk:  float,
    leverage_max: float,
    transaction_cost: float,
) -> np.ndarray:
    """
    Implementa a regra single-asset do Stage 3 em Numba.

    Semântica idêntica à versão Python:
      - bull := 1 - pred_label
      - pos := bull * risk_scale
      - tc  := |Δpos| * transaction_cost * (1 + gamma_trade)
      - ret := pos * er - tc
    """
    n = min(len(pred_labels), len(er))
    port_ret = np.zeros(n, dtype=np.float64)
    prev_pos = 0.0

    baseline_gr = 10.0
    if (not np.isfinite(gamma_risk)) or gamma_risk <= 0.0:
        risk_scale = 1.0
    else:
        risk_scale = baseline_gr / gamma_risk

    lev_cap = leverage_max
    if lev_cap < 0.0 or not np.isfinite(lev_cap):
        lev_cap = 0.0
    if risk_scale < 0.0:
        risk_scale = 0.0
    elif risk_scale > lev_cap:
        risk_scale = lev_cap

    trade_penalty = transaction_cost * (1.0 + gamma_trade)

    for t in range(n):
        bull = 1.0 - pred_labels[t]
        pos = bull * risk_scale
        tc = abs(pos - prev_pos) * trade_penalty
        port_ret[t] = pos * er[t] - tc
        prev_pos = pos

    return port_ret


# ---------------------------------------------------------------------------
# Wrapper Python: retorna dict de métricas
# ---------------------------------------------------------------------------

def compute_metrics_array(
    returns:     np.ndarray,
    rf_daily:    float = 0.0,
    true_labels: Optional[np.ndarray] = None,
    pred_labels: Optional[np.ndarray] = None,
    ann_factor:  int   = 252,
) -> Dict[str, float]:
    """
    Calcula todas as métricas de uma vez.

    Parameters
    ----------
    returns     : retornos diários do portfólio (array 1D)
    rf_daily    : taxa livre de risco diária média
    true_labels : labels verdadeiros (para ADD, FAR) – opcional
    pred_labels : labels preditos (para ADD, FAR) – opcional
    ann_factor  : fator de anualização

    Returns
    -------
    dict com chaves: Sortino, Sharpe, MDD, Calmar, ADD, FAR, MissRate
    """
    returns_arr = np.asarray(returns, dtype=np.float64)

    metrics: Dict[str, float] = {
        "Sortino": float(sortino_ratio_jit(returns_arr, rf_daily, ann_factor)),
        "Sharpe":  float(sharpe_ratio_jit (returns_arr, rf_daily, ann_factor)),
        "MDD":     float(max_drawdown_jit (returns_arr)),
        "Calmar":  float(calmar_ratio_jit (returns_arr, rf_daily, ann_factor)),
    }

    if true_labels is not None and pred_labels is not None:
        tl = np.asarray(true_labels, dtype=np.int64)
        pl_ = np.asarray(pred_labels, dtype=np.int64)
        metrics["ADD"] = float(compute_add_jit(tl, pl_))
        metrics["FAR"] = float(false_alarm_rate_jit(tl, pl_))
        metrics["MissRate"] = float(miss_rate_jit(tl, pl_))

    return metrics
