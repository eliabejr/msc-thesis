"""
ablation/regime_diagnostics.py
================================
Diagnósticos de qualidade de regimes para o ablation study.

Funções:
  - detect_changepoints : identifica índices de mudança de regime
  - compute_add         : Average Detection Delay entre dois sinais
  - compute_ari         : Adjusted Rand Index entre dois conjuntos de labels
  - mean_run_length     : duração média dos episódios de mesmo regime
  - regime_concordance  : taxa de concordância ponto-a-ponto entre sinais

Uso:
  >>> import numpy as np
  >>> from src.ablation.regime_diagnostics import compute_add, compute_ari
  >>> true = np.array([0,0,0,1,1,0,0,1,1,1])
  >>> pred = np.array([0,0,1,1,1,0,1,1,1,1])
  >>> compute_add(true, pred)   # ADD em dias
  >>> compute_ari(true, pred)   # ARI ∈ [-1, 1]
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import numba as nb
from sklearn.metrics import adjusted_rand_score

from src.ablation.jit_metrics import compute_add_jit, false_alarm_rate_jit, miss_rate_jit


# ---------------------------------------------------------------------------
# Detecção de changepoints
# ---------------------------------------------------------------------------

def detect_changepoints(labels: np.ndarray) -> np.ndarray:
    """
    Retorna os índices onde ocorre mudança de regime.

    Parameters
    ----------
    labels : array de labels (0 ou 1)

    Returns
    -------
    Array de índices t onde labels[t] ≠ labels[t-1].
    """
    labels = np.asarray(labels)
    changes = np.where(np.diff(labels) != 0)[0] + 1
    return changes


def changepoint_binary_labels(labels: np.ndarray) -> np.ndarray:
    """
    Converte um sinal de regimes em uma série binária de changepoints.

    O elemento ``t`` da saída indica se houve mudança entre ``labels[t-1]`` e
    ``labels[t]``. A série resultante tem comprimento ``T-1``.
    """
    arr = np.asarray(labels)
    if len(arr) < 2:
        return np.asarray([], dtype=np.int64)
    return (arr[1:] != arr[:-1]).astype(np.int64)


# ---------------------------------------------------------------------------
# ADD (wrapper sobre JIT)
# ---------------------------------------------------------------------------

def compute_add(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    max_delay:   int = 126,
) -> float:
    """
    Average Detection Delay (wrapper Python sobre o JIT interno).

    Mede a latência média em detectar uma mudança de regime.

    Parameters
    ----------
    true_labels : labels de referência (0 = bull, 1 = bear)
    pred_labels : labels preditos
    max_delay   : penalidade máxima se detecção não ocorrer na janela

    Returns
    -------
    float : ADD em dias
    """
    tl = np.asarray(true_labels, dtype=np.int64)
    pl = np.asarray(pred_labels, dtype=np.int64)
    return float(compute_add_jit(tl, pl, max_delay))


def compute_miss_rate(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    max_delay:   int = 126,
) -> float:
    """
    Miss Rate: fração de mudanças reais não detectadas dentro de ``max_delay``.
    """
    tl = np.asarray(true_labels, dtype=np.int64)
    pl = np.asarray(pred_labels, dtype=np.int64)
    return float(miss_rate_jit(tl, pl, max_delay))


# ---------------------------------------------------------------------------
# Adjusted Rand Index
# ---------------------------------------------------------------------------

def compute_ari(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
) -> float:
    """
    Adjusted Rand Index (ARI) entre dois conjuntos de labels.

    ARI = 1.0 indica concordância perfeita.
    ARI = 0.0 é o esperado ao acaso.
    ARI < 0  indica menos concordância que o acaso.

    Usa sklearn.metrics.adjusted_rand_score internamente.
    """
    return float(adjusted_rand_score(
        np.asarray(labels_a),
        np.asarray(labels_b),
    ))


# ---------------------------------------------------------------------------
# Mean Run Length
# ---------------------------------------------------------------------------

def mean_run_length(labels: np.ndarray) -> Dict[str, float]:
    """
    Duração média dos episódios de mesmo regime (run-length).

    Parameters
    ----------
    labels : array de labels (0 ou 1)

    Returns
    -------
    dict com:
      - 'mean_all'  : duração média de todos os episódios
      - 'mean_bull' : duração média dos episódios de regime 0 (bull)
      - 'mean_bear' : duração média dos episódios de regime 1 (bear)
      - 'n_runs'    : número total de episódios
    """
    labels = np.asarray(labels)
    if len(labels) == 0:
        return {"mean_all": 0.0, "mean_bull": 0.0, "mean_bear": 0.0, "n_runs": 0}

    runs: List[Tuple[int, int]] = []  # (label, length)
    current = labels[0]
    run_len = 1

    for i in range(1, len(labels)):
        if labels[i] == current:
            run_len += 1
        else:
            runs.append((int(current), run_len))
            current = labels[i]
            run_len = 1
    runs.append((int(current), run_len))

    all_lens  = [r[1] for r in runs]
    bull_lens = [r[1] for r in runs if r[0] == 0]
    bear_lens = [r[1] for r in runs if r[0] == 1]

    return {
        "mean_all":  float(np.mean(all_lens))  if all_lens  else 0.0,
        "mean_bull": float(np.mean(bull_lens)) if bull_lens else 0.0,
        "mean_bear": float(np.mean(bear_lens)) if bear_lens else 0.0,
        "n_runs":    len(runs),
    }


# ---------------------------------------------------------------------------
# Concordância ponto-a-ponto
# ---------------------------------------------------------------------------

def regime_concordance(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
) -> float:
    """
    Taxa de concordância ponto-a-ponto entre dois sinais de regime.

    Concordance = sum(labels_a == labels_b) / T

    Retorna valor em [0, 1].
    """
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    return float((a == b).mean())


def binary_accuracy(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
) -> float:
    """
    Accuracy binária simples entre dois sinais alinhados.
    """
    tl = np.asarray(true_labels)
    pl = np.asarray(pred_labels)
    if len(tl) != len(pl) or len(tl) == 0:
        return 0.0
    return float((tl == pl).mean())


def binary_balanced_accuracy(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
) -> float:
    """
    Balanced Accuracy binária.

    Calcula a média do recall das classes presentes em ``true_labels``.
    """
    tl = np.asarray(true_labels)
    pl = np.asarray(pred_labels)
    if len(tl) != len(pl) or len(tl) == 0:
        return 0.0

    recalls: List[float] = []
    for cls in (0, 1):
        mask = tl == cls
        support = int(mask.sum())
        if support > 0:
            recalls.append(float((pl[mask] == cls).mean()))

    if not recalls:
        return 0.0
    return float(np.mean(recalls))


def changepoint_detection_summary(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    max_delay:   int = 126,
) -> Dict[str, float]:
    """
    Resumo das métricas de detecção de changepoint.

    Sejam ``c_t = 1[y_t != y_{t-1}]`` e ``ĉ_t = 1[ŷ_t != ŷ_{t-1}]`` para
    ``t = 2, ..., T``. Então:

    - ``Accuracy_cp = (1/(T-1)) * Σ 1[c_t = ĉ_t]``
    - ``BalancedAccuracy_cp = (Recall_change + Recall_stable) / 2``
    - ``FAR = FP_stable / N_stable``
    - ``MissRate = MissedChanges / N_changes``
    """
    tl = np.asarray(true_labels, dtype=np.int64)
    pl = np.asarray(pred_labels, dtype=np.int64)
    true_cp = changepoint_binary_labels(tl)
    pred_cp = changepoint_binary_labels(pl)
    return {
        "ADD": compute_add(tl, pl, max_delay),
        "FAR": float(false_alarm_rate_jit(tl, pl)),
        "MissRate": compute_miss_rate(tl, pl, max_delay),
        "CP_Accuracy": binary_accuracy(true_cp, pred_cp),
        "CP_BalancedAccuracy": binary_balanced_accuracy(true_cp, pred_cp),
    }


def state_classification_summary(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
) -> Dict[str, float]:
    """
    Resumo das métricas de classificação de estado do regime.
    """
    tl = np.asarray(true_labels, dtype=np.int64)
    pl = np.asarray(pred_labels, dtype=np.int64)
    return {
        "StateAccuracy": binary_accuracy(tl, pl),
        "StateBalancedAccuracy": binary_balanced_accuracy(tl, pl),
        "Concordance": regime_concordance(tl, pl),
        "ARI": compute_ari(tl, pl),
    }


# ---------------------------------------------------------------------------
# Resumo diagnóstico completo
# ---------------------------------------------------------------------------

def regime_diagnostics_summary(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    max_delay:   int = 126,
) -> Dict[str, float]:
    """
    Calcula um resumo completo de diagnósticos de regime.

    Parameters
    ----------
    true_labels : labels de referência
    pred_labels : labels preditos
    max_delay   : janela máxima para ADD

    Returns
    -------
    dict com: métricas de changepoint, estado e estabilidade
    """
    tl = np.asarray(true_labels, dtype=np.int64)
    pl = np.asarray(pred_labels, dtype=np.int64)

    mrl = mean_run_length(pl)
    cp = changepoint_detection_summary(tl, pl, max_delay)
    state = state_classification_summary(tl, pl)

    return {
        "ADD":         cp["ADD"],
        "FAR":         cp["FAR"],
        "MissRate":    cp["MissRate"],
        "CP_Accuracy": cp["CP_Accuracy"],
        "CP_BalancedAccuracy": cp["CP_BalancedAccuracy"],
        "ARI":         state["ARI"],
        "Concordance": state["Concordance"],
        "StateAccuracy": state["StateAccuracy"],
        "StateBalancedAccuracy": state["StateBalancedAccuracy"],
        "MRL_all":     mrl["mean_all"],
        "MRL_bull":    mrl["mean_bull"],
        "MRL_bear":    mrl["mean_bear"],
        "N_runs":      mrl["n_runs"],
    }
