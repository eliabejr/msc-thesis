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

from src.ablation.jit_metrics import compute_add_jit, false_alarm_rate_jit


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
    dict com: ADD, FAR, ARI, Concordance, MRL_all, MRL_bull, MRL_bear
    """
    tl = np.asarray(true_labels, dtype=np.int64)
    pl = np.asarray(pred_labels, dtype=np.int64)

    mrl = mean_run_length(pl)

    return {
        "ADD":         compute_add(tl, pl, max_delay),
        "FAR":         float(false_alarm_rate_jit(tl, pl)),
        "ARI":         compute_ari(tl, pl),
        "Concordance": regime_concordance(tl, pl),
        "MRL_all":     mrl["mean_all"],
        "MRL_bull":    mrl["mean_bull"],
        "MRL_bear":    mrl["mean_bear"],
        "N_runs":      mrl["n_runs"],
    }
