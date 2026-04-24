"""
ablation/ablation_runner.py
============================
Implementação completa do protocolo experimental de ablation study.

Baseado no protocolo de design descrito em:
  «Ablation Study: Dissecting the JM-XGB Pipeline for Regime-Aware Asset Allocation»

Estrutura:
  1. AblationConfig / AblationResult  – dataclasses de configuração e resultado
  2. prepare_ablation_data            – preparação de dados OHLC + features
  3. run_single_ablation              – execução de uma configuração
  4. run_ablation_sweep               – varredura completa de um componente
  5. run_full_ablation_study          – estudo completo com paralelização
  6. Configurações de cada ablation   – A1–D1, I1, I2
  7. Checkpointing / fault tolerance

Paralelização via joblib (across assets).
Resultados salvos como Parquet (Polars) ou pickle.

Referências:
  Demšar (2006) — Comparação de classificadores ML
  Bailey & López de Prado (2014) — Deflated Sharpe Ratio
  Shu, Yu & Mulvey (2024) — JM-XGB pipeline
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from src.config.settings import (
    ASSETS,
    REBAL_MONTHS,
    TEST_END,
    TEST_START,
    TRADING_DAYS_YEAR,
    TRAIN_YEARS,
    VAL_YEARS,
    WEIGHT_UB,
    LEVERAGE_MAX,
    TRANSACTION_COST,
)
from src.models.jump_model import JumpModel
from src.features.return_features import ReturnFeatureBuilder
from src.features.macro_features import MacroFeatureBuilder
from src.ablation.jit_metrics import (
    compute_metrics_array,
    sortino_ratio_jit,
    max_drawdown_jit,
    sharpe_ratio_jit,
    simple_portfolio_jit,
)
from src.ablation.regime_diagnostics import (
    compute_add,
    compute_ari,
    mean_run_length,
    regime_concordance,
    regime_diagnostics_summary,
)
from src.ablation.volatility_estimators import (
    rolling_close_to_close,
    rolling_parkinson,
    rolling_garman_klass,
    rolling_rogers_satchell,
    rolling_yang_zhang,
)
from src.ablation.polars_utils import float_nan_to_null

logger = logging.getLogger(__name__)

N_BOOTSTRAP = 20  # número de replicações bootstrap por configuração

# Budget máximo de combinações do sweep conjunto W1 (jump_penalty × vol × k × recal × model).
# Grid foi dimensionado para ficar dentro deste teto; se crescer, aplicar redução determinística.
W1_BUDGET_MAX: int = 150

# Fração do período de teste usada como validação walk-forward para seleção
# de hiperparâmetros sem leakage. Remanescente (1 - fração) é o holdout de
# reporting. Aplicado aos retornos já concatenados em run_single_ablation().
W1_VALIDATION_FRACTION: float = 0.60


# ---------------------------------------------------------------------------
# 1. Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AblationConfig:
    """
    Configuração de um experimento de ablation.

    Todos os parâmetros têm valores default correspondentes à baseline
    descrita em Shu et al. (2024).
    """
    # Identificação
    name:             str   = "baseline"
    description:      str   = ""
    ablation_id:      str   = ""

    # Stage 1 – Regime Identification
    lambda_penalty:   float = 50.0
    n_regimes:        int   = 2
    vol_estimator:    str   = "close_to_close"  # cc|parkinson|gk|rs|yz
    vol_window:       int   = 21
    train_years:      int   = TRAIN_YEARS
    feature_set:      str   = "standard"        # minimal|standard|extended|kitchen_sink

    # Stage 2 – Regime Forecasting
    forecaster_type:  str   = "xgboost"         # xgboost|logistic_regression|decision_tree|random_forest|persistence
    max_depth:        int   = 5                  # para decision_tree
    n_estimators:     int   = 100                # para rf/xgboost
    forecast_lags:    int   = 5

    # Stage 3 – Portfolio Allocation
    gamma_risk:       float = 10.0
    gamma_trade:      float = 1.0
    weight_ub:        float = WEIGHT_UB
    leverage_max:     float = LEVERAGE_MAX

    # Cross-stage
    recal_frequency:  str   = "semi-annual"      # monthly|quarterly|semi-annual|annual
    rebal_months:     Tuple = field(default_factory=lambda: tuple(REBAL_MONTHS))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _metric_nan_to_none(value: Any) -> Any:
    """Polars ignora null em .mean() mas propaga float NaN — usa None para ausentes."""
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


@dataclass
class AblationResult:
    """
    Resultado de uma única execução de ablation.

    Conforme especificado no Protocolo Experimental §6.1.
    """
    # Identificação
    ablation_id:            str   = ""
    asset:                  str   = ""
    config:                 str   = ""
    seed:                   int   = 0

    # Métricas de detecção
    add:                    float = np.nan
    miss_rate:              float = np.nan
    false_alarm_rate:       float = np.nan
    cp_accuracy:            float = np.nan
    cp_balanced_accuracy:   float = np.nan

    # Métricas de previsão
    accuracy:               float = np.nan
    state_accuracy:         float = np.nan
    state_balanced_accuracy: float = np.nan
    f1_score:               float = np.nan

    # Métricas de alocação
    total_return:           float = np.nan
    volatility:             float = np.nan
    sharpe_ratio:           float = np.nan
    sortino_ratio:          float = np.nan
    max_drawdown:           float = np.nan
    calmar_ratio:           float = np.nan
    turnover:               float = np.nan

    # Wealth curve-based metrics (alvo financeiro do sweep conjunto W1).
    # terminal_wealth:      riqueza final no período completo de teste (reporting)
    # terminal_wealth_val:  riqueza final na janela de validação walk-forward (seleção sem leakage)
    # terminal_wealth_oos:  riqueza final na janela de holdout (avaliação honesta da config escolhida)
    # n_position_switches:  número total de trocas de posição (proxy de instabilidade)
    terminal_wealth:        float = np.nan
    terminal_wealth_val:    float = np.nan
    terminal_wealth_oos:    float = np.nan
    n_position_switches:    float = np.nan

    # Métricas de estabilidade
    regime_ari:             float = np.nan
    regime_agreement:       float = np.nan
    mean_run_length:        float = np.nan

    # Métricas computacionais
    training_time_seconds:  float = np.nan
    inference_time_ms:      float = np.nan

    def to_dict(self) -> Dict[str, Any]:
        return {k: _metric_nan_to_none(v) for k, v in asdict(self).items()}


# ---------------------------------------------------------------------------
# 2. Mapa de frequência de recalibração
# ---------------------------------------------------------------------------

def _rebal_months_from_freq(freq: str) -> Tuple[int, ...]:
    """Converte string de frequência para tupla de meses de rebalanceamento."""
    freq_map = {
        "monthly":     tuple(range(1, 13)),
        "quarterly":   (1, 4, 7, 10),
        "semi-annual": (1, 7),
        "annual":      (1,),
    }
    if freq not in freq_map:
        raise ValueError(f"Frequência desconhecida: '{freq}'. Use: {list(freq_map)}")
    return freq_map[freq]


def _generate_refit_dates(
    index:      pd.DatetimeIndex,
    start:      str,
    end:        str,
    rebal_months: Tuple[int, ...],
    train_years: int = TRAIN_YEARS,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Gera lista de (train_end, test_start, test_end) para o expanding window.

    Retorna lista de blocos para refitting do modelo.
    """
    dates = index[(index >= start) & (index <= end)]
    rebal_dates = []
    seen: Set[Tuple[int, int]] = set()
    for dt in dates:
        key = (dt.year, dt.month)
        if dt.month in rebal_months and key not in seen:
            rebal_dates.append(dt)
            seen.add(key)

    blocks = []
    for i, rebal_dt in enumerate(rebal_dates):
        train_end = rebal_dt - pd.Timedelta(days=1)
        test_start = rebal_dt
        test_end = rebal_dates[i + 1] - pd.Timedelta(days=1) if i + 1 < len(rebal_dates) else pd.Timestamp(end)
        blocks.append((train_end, test_start, test_end))

    return blocks


# ---------------------------------------------------------------------------
# 3. Preparação de dados
# ---------------------------------------------------------------------------

def prepare_ablation_data(
    asset:      str,
    er:         pd.DataFrame,
    rf:         pd.Series,
    fred:       pd.DataFrame,
    ohlc_cache: Optional[Dict[str, pd.DataFrame]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray], np.ndarray]:
    """
    Prepara todos os dados necessários para o ablation study.

    Implementa §3.1 do Protocolo Experimental.

    Parameters
    ----------
    asset      : nome do ativo
    er         : excess returns (dates × assets)
    rf         : taxa livre de risco diária
    fred       : features macro alinhadas
    ohlc_cache : cache de dados OHLC por ticker (evita downloads repetidos)

    Returns
    -------
    Tuple com:
      - ohlc          : pd.DataFrame com OHLC do ativo
      - features      : pd.DataFrame com features pré-computadas
      - vol_estimators: dict de np.ndarray com cada estimador de volatilidade
      - true_regimes  : np.ndarray com regimes de consenso (ground truth)
    """
    from src.config.settings import ASSET_TICKERS
    ticker = ASSET_TICKERS.get(asset, asset)

    # --- OHLC ---
    if ohlc_cache and asset in ohlc_cache:
        ohlc = ohlc_cache[asset]
    else:
        try:
            import yfinance as yf
            raw = yf.download(
                ticker,
                start=er.index[0].strftime("%Y-%m-%d"),
                end=er.index[-1].strftime("%Y-%m-%d"),
                auto_adjust=False,
                progress=False,
                multi_level_index=False,
            )
            if raw.empty:
                raise ValueError(f"Sem dados OHLC para {ticker}")

            raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]
            ohlc = raw[["open", "high", "low", "close"]].dropna()
            ohlc.index = pd.to_datetime(ohlc.index)
        except Exception as exc:
            logger.warning("OHLC download falhou para %s: %s. Usando Close apenas.", asset, exc)
            close = (1 + er[asset]).cumprod()
            close.name = "close"
            ohlc = pd.DataFrame({
                "open":  close.shift(1).bfill(),
                "high":  close * 1.001,
                "low":   close * 0.999,
                "close": close,
            })

    # Alinhar ao índice de er
    ohlc = ohlc.reindex(er.index).ffill().dropna()

    o = ohlc["open"].values
    h = ohlc["high"].values
    l = ohlc["low"].values
    c = ohlc["close"].values

    # --- Estimadores de volatilidade (comprimento = ohlc) ---
    vol_estimators: Dict[str, np.ndarray] = {
        "close_to_close":   rolling_close_to_close(c,          window=21),
        "parkinson":        rolling_parkinson(h, l,            window=21),
        "garman_klass":     rolling_garman_klass(o, h, l, c,  window=21),
        "rogers_satchell":  rolling_rogers_satchell(o, h, l, c, window=21),
        "yang_zhang":       rolling_yang_zhang(o, h, l, c,    window=21),
    }
    # Alinha ao índice completo de er (run_single_ablation usa er.index)
    vol_idx = ohlc.index
    for key, arr in list(vol_estimators.items()):
        s = pd.Series(arr, index=vol_idx).reindex(er.index).ffill().bfill()
        vol_estimators[key] = s.values.astype(np.float64, copy=False)

    # --- Features padrão ---
    ret_builder   = ReturnFeatureBuilder()
    macro_builder = MacroFeatureBuilder()

    ret_feats  = ret_builder.build(er[asset], asset, for_jm=True)
    macro_feats= macro_builder.build(fred, er)
    features   = ret_feats.join(macro_feats, how="left").ffill()

    # --- Regimes de consenso (pseudo-ground-truth) ---
    # Usa consenso de 5 fits com diferentes seeds
    true_regimes = _generate_consensus_regimes(er[asset], features, n_methods=5)

    return ohlc, features, vol_estimators, true_regimes


def _generate_consensus_regimes(
    returns:  pd.Series,
    features: pd.DataFrame,
    n_methods: int = 5,
) -> np.ndarray:
    """
    Gera regimes de consenso como pseudo-ground-truth.

    Faz n_methods fits do JM com diferentes seeds e calcula a moda.
    """
    n = len(returns)
    X = features.reindex(returns.index).dropna()
    votes = np.zeros((len(X), n_methods), dtype=int)

    er_arr  = returns.reindex(X.index).values
    labels_all = []

    for seed in range(n_methods):
        jm = JumpModel(n_states=2, jump_pen=50.0, random_state=seed)
        jm.fit(X.values)
        stats = jm.regime_stats(er_arr)
        bull  = max(stats, key=lambda k: stats[k]["mean_daily"])
        lbl   = (jm.labels_ != bull).astype(int)
        labels_all.append(lbl)
        votes[:, seed] = lbl

    consensus = (votes.mean(axis=1) >= 0.5).astype(int)
    cons = pd.Series(consensus, index=X.index).reindex(features.index).ffill().bfill()
    return cons.fillna(0).astype(int).values


# ---------------------------------------------------------------------------
# 4. Execução de uma única configuração  (§3.2)
# ---------------------------------------------------------------------------

def run_single_ablation(
    config:        AblationConfig,
    asset:         str,
    er:            pd.DataFrame,
    rf:            pd.Series,
    features:      pd.DataFrame,
    vol_estimators: Dict[str, np.ndarray],
    true_regimes:  np.ndarray,
    seed:          int = 0,
    test_start:    str = TEST_START,
    test_end:      str = TEST_END,
) -> AblationResult:
    """
    Executa uma única configuração de ablation.

    Implementa §3.2 do Protocolo Experimental.

    Parameters
    ----------
    config         : configuração do experimento
    asset          : ativo alvo
    er             : excess returns
    rf             : taxa livre de risco
    features       : features pré-computadas
    vol_estimators : dict de estimadores de volatilidade
    true_regimes   : regimes de referência (pseudo-ground-truth)
    seed           : semente aleatória
    test_start/end : período de teste
    """
    np.random.seed(seed)
    t0 = time.perf_counter()

    result = AblationResult(
        ablation_id=config.ablation_id,
        asset=asset,
        config=config.name,
        seed=seed,
    )

    try:
        idx = er.index
        rebal_months = _rebal_months_from_freq(config.recal_frequency)
        blocks = _generate_refit_dates(idx, test_start, test_end, rebal_months, config.train_years)

        if not blocks:
            logger.warning("[%s/%s] Sem blocos de rebalanceamento.", asset, config.name)
            return result

        # Obtém série de volatilidade do estimador configurado
        vol_key = config.vol_estimator
        vol_series = vol_estimators.get(vol_key, vol_estimators["close_to_close"])
        vol_s = pd.Series(vol_series, index=er.index).reindex(idx).ffill()

        # Prepara features enriquecidas com volatilidade
        feat_enriched = _prepare_clustering_features(features, vol_s, config.feature_set)

        all_preds: List[np.ndarray] = []
        all_true:  List[np.ndarray] = []
        all_returns: List[np.ndarray] = []
        # Expanding window com recalibração
        for train_end, test_start_bl, test_end_bl in blocks:
            train_idx = idx[(idx <= train_end)]
            test_idx  = idx[(idx >= test_start_bl) & (idx <= test_end_bl)]

            if len(train_idx) < 252 or len(test_idx) == 0:
                continue

            # ------ Stage 1: Regime Identification ------
            X_train = feat_enriched.reindex(train_idx).dropna()
            if len(X_train) < 100:
                continue

            t_fit_start = time.perf_counter()
            jm = JumpModel(
                n_states   = config.n_regimes,
                jump_pen   = config.lambda_penalty,
                random_state = seed,
            )
            jm.fit(X_train.values)
            fit_time = time.perf_counter() - t_fit_start

            er_train   = er[asset].reindex(X_train.index).values
            stats_dict = jm.regime_stats(er_train)
            bull_state = max(stats_dict, key=lambda k: stats_dict[k]["mean_daily"])
            jm_labels  = (jm.labels_ != bull_state).astype(int)

            # ------ Stage 2: Forecasting ------
            t_infer_start = time.perf_counter()
            X_test = feat_enriched.reindex(test_idx).ffill().dropna()
            if len(X_test) == 0:
                continue

            predicted = _forecast_regime(
                config       = config,
                X_train      = X_train,
                y_train      = jm_labels,
                X_test       = X_test,
                seed         = seed,
            )
            infer_time_ms = (time.perf_counter() - t_infer_start) * 1000

            # ------ Stage 3: Allocation ------
            er_test = er[asset].reindex(test_idx).fillna(0.0).values
            port_ret = _simple_portfolio(
                pred_labels  = predicted,
                er           = er_test,
                gamma_trade  = config.gamma_trade,
                gamma_risk   = config.gamma_risk,
                leverage_max = config.leverage_max,
            )

            # Alinha true_regimes ao test_idx (true_regimes tem len = features.index, não feat_enriched)
            true_block = _align_true_regimes(true_regimes, features.index, test_idx)

            all_preds.append(predicted)
            all_true.append(true_block)
            all_returns.append(port_ret)

        if not all_preds:
            return result

        # Agrega métricas
        preds_all  = np.concatenate(all_preds)
        true_all   = np.concatenate(all_true)
        rets_all   = np.concatenate(all_returns)
        rf_daily   = float(rf.mean())

        # Métricas de detecção
        diag = regime_diagnostics_summary(true_all, preds_all)
        result.add                = diag["ADD"]
        result.miss_rate          = diag["MissRate"]
        result.false_alarm_rate   = diag["FAR"]
        result.cp_accuracy        = diag["CP_Accuracy"]
        result.cp_balanced_accuracy = diag["CP_BalancedAccuracy"]
        result.regime_ari         = diag["ARI"]
        result.regime_agreement   = diag["Concordance"]
        result.mean_run_length    = diag["MRL_all"]

        # Métricas de classificação
        result.accuracy = diag["StateAccuracy"]
        result.state_accuracy = diag["StateAccuracy"]
        result.state_balanced_accuracy = diag["StateBalancedAccuracy"]
        from sklearn.metrics import f1_score
        result.f1_score = float(f1_score(true_all, preds_all, average="binary", zero_division=0))

        # Métricas de portfólio
        pm = compute_metrics_array(rets_all, rf_daily=rf_daily, ann_factor=TRADING_DAYS_YEAR)
        result.sortino_ratio = pm["Sortino"]
        result.sharpe_ratio  = pm["Sharpe"]
        result.max_drawdown  = pm["MDD"]
        result.calmar_ratio  = pm["Calmar"]
        result.total_return  = float(np.sum(rets_all) * TRADING_DAYS_YEAR / len(rets_all)) if len(rets_all) else np.nan
        result.volatility    = float(np.std(rets_all, ddof=1) * np.sqrt(TRADING_DAYS_YEAR))

        # Wealth curve: valor final de cumprod(1 + ret), iniciando em 1.0.
        # Split walk-forward: primeiro V% do período vira janela de validação
        # (para seleção sem leakage) e o restante vira holdout (para reporting).
        result.terminal_wealth = float(np.prod(1.0 + rets_all))
        if len(rets_all) >= 2:
            split = max(1, int(round(len(rets_all) * W1_VALIDATION_FRACTION)))
            split = min(split, len(rets_all) - 1)
            result.terminal_wealth_val = float(np.prod(1.0 + rets_all[:split]))
            result.terminal_wealth_oos = float(np.prod(1.0 + rets_all[split:]))

        # Turnover e número absoluto de trocas de posição (diagnóstico de instabilidade)
        pos_changes = np.abs(np.diff(np.concatenate([[0], preds_all])))
        result.turnover = float(pos_changes.mean() * TRADING_DAYS_YEAR)
        result.n_position_switches = float(pos_changes.sum())

        # Tempo computacional
        result.training_time_seconds = time.perf_counter() - t0
        result.inference_time_ms     = infer_time_ms if all_preds else np.nan

    except Exception as exc:
        logger.warning("[%s/%s/seed=%d] Falhou: %s", asset, config.name, seed, exc)

    return result


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _prepare_clustering_features(
    features:    pd.DataFrame,
    vol_series:  pd.Series,
    feature_set: str,
) -> pd.DataFrame:
    """
    Combina features de retorno com estimador de volatilidade.

    feature_set controla quantas features são incluídas:
      - minimal   : apenas retornos com lag
      - standard  : retornos + volatilidade (configuração base)
      - extended  : standard + momentum
      - kitchen_sink : todas as disponíveis
    """
    base = features.copy()
    base["vol"] = vol_series.reindex(base.index).ffill()

    if feature_set == "minimal":
        ret_cols = [c for c in base.columns if "avg_ret" in c]
        return base[ret_cols].dropna()
    elif feature_set == "standard":
        return base.dropna()
    elif feature_set == "extended":
        # Adiciona momentum (retorno cumulativo 21 dias)
        if hasattr(base, "_er_ref"):
            pass  # placeholder para expansão
        return base.dropna()
    else:  # kitchen_sink
        return base.dropna()


def _forecast_regime(
    config:   AblationConfig,
    X_train:  pd.DataFrame,
    y_train:  np.ndarray,
    X_test:   pd.DataFrame,
    seed:     int = 0,
) -> np.ndarray:
    """
    Gera previsões de regime com o modelo especificado no config.

    Implementa a lógica de forecasting do §3.2 do protocolo.
    """
    model_type = config.forecaster_type.lower()

    if model_type == "persistence":
        # Prediz o último label conhecido
        last = int(y_train[-1]) if len(y_train) > 0 else 0
        return np.full(len(X_test), last, dtype=int)

    # Label shifted +1 para treino supervisionado
    y_shifted = np.roll(y_train, -1)
    y_shifted[-1] = y_train[-1]
    valid = ~np.isnan(y_shifted.astype(float))

    if valid.sum() < 20:
        return np.full(len(X_test), 0, dtype=int)

    clf = _get_forecaster(config)
    try:
        t0 = time.perf_counter()
        clf.fit(X_train.values[valid], y_shifted[valid].astype(int))
        pred = clf.predict(X_test.values).astype(int)
    except Exception as exc:
        logger.debug("Forecaster %s falhou: %s", model_type, exc)
        return np.full(len(X_test), 0, dtype=int)

    return pred


def _get_forecaster(config: AblationConfig):
    """Instancia o classificador sklearn/xgboost de acordo com o config."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    mtype = config.forecaster_type.lower()
    if mtype == "logistic_regression":
        return LogisticRegression(max_iter=500, random_state=42)
    elif mtype == "decision_tree":
        return DecisionTreeClassifier(max_depth=config.max_depth, random_state=42)
    elif mtype == "random_forest":
        return RandomForestClassifier(n_estimators=config.n_estimators, random_state=42, n_jobs=1)
    elif mtype == "xgboost":
        return XGBClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=0.3,
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
            verbosity=0,
        )
    else:
        raise ValueError(f"Modelo desconhecido: {config.forecaster_type}")


def _simple_portfolio(
    pred_labels: np.ndarray,
    er:          np.ndarray,
    gamma_trade: float,
    gamma_risk:  float,
    leverage_max: float,
) -> np.ndarray:
    """
    Regra de alocação reduzida (single-asset) para o ablation study.

    - Sinal: `pred_labels[t]` indica bear (=1) vs bull (=0).
    - Exposição: em bull, mantém uma fração do capital no ativo; em bear, zera.
    - `gamma_risk` controla o *risk budget*: maior aversão a risco → menor exposição.
    - `gamma_trade` controla a severidade do custo de transação (via penalização
      de mudança de posição).

    Observação: Este projeto replica a lógica econômica do Stage 3 (trade-off
    risco × retorno + fricção de trading) em um setting single-asset para tornar
    as ablações marginais bem definidas e computacionalmente baratas.
    """
    pred_arr = np.asarray(pred_labels, dtype=np.float64)
    er_arr = np.asarray(er, dtype=np.float64)
    gamma_trade_val = float(gamma_trade) if gamma_trade is not None else 0.0
    gamma_risk_val = float(gamma_risk) if gamma_risk is not None else np.nan
    leverage_max_val = float(leverage_max) if leverage_max is not None else 0.0
    return simple_portfolio_jit(
        pred_arr,
        er_arr,
        gamma_trade_val,
        gamma_risk_val,
        leverage_max_val,
        float(TRANSACTION_COST),
    )


def _align_true_regimes(
    true_regimes: np.ndarray,
    feat_index:   pd.DatetimeIndex,
    test_idx:     pd.DatetimeIndex,
) -> np.ndarray:
    """Alinha o array de true_regimes (comprimento do feat_index) ao test_idx."""
    full = pd.Series(true_regimes, index=feat_index)
    aligned = full.reindex(test_idx).ffill().fillna(0)
    return aligned.values.astype(int)


# ---------------------------------------------------------------------------
# 5. Varredura completa de um componente  (§3.3)
# ---------------------------------------------------------------------------

def run_ablation_sweep(
    ablation_id:  str,
    assets:       List[str],
    er:           pd.DataFrame,
    rf:           pd.Series,
    fred:         pd.DataFrame,
    n_bootstrap:  int = N_BOOTSTRAP,
    test_start:   str = TEST_START,
    test_end:     str = TEST_END,
    n_jobs:       int = -1,
    checkpoint_dir: Optional[str] = None,
) -> pl.DataFrame:
    """
    Executa varredura completa de ablation para um componente.

    Implementa §3.3 do Protocolo Experimental.

    Parameters
    ----------
    ablation_id  : identificador da ablation (A1, A2, B1, C1, C2, D1, I1, I2)
    assets       : lista de ativos
    er, rf, fred : dados do pipeline base
    n_bootstrap  : replicações bootstrap
    n_jobs       : paralelismo across ativos (-1 = todos os cores)
    checkpoint_dir : diretório para checkpoints

    Returns
    -------
    pl.DataFrame com todos os resultados
    """
    configs = get_ablation_configs(ablation_id)
    logger.info("[%s] Iniciando: %d configs × %d assets × %d seeds = %d runs",
                ablation_id, len(configs), len(assets), n_bootstrap,
                len(configs) * len(assets) * n_bootstrap)

    # Checkpointing
    completed: Set[Tuple] = set()
    if checkpoint_dir:
        ckpt_file = Path(checkpoint_dir) / f"ablation_{ablation_id}_checkpoint.pkl"
        if ckpt_file.exists():
            with open(ckpt_file, "rb") as f:
                completed = pickle.load(f)
            logger.info("[%s] Retomando checkpoint: %d runs concluídas.", ablation_id, len(completed))
    else:
        ckpt_file = None

    def _run_asset(asset: str) -> List[AblationResult]:
        """Executa todas as configurações para um ativo."""
        ohlc, features, vol_estimators, true_regimes = prepare_ablation_data(
            asset, er, rf, fred
        )
        asset_results = []

        for config in configs:
            for seed in range(n_bootstrap):
                run_id = (asset, config.name, seed)
                if run_id in completed:
                    continue

                res = run_single_ablation(
                    config         = config,
                    asset          = asset,
                    er             = er,
                    rf             = rf,
                    features       = features,
                    vol_estimators = vol_estimators,
                    true_regimes   = true_regimes,
                    seed           = seed,
                    test_start     = test_start,
                    test_end       = test_end,
                )
                asset_results.append(res)
                completed.add(run_id)

        # Checkpoint intermediário
        if ckpt_file and len(completed) % 100 == 0:
            with open(ckpt_file, "wb") as f:
                pickle.dump(completed, f)

        return asset_results

    # Paralelização across ativos
    all_asset_results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_run_asset)(asset) for asset in assets
    )

    # Flatten e converter para Polars
    flat = [r.to_dict() for sublist in all_asset_results for r in sublist]
    return pl.DataFrame(flat)


# ---------------------------------------------------------------------------
# 6. Estudo completo com todas as ablations  (§7.2)
# ---------------------------------------------------------------------------

def run_full_ablation_study(
    er:            pd.DataFrame,
    rf:            pd.Series,
    fred:          pd.DataFrame,
    ablation_ids:  Optional[List[str]] = None,
    assets:        Optional[List[str]] = None,
    n_bootstrap:   int = N_BOOTSTRAP,
    n_jobs:        int = -1,
    results_dir:   str = "results/ablation",
    checkpoint_dir: str = "checkpoints",
) -> Dict[str, pl.DataFrame]:
    """
    Executa o ablation study completo com paralelização e checkpointing.

    Implementa §7.2 do Protocolo Experimental.

    Parameters
    ----------
    ablation_ids : lista de IDs a executar (default: todos)
    assets       : lista de ativos (default: todos os 12)
    results_dir  : diretório para salvar resultados em Parquet
    """
    if ablation_ids is None:
        ablation_ids = ["A1", "A2", "A3", "B1", "B2", "C1", "C2", "D1", "I1", "I2", "W1"]
    if assets is None:
        assets = list(ASSETS)

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, pl.DataFrame] = {}

    for ablation_id in ablation_ids:
        logger.info("\n%s\nRunning Ablation %s\n%s", "="*60, ablation_id, "="*60)

        out_path = Path(results_dir) / f"ablation_{ablation_id}.parquet"
        if out_path.exists():
            logger.info("[%s] Carregando resultados existentes de %s", ablation_id, out_path)
            all_results[ablation_id] = pl.read_parquet(out_path)
            continue

        df = run_ablation_sweep(
            ablation_id    = ablation_id,
            assets         = assets,
            er             = er,
            rf             = rf,
            fred           = fred,
            n_bootstrap    = n_bootstrap,
            n_jobs         = n_jobs,
            checkpoint_dir = checkpoint_dir,
        )

        df.write_parquet(out_path)
        all_results[ablation_id] = df
        logger.info("[%s] Concluído. Resultados salvos em %s", ablation_id, out_path)

    return all_results


# ---------------------------------------------------------------------------
# 7. Configurações das ablations  (§5)
# ---------------------------------------------------------------------------

# Mapeamento de nomes de componente
COMPONENT_NAMES = {
    "A1": "Jump Penalty λ",
    "A2": "Volatility Estimator",
    "A3": "Number of Regimes k",
    "B1": "Forecasting Model",
    "B2": "Feature Set",
    "C1": "Risk Aversion γ_risk",
    "C2": "Trade Aversion γ_trade",
    "D1": "Recalibration Frequency",
    "I1": "λ × Estimator (Interaction)",
    "I2": "γ_risk × γ_trade (Interaction)",
    "W1": "Joint Hyperparameter Sweep (wealth objective)",
}


BASELINE_CONFIG = AblationConfig(
    name             = "baseline",
    ablation_id      = "BASE",
    lambda_penalty   = 50.0,
    n_regimes        = 2,
    vol_estimator    = "close_to_close",
    forecaster_type  = "xgboost",
    feature_set      = "standard",
    gamma_risk       = 10.0,
    gamma_trade      = 1.0,
    recal_frequency  = "semi-annual",
)


# --- Configurações A1: Jump Penalty ---
ABLATION_A1_CONFIGS: List[AblationConfig] = [
    AblationConfig(name="lambda_10",  ablation_id="A1", lambda_penalty=10.0),
    AblationConfig(name="lambda_25",  ablation_id="A1", lambda_penalty=25.0),
    AblationConfig(name="lambda_50",  ablation_id="A1", lambda_penalty=50.0),   # baseline
    AblationConfig(name="lambda_75",  ablation_id="A1", lambda_penalty=75.0),
    AblationConfig(name="lambda_100", ablation_id="A1", lambda_penalty=100.0),
    AblationConfig(name="lambda_150", ablation_id="A1", lambda_penalty=150.0),
    AblationConfig(name="lambda_200", ablation_id="A1", lambda_penalty=200.0),
]

# --- Configurações A2: Volatility Estimator ---
ABLATION_A2_CONFIGS: List[AblationConfig] = [
    AblationConfig(name="vol_cc",       ablation_id="A2", vol_estimator="close_to_close"),   # baseline
    AblationConfig(name="vol_parkinson",ablation_id="A2", vol_estimator="parkinson"),
    AblationConfig(name="vol_gk",       ablation_id="A2", vol_estimator="garman_klass"),
    AblationConfig(name="vol_rs",       ablation_id="A2", vol_estimator="rogers_satchell"),
    AblationConfig(name="vol_yz",       ablation_id="A2", vol_estimator="yang_zhang"),
]

# --- Configurações A3: Number of Regimes ---
ABLATION_A3_CONFIGS: List[AblationConfig] = [
    AblationConfig(name="k2", ablation_id="A3", n_regimes=2),   # baseline
    AblationConfig(name="k3", ablation_id="A3", n_regimes=3),
    AblationConfig(name="k4", ablation_id="A3", n_regimes=4),
]

# --- Configurações B1: Forecasting Model ---
ABLATION_B1_CONFIGS: List[AblationConfig] = [
    AblationConfig(name="persistence",  ablation_id="B1", forecaster_type="persistence"),
    AblationConfig(name="logistic",     ablation_id="B1", forecaster_type="logistic_regression"),
    AblationConfig(name="tree_d3",      ablation_id="B1", forecaster_type="decision_tree", max_depth=3),
    AblationConfig(name="tree_d5",      ablation_id="B1", forecaster_type="decision_tree", max_depth=5),
    AblationConfig(name="rf_100",       ablation_id="B1", forecaster_type="random_forest", n_estimators=100),
    AblationConfig(name="xgb_default",  ablation_id="B1", forecaster_type="xgboost"),   # baseline
]

# --- Configurações B2: Feature Set ---
ABLATION_B2_CONFIGS: List[AblationConfig] = [
    AblationConfig(name="minimal",     ablation_id="B2", feature_set="minimal"),
    AblationConfig(name="standard",    ablation_id="B2", feature_set="standard"),   # baseline
    AblationConfig(name="extended",    ablation_id="B2", feature_set="extended"),
    AblationConfig(name="kitchen_sink",ablation_id="B2", feature_set="kitchen_sink"),
]

# --- Configurações C1: Risk Aversion ---
ABLATION_C1_CONFIGS: List[AblationConfig] = [
    AblationConfig(name="grisk_5",  ablation_id="C1", gamma_risk=5.0),
    AblationConfig(name="grisk_10", ablation_id="C1", gamma_risk=10.0),   # baseline
    AblationConfig(name="grisk_15", ablation_id="C1", gamma_risk=15.0),
    AblationConfig(name="grisk_20", ablation_id="C1", gamma_risk=20.0),
    AblationConfig(name="grisk_30", ablation_id="C1", gamma_risk=30.0),
]

# --- Configurações C2: Trade Aversion ---
ABLATION_C2_CONFIGS: List[AblationConfig] = [
    AblationConfig(name="gtrade_0",   ablation_id="C2", gamma_trade=0.0),   # baseline
    AblationConfig(name="gtrade_0.5", ablation_id="C2", gamma_trade=0.5),
    AblationConfig(name="gtrade_1",   ablation_id="C2", gamma_trade=1.0),
    AblationConfig(name="gtrade_2",   ablation_id="C2", gamma_trade=2.0),
    AblationConfig(name="gtrade_5",   ablation_id="C2", gamma_trade=5.0),
]

# --- Configurações D1: Recalibration Frequency ---
ABLATION_D1_CONFIGS: List[AblationConfig] = [
    AblationConfig(name="recal_monthly",    ablation_id="D1", recal_frequency="monthly"),
    AblationConfig(name="recal_quarterly",  ablation_id="D1", recal_frequency="quarterly"),
    AblationConfig(name="recal_semiannual", ablation_id="D1", recal_frequency="semi-annual"),   # baseline
    AblationConfig(name="recal_annual",     ablation_id="D1", recal_frequency="annual"),
]

# --- Configurações I1: λ × Estimator (Interação) ---
ABLATION_I1_CONFIGS: List[AblationConfig] = [
    AblationConfig(name="i1_l25_cc",  ablation_id="I1", lambda_penalty=25.0, vol_estimator="close_to_close"),
    AblationConfig(name="i1_l25_pk",  ablation_id="I1", lambda_penalty=25.0, vol_estimator="parkinson"),
    AblationConfig(name="i1_l25_yz",  ablation_id="I1", lambda_penalty=25.0, vol_estimator="yang_zhang"),
    AblationConfig(name="i1_l50_cc",  ablation_id="I1", lambda_penalty=50.0, vol_estimator="close_to_close"),
    AblationConfig(name="i1_l50_pk",  ablation_id="I1", lambda_penalty=50.0, vol_estimator="parkinson"),
    AblationConfig(name="i1_l50_yz",  ablation_id="I1", lambda_penalty=50.0, vol_estimator="yang_zhang"),
    AblationConfig(name="i1_l100_cc", ablation_id="I1", lambda_penalty=100.0, vol_estimator="close_to_close"),
    AblationConfig(name="i1_l100_pk", ablation_id="I1", lambda_penalty=100.0, vol_estimator="parkinson"),
    AblationConfig(name="i1_l100_yz", ablation_id="I1", lambda_penalty=100.0, vol_estimator="yang_zhang"),
]

# --- Configurações I2: γ_risk × γ_trade (Interação) ---
ABLATION_I2_CONFIGS: List[AblationConfig] = [
    AblationConfig(name="i2_gr5_gt0",  ablation_id="I2", gamma_risk=5.0,  gamma_trade=0.0),
    AblationConfig(name="i2_gr5_gt1",  ablation_id="I2", gamma_risk=5.0,  gamma_trade=1.0),
    AblationConfig(name="i2_gr5_gt2",  ablation_id="I2", gamma_risk=5.0,  gamma_trade=2.0),
    AblationConfig(name="i2_gr10_gt0", ablation_id="I2", gamma_risk=10.0, gamma_trade=0.0),
    AblationConfig(name="i2_gr10_gt1", ablation_id="I2", gamma_risk=10.0, gamma_trade=1.0),
    AblationConfig(name="i2_gr10_gt2", ablation_id="I2", gamma_risk=10.0, gamma_trade=2.0),
    AblationConfig(name="i2_gr20_gt0", ablation_id="I2", gamma_risk=20.0, gamma_trade=0.0),
    AblationConfig(name="i2_gr20_gt1", ablation_id="I2", gamma_risk=20.0, gamma_trade=1.0),
    AblationConfig(name="i2_gr20_gt2", ablation_id="I2", gamma_risk=20.0, gamma_trade=2.0),
]

# --- Configurações W1: Joint sweep para maximizar wealth terminal ---
# Dimensões: jump_penalty × vol_estimator × n_regimes × recal_frequency × forecaster_type.
# Valores derivados das ablações marginais (A1, A2, A3, B1, D1), reduzidos para caber em W1_BUDGET_MAX.
_W1_LAMBDAS        = [25.0, 50.0, 100.0]
_W1_VOL_ESTIMATORS = ["close_to_close", "yang_zhang"]
_W1_N_REGIMES      = [2, 3]
_W1_RECAL_FREQS    = ["monthly", "quarterly", "semi-annual", "annual"]
_W1_FORECASTERS    = ["logistic_regression", "random_forest", "xgboost"]

_W1_VOL_ABBR  = {"close_to_close": "cc", "parkinson": "pk", "yang_zhang": "yz",
                 "garman_klass": "gk", "rogers_satchell": "rs"}
_W1_RECAL_ABBR = {"monthly": "mon", "quarterly": "qua", "semi-annual": "sem", "annual": "ann"}
_W1_FC_ABBR   = {"logistic_regression": "log", "random_forest": "rf", "xgboost": "xgb",
                 "decision_tree": "tree", "persistence": "pers"}


def _build_w1_configs(budget_max: int = W1_BUDGET_MAX) -> List[AblationConfig]:
    """
    Monta o grid W1 verificando o budget máximo de combinações.

    Estratégia: produto cartesiano determinístico. Se o produto exceder
    ``budget_max``, levanta ``ValueError`` com orientação para o usuário
    reduzir explicitamente alguma dimensão. Mantém a seleção reprodutível
    e documentada (sem amostragem estocástica implícita).
    """
    n_total = (len(_W1_LAMBDAS)
               * len(_W1_VOL_ESTIMATORS)
               * len(_W1_N_REGIMES)
               * len(_W1_RECAL_FREQS)
               * len(_W1_FORECASTERS))
    if n_total > budget_max:
        raise ValueError(
            f"Grid W1 com {n_total} combinações excede budget={budget_max}. "
            f"Reduza explicitamente alguma dimensão em _W1_* ou ajuste W1_BUDGET_MAX."
        )

    configs: List[AblationConfig] = []
    for lam in _W1_LAMBDAS:
        for vol in _W1_VOL_ESTIMATORS:
            for k in _W1_N_REGIMES:
                for recal in _W1_RECAL_FREQS:
                    for fc in _W1_FORECASTERS:
                        name = (
                            f"w1_l{int(lam)}_"
                            f"{_W1_VOL_ABBR.get(vol, vol)}_"
                            f"k{k}_"
                            f"{_W1_RECAL_ABBR.get(recal, recal)}_"
                            f"{_W1_FC_ABBR.get(fc, fc)}"
                        )
                        configs.append(AblationConfig(
                            name            = name,
                            ablation_id     = "W1",
                            lambda_penalty  = lam,
                            vol_estimator   = vol,
                            n_regimes       = k,
                            recal_frequency = recal,
                            forecaster_type = fc,
                        ))
    return configs


ABLATION_W1_CONFIGS: List[AblationConfig] = _build_w1_configs()


# Mapa global: ablation_id → lista de configs
ABLATION_CONFIG_MAP: Dict[str, List[AblationConfig]] = {
    "A1": ABLATION_A1_CONFIGS,
    "A2": ABLATION_A2_CONFIGS,
    "A3": ABLATION_A3_CONFIGS,
    "B1": ABLATION_B1_CONFIGS,
    "B2": ABLATION_B2_CONFIGS,
    "C1": ABLATION_C1_CONFIGS,
    "C2": ABLATION_C2_CONFIGS,
    "D1": ABLATION_D1_CONFIGS,
    "I1": ABLATION_I1_CONFIGS,
    "I2": ABLATION_I2_CONFIGS,
    "W1": ABLATION_W1_CONFIGS,
}


def get_ablation_configs(ablation_id: str) -> List[AblationConfig]:
    """Retorna a lista de configurações para um dado ablation_id."""
    if ablation_id not in ABLATION_CONFIG_MAP:
        raise ValueError(f"Ablation '{ablation_id}' desconhecido. Disponíveis: {list(ABLATION_CONFIG_MAP)}")
    return ABLATION_CONFIG_MAP[ablation_id]


def get_component_name(ablation_id: str) -> str:
    """Retorna o nome legível do componente para o ablation_id."""
    return COMPONENT_NAMES.get(ablation_id, ablation_id)


# ---------------------------------------------------------------------------
# 8. Análise estatística de uma ablation  (§4.1)
# ---------------------------------------------------------------------------

def best_config_argmax(
    results:        pl.DataFrame,
    metric:         str = "terminal_wealth",
    tiebreak_cols:  Optional[List[Tuple[str, bool]]] = None,
    higher_is_better: bool = True,
) -> Dict[str, Any]:
    """
    Seleciona a configuração argmax de ``metric`` (média sobre asset × seed).

    Diferente de ``analyze_ablation()``, que compara cada config contra um
    baseline via ``mean_diff``, este helper faz a seleção direta ``argmax``
    da média da métrica, com tie-break determinístico. É a semântica correta
    para o sweep conjunto W1, onde o alvo é a melhor combinação global.

    Parameters
    ----------
    results           : DataFrame com colunas 'config' + métricas.
    metric            : coluna-alvo (ex.: 'terminal_wealth_val' para seleção
                        walk-forward; 'terminal_wealth' para reporting).
    tiebreak_cols     : lista de (coluna, higher_is_better) usados como tie-break
                        na ordem informada. Default: drawdown (menos negativo é
                        melhor) e turnover (menor é melhor).
    higher_is_better  : direção do objetivo principal.

    Returns
    -------
    dict com:
      - config         : nome da config argmax
      - metric_value   : valor médio da métrica
      - details        : médias de métricas auxiliares (quando disponíveis)
      - ranking        : DataFrame pandas ordenado por ``metric``
    """
    if tiebreak_cols is None:
        tiebreak_cols = [("max_drawdown", True), ("turnover", False)]

    results = float_nan_to_null(results)
    if metric not in results.columns or results.is_empty():
        return {
            "config": None, "metric_value": float("nan"),
            "details": {}, "ranking": pd.DataFrame(),
        }

    # Agrega por config: média sobre assets × seeds para robustez
    agg_cols = [c for c, _ in tiebreak_cols if c in results.columns]
    agg_exprs = [pl.col(metric).mean().alias(metric)]
    for c in agg_cols:
        agg_exprs.append(pl.col(c).mean().alias(c))
    ranking = (
        results.group_by("config")
               .agg(agg_exprs)
    )

    # Ordenação canônica: colocar o MELHOR no topo (iloc[0]).
    # Para isso, ordenamos com ascending=True se a coluna é "lower is better",
    # e ascending=False se "higher is better".
    sort_cols: List[pl.Expr] = [pl.col(metric)]
    descending = [higher_is_better]
    for col, hib in tiebreak_cols:
        if col in results.columns:
            sort_cols.append(pl.col(col))
            descending.append(hib)
    ranking = ranking.sort(by=sort_cols, descending=descending)

    top_row = ranking.row(0, named=True)
    details = {c: float(top_row[c]) for c in agg_cols if c in top_row}

    return {
        "config":       str(top_row["config"]),
        "metric_value": float(top_row[metric]) if top_row[metric] is not None else float("nan"),
        "details":      details,
        "ranking":      ranking.to_pandas(),
    }


def _build_perf_matrix_polars(
    results: pl.DataFrame,
    metric: str,
) -> Tuple[pl.DataFrame, pd.DataFrame, List[str]]:
    """
    Agrega `metric` por asset × config em Polars e retorna a matriz wide.

    Returns
    -------
    perf_matrix_pl : pl.DataFrame wide com coluna 'asset' + uma coluna por config
    perf_matrix_pd : a mesma matriz em pandas, indexada por asset (compatibilidade)
    configs        : lista de configurações na ordem das colunas da matriz
    """
    perf_long = (
        results.group_by(["asset", "config"])
               .agg(pl.col(metric).mean().alias(metric))
               .sort(["asset", "config"])
    )
    perf_matrix_pl = (
        perf_long.pivot(
            values=metric,
            index="asset",
            on="config",
            aggregate_function="first",
        )
        .sort("asset")
    )
    configs = [c for c in perf_matrix_pl.columns if c != "asset"]
    perf_matrix_pd = perf_matrix_pl.to_pandas().set_index("asset") if configs else pd.DataFrame()
    return perf_matrix_pl, perf_matrix_pd, configs


def analyze_ablation(
    results:    pl.DataFrame,
    metric:     str = "add",
    alpha:      float = 0.05,
) -> Dict[str, Any]:
    """
    Análise estatística de uma ablation conforme §4.1 do protocolo.

    Implementa Friedman test + pairwise Wilcoxon + Holm correction + Cohen's d.

    Parameters
    ----------
    results : DataFrame com colunas 'asset', 'config', e a métrica
    metric  : coluna da métrica alvo (e.g. 'add', 'sortino_ratio')
    alpha   : nível de significância

    Returns
    -------
    dict com:
      - perf_matrix        : pivot (assets × configs) da métrica
      - friedman           : resultado do teste de Friedman
      - pairwise           : comparações pareadas vs. baseline
      - variance_decomp    : contribuição de config vs. asset para variância
    """
    from src.ablation.statistical_tests import (
        friedman_test, wilcoxon_test, holm_correction, cohens_d
    )
    from scipy.stats import friedmanchisquare

    results = float_nan_to_null(results)
    if metric not in results.columns:
        raise ValueError(f"Métrica '{metric}' ausente no DataFrame de resultados.")

    # Pivot: assets × configs (em Polars no caminho crítico; pandas só na borda)
    perf_matrix_pl, perf_matrix, configs = _build_perf_matrix_polars(results, metric)
    if not configs:
        return {
            "ablation_id":     results["ablation_id"][0] if "ablation_id" in results.columns else "",
            "metric":          metric,
            "perf_matrix":     perf_matrix,
            "friedman":        {"statistic": np.nan, "p_value": 1.0, "significant": False},
            "pairwise":        pd.DataFrame(),
            "variance_decomp": {"config_contribution": 0.0, "asset_contribution": 0.0, "residual": 1.0},
            "argmax":          best_config_argmax(results, metric=metric, higher_is_better=True),
        }

    baseline   = "baseline" if "baseline" in configs else configs[0]

    # Friedman: medidas repetidas — mesmos ativos em todas as condições
    perf_fried_pl = perf_matrix_pl.drop_nulls()
    if perf_fried_pl.height < 3:
        perf_fried_pl = perf_matrix_pl

    groups = [perf_fried_pl[c].to_numpy() for c in configs]
    groups = [g for g in groups if len(g) >= 3]
    fstat, fp = np.nan, 1.0
    if len(groups) == len(configs) and len(groups) >= 3:
        try:
            fstat, fp = friedmanchisquare(*groups)
        except Exception:
            fstat, fp = np.nan, 1.0

    friedman = {"statistic": fstat, "p_value": fp, "significant": fp < alpha}

    # 2. Pairwise comparisons vs. baseline (pareamento por ativo)
    pairwise_rows = []
    if fp < alpha and baseline in configs:
        perf_long = (
            results.group_by(["asset", "config"])
                   .agg(pl.col(metric).mean().alias(metric))
        )
        base_pair = (
            perf_long.filter(pl.col("config") == baseline)
                     .select(["asset", pl.col(metric).alias("baseline_value")])
        )
        for config in configs:
            if config == baseline:
                continue
            cfg_pair = (
                perf_long.filter(pl.col("config") == config)
                         .select(["asset", pl.col(metric).alias("config_value")])
            )
            pair = (
                base_pair.join(cfg_pair, on="asset", how="inner")
                         .drop_nulls()
                         .sort("asset")
            )
            if pair.height < 5:
                continue
            base_vals = pair["baseline_value"].to_numpy()
            cfg_vals = pair["config_value"].to_numpy()
            n = len(base_vals)
            test = wilcoxon_test(base_vals, cfg_vals, alpha=alpha)
            d    = cohens_d(cfg_vals, base_vals)
            pairwise_rows.append({
                "config":       config,
                "vs_baseline":  baseline,
                "mean_baseline":round(float(np.mean(base_vals[:n])), 4),
                "mean_config":  round(float(np.mean(cfg_vals[:n])),  4),
                "mean_diff":    round(float(np.mean(cfg_vals[:n]) - np.mean(base_vals[:n])), 4),
                "wilcoxon_stat":test["statistic"],
                "p_value":      test["p_value"],
                "cohens_d":     round(d, 3),
            })

    pairwise_df = pd.DataFrame(pairwise_rows)
    if not pairwise_df.empty:
        holm_df = holm_correction(
            pairwise_df["p_value"].tolist(),
            names=pairwise_df["config"].tolist(),
            alpha=alpha,
        ).to_pandas()
        pairwise_df = pairwise_df.merge(
            holm_df[["name", "p_adjusted", "reject_h0"]],
            left_on="config", right_on="name", how="left",
        ).drop(columns=["name"])
        pairwise_df["significant"] = pairwise_df["reject_h0"]

    # 3. Após pivot, variância amostral (ddof=1) exige ≥2 linhas/colunas não degeneradas
    total_var = float(results.select(pl.col(metric).var()).item()) if results.height > 1 else 0.0
    n_a, n_c = perf_matrix_pl.height, len(configs)
    if n_a < 2:
        config_var, asset_var = 0.0, 0.0
    elif n_c < 2:
        matrix = perf_matrix.to_numpy(dtype=float, copy=False)
        asset_var = float(np.nanmean(np.nanvar(matrix, axis=1, ddof=1)))
        config_var = 0.0
    else:
        matrix = perf_matrix.to_numpy(dtype=float, copy=False)
        config_var = float(np.nanmean(np.nanvar(matrix, axis=0, ddof=1)))
        asset_var = float(np.nanmean(np.nanvar(matrix, axis=1, ddof=1)))
        if not np.isfinite(config_var):
            config_var = 0.0
        if not np.isfinite(asset_var):
            asset_var = 0.0
    variance_decomp = {
        "config_contribution": float(config_var / total_var) if total_var > 0 else 0.0,
        "asset_contribution":  float(asset_var  / total_var) if total_var > 0 else 0.0,
        "residual": max(
            0.0,
            1.0 - (config_var + asset_var) / max(total_var, 1e-12),
        ),
    }

    # 4. Seleção direta por argmax (semântica independente do baseline).
    # Retorna a config com maior média da métrica + diagnósticos de risco
    # em tie-break. Usada pelo sweep conjunto W1 (sem baseline canônico).
    higher_is_better = metric not in {"add", "miss_rate", "false_alarm_rate",
                                      "max_drawdown", "volatility",
                                      "turnover", "n_position_switches"}
    argmax = best_config_argmax(results, metric=metric,
                                higher_is_better=higher_is_better)

    return {
        "ablation_id":     results["ablation_id"][0] if "ablation_id" in results.columns else "",
        "metric":          metric,
        "perf_matrix":     perf_matrix,
        "friedman":        friedman,
        "pairwise":        pairwise_df,
        "variance_decomp": variance_decomp,
        "argmax":          argmax,
    }


# ---------------------------------------------------------------------------
# 9. Comparação cross-ablation  (§4.2)
# ---------------------------------------------------------------------------

def compare_ablations(
    all_results: Dict[str, pl.DataFrame],
    metric:      str = "add",
) -> pl.DataFrame:
    """
    Compara o impacto de cada ablation no metric especificado.

    Implementa §4.2 do Protocolo Experimental.

    Returns
    -------
    pl.DataFrame com ranking de ablations por variância explicada.
    """
    impact_rows = []

    for ablation_id, results in all_results.items():
        analysis = analyze_ablation(results, metric)
        pw = analysis["pairwise"]

        if not pw.empty and "mean_diff" in pw.columns:
            best_idx = pw["mean_diff"].abs().idxmax()
            best_improvement = float(pw.loc[best_idx, "mean_diff"])
            best_config      = pw.loc[best_idx, "config"]
            max_d            = float(pw["cohens_d"].abs().max()) if "cohens_d" in pw.columns else 0.0
        else:
            best_improvement = 0.0
            best_config      = "baseline"
            max_d            = 0.0

        fp = analysis["friedman"]["p_value"]
        fp = float(fp) if np.isfinite(fp) else 1.0
        ve = analysis["variance_decomp"]["config_contribution"]
        ve = float(ve) if np.isfinite(ve) else 0.0

        # Seleção direta argmax (semântica diferente de mean_diff vs baseline):
        # usada para relatórios do sweep conjunto W1 / rankings pelo alvo financeiro.
        argmax_info = analysis.get("argmax") or {}
        argmax_config = argmax_info.get("config")
        argmax_value  = argmax_info.get("metric_value", float("nan"))

        impact_rows.append({
            "ablation_id":           ablation_id,
            "component":             get_component_name(ablation_id),
            "n_configs":             results["config"].n_unique(),
            "friedman_p":            round(fp, 4),
            "significant":           analysis["friedman"]["significant"],
            # Seleção vs baseline por mean_diff (comparativo clássico das ablações marginais)
            "best_config_vs_baseline": best_config,
            "mean_diff_vs_baseline":   round(best_improvement, 4),
            # Seleção direta argmax por métrica (alvo financeiro / W1)
            "argmax_config":           argmax_config,
            "argmax_metric_value":     (round(float(argmax_value), 4)
                                        if argmax_value is not None
                                        and np.isfinite(float(argmax_value)) else None),
            # Aliases retrocompatíveis: apontam para a semântica clássica (vs baseline).
            # Em W1 ou análises sem baseline canônico, prefira 'argmax_config'/'argmax_metric_value'.
            "best_config":       best_config,
            "best_improvement":  round(best_improvement, 4),
            "variance_explained":    round(ve, 4),
            "max_effect_size":       round(max_d, 3),
        })

    df = pl.DataFrame(impact_rows)
    df = df.sort("variance_explained", descending=True)
    df = df.with_columns(
        pl.int_range(1, pl.len() + 1, eager=False).alias("rank")
    )
    return df
