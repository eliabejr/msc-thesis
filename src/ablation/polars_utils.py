"""
ablation/polars_utils.py
=========================
Utilitários de dados usando Polars para o ablation study.

Funções:
  - series_to_polars          : pd.Series → pl.DataFrame
  - dataframe_to_polars       : pd.DataFrame → pl.DataFrame
  - load_regime_forecasts     : carrega regime_forecasts.pkl → pl.DataFrame
  - load_portfolio_results    : carrega portfolio_results.pkl → pl.DataFrame
  - build_ablation_summary    : consolida resultados de ablation em pl.DataFrame
  - rolling_metrics_polars    : métricas rolling usando Polars expressions
  - pivot_ablation_results    : pivot de resultados para visualização

Uso:
  >>> from src.ablation.polars_utils import load_regime_forecasts
  >>> df = load_regime_forecasts("results/regime_forecasts.pkl")
  >>> df.head()
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import polars as pl


# ---------------------------------------------------------------------------
# Conversores pandas → Polars
# ---------------------------------------------------------------------------

def series_to_polars(
    s:    pd.Series,
    name: Optional[str] = None,
    include_index: bool = True,
) -> pl.DataFrame:
    """
    Converte pd.Series → pl.DataFrame.

    Parameters
    ----------
    s             : série pandas
    name          : nome da coluna de valores (default: s.name ou 'value')
    include_index : se True, inclui o índice como coluna 'date'

    Returns
    -------
    pl.DataFrame com colunas ['date', <name>] ou apenas [<name>]
    """
    col_name = name or (s.name if s.name else "value")
    if include_index:
        return pl.DataFrame({
            "date":   list(s.index),
            col_name: s.values.tolist(),
        })
    return pl.DataFrame({col_name: s.values.tolist()})


def dataframe_to_polars(
    df: pd.DataFrame,
    date_col: Optional[str] = None,
) -> pl.DataFrame:
    """
    Converte pd.DataFrame → pl.DataFrame.

    Parameters
    ----------
    df       : DataFrame pandas
    date_col : se None e o índice for DatetimeIndex, o índice vira coluna 'date'

    Returns
    -------
    pl.DataFrame
    """
    if isinstance(df.index, pd.DatetimeIndex) and date_col is None:
        df = df.copy()
        df.index.name = "date"
        df = df.reset_index()
    return pl.from_pandas(df)


# ---------------------------------------------------------------------------
# Carregamento de artefatos do projeto base
# ---------------------------------------------------------------------------

def load_regime_forecasts(path: Union[str, Path]) -> pl.DataFrame:
    """
    Carrega regime_forecasts.pkl e retorna pl.DataFrame.

    O arquivo é um pd.DataFrame (dates × assets) com labels 0/1.

    Returns
    -------
    pl.DataFrame com colunas: date, {asset_names...}
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    # Aceita dict ou DataFrame
    if isinstance(data, dict):
        rf = data.get("regime_forecasts", data)
    else:
        rf = data

    if isinstance(rf, pd.DataFrame):
        return dataframe_to_polars(rf)

    raise TypeError(f"Formato inesperado em {path}: {type(rf)}")


def load_portfolio_results(path: Union[str, Path]) -> Dict[str, pl.DataFrame]:
    """
    Carrega portfolio_results.pkl e retorna dict de Polars DataFrames.

    Returns
    -------
    dict com chaves por estratégia, cada um pl.DataFrame com:
      date, portfolio_return, turnover, weights...
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    result: Dict[str, pl.DataFrame] = {}
    for strat_name, strat_data in data.items():
        if isinstance(strat_data, dict):
            # Converte cada sub-item
            frames: Dict[str, list] = {"date": []}
            for key, val in strat_data.items():
                if isinstance(val, pd.Series):
                    if not frames["date"]:
                        frames["date"] = list(val.index)
                    frames[key] = val.values.tolist()
                elif isinstance(val, pd.DataFrame):
                    for col in val.columns:
                        frames[f"{key}_{col}"] = val[col].values.tolist()
            result[strat_name] = pl.DataFrame(frames)
        elif isinstance(strat_data, pd.DataFrame):
            result[strat_name] = dataframe_to_polars(strat_data)

    return result


# ---------------------------------------------------------------------------
# Construção de tabelas de ablation
# ---------------------------------------------------------------------------

def build_ablation_summary(
    results: List[Dict[str, Any]],
    config_cols: List[str],
    metric_cols: List[str],
) -> pl.DataFrame:
    """
    Consolida lista de dicts de resultados de ablation em pl.DataFrame.

    Parameters
    ----------
    results     : lista de dicts, cada um com configuração + métricas
    config_cols : nomes das colunas de configuração (e.g. ['lambda', 'estimator'])
    metric_cols : nomes das colunas de métricas (e.g. ['ADD', 'Sortino', 'MDD'])

    Returns
    -------
    pl.DataFrame ordenado pelos config_cols

    Exemplo:
    --------
    >>> results = [
    ...     {"lambda": 10,  "ADD": 2.3, "Sortino": 0.31},
    ...     {"lambda": 50,  "ADD": 4.2, "Sortino": 0.42},
    ... ]
    >>> build_ablation_summary(results, ["lambda"], ["ADD", "Sortino"])
    """
    if not results:
        return pl.DataFrame()

    all_cols = config_cols + metric_cols
    data: Dict[str, list] = {col: [] for col in all_cols}

    for row in results:
        for col in all_cols:
            data[col].append(row.get(col, None))

    df = pl.DataFrame(data)
    if config_cols:
        df = df.sort(config_cols)
    return df


def pivot_ablation_results(
    df:          pl.DataFrame,
    index_col:   str,
    value_col:   str,
    column_col:  str,
) -> pl.DataFrame:
    """
    Pivot de resultados de ablation para comparação lado-a-lado.

    Parameters
    ----------
    df          : DataFrame de resultados
    index_col   : coluna que define as linhas (e.g. 'lambda')
    value_col   : coluna com os valores (e.g. 'ADD')
    column_col  : coluna que define as colunas (e.g. 'asset')

    Returns
    -------
    pl.DataFrame pivotado
    """
    return df.pivot(
        values=value_col,
        index=index_col,
        on=column_col,
        aggregate_function="mean",
    )


# ---------------------------------------------------------------------------
# Métricas rolling usando Polars
# ---------------------------------------------------------------------------

def rolling_metrics_polars(
    df:       pl.DataFrame,
    ret_col:  str,
    rf_col:   Optional[str] = None,
    window:   int = 63,
    ann:      int = 252,
) -> pl.DataFrame:
    """
    Calcula métricas rolling usando Polars expressions nativas.

    Retorna o DataFrame original com colunas adicionais:
      - rolling_sharpe_{window}d
      - rolling_sortino_{window}d
      - rolling_vol_{window}d
      - rolling_ret_{window}d

    Parameters
    ----------
    df      : Polars DataFrame com coluna de retornos
    ret_col : nome da coluna de retornos diários
    rf_col  : nome da coluna de taxa livre de risco (opcional)
    window  : janela em dias
    ann     : fator de anualização
    """
    # Retornos em excesso
    if rf_col and rf_col in df.columns:
        excess = (pl.col(ret_col) - pl.col(rf_col)).alias("_excess")
    else:
        excess = pl.col(ret_col).alias("_excess")

    df = df.with_columns(excess)

    # Volatilidade rolling
    df = df.with_columns(
        (pl.col("_excess").rolling_std(window_size=window) * (ann ** 0.5))
        .alias(f"rolling_vol_{window}d")
    )

    # Retorno médio rolling (anualizado)
    df = df.with_columns(
        (pl.col("_excess").rolling_mean(window_size=window) * ann)
        .alias(f"rolling_ret_{window}d")
    )

    # Sharpe rolling
    df = df.with_columns(
        (pl.col(f"rolling_ret_{window}d") / pl.col(f"rolling_vol_{window}d").clip(lower_bound=1e-10))
        .alias(f"rolling_sharpe_{window}d")
    )

    # Downside returns para Sortino
    df = df.with_columns(
        pl.when(pl.col("_excess") < 0)
          .then(pl.col("_excess"))
          .otherwise(0.0)
          .alias("_downside")
    )

    df = df.with_columns(
        (
            pl.col(f"rolling_ret_{window}d") /
            (
                (pl.col("_downside").pow(2).rolling_mean(window_size=window) ** 0.5)
                * (ann ** 0.5)
            ).clip(lower_bound=1e-10)
        ).alias(f"rolling_sortino_{window}d")
    )

    return df.drop(["_excess", "_downside"])


# ---------------------------------------------------------------------------
# Tabela de métricas formatada em Polars
# ---------------------------------------------------------------------------

def format_metrics_table(
    metrics_dict: Dict[str, Dict[str, float]],
    pct_cols: Optional[List[str]] = None,
    round_cols: Optional[Dict[str, int]] = None,
) -> pl.DataFrame:
    """
    Formata dicionário de {estratégia: métricas} em Polars DataFrame.

    Parameters
    ----------
    metrics_dict : {nome_estratégia: {métrica: valor}}
    pct_cols     : colunas a multiplicar por 100 e formatar como %
    round_cols   : {col: n_decimais}

    Returns
    -------
    pl.DataFrame com coluna 'strategy' + métricas
    """
    if pct_cols is None:
        pct_cols = ["Return", "Volatility", "MDD"]
    if round_cols is None:
        round_cols = {"Sharpe": 2, "Sortino": 2, "Calmar": 2, "ADD": 1, "FAR": 3}

    rows = []
    for strategy, metrics in metrics_dict.items():
        row: Dict[str, Any] = {"strategy": strategy}
        row.update(metrics)
        rows.append(row)

    df = pl.DataFrame(rows)

    # Arredondar colunas especificadas
    for col, decimals in round_cols.items():
        if col in df.columns:
            df = df.with_columns(
                pl.col(col).round(decimals)
            )

    return df


# ---------------------------------------------------------------------------
# Decomposição de variância (ANOVA simples)
# ---------------------------------------------------------------------------

def variance_decomposition_polars(
    df:           pl.DataFrame,
    factor_cols:  List[str],
    metric_col:   str,
) -> pl.DataFrame:
    """
    Decomposição simples de variância (estilo ANOVA) para identificar
    quais fatores mais contribuem para a variância de uma métrica.

    Calcula SS_between para cada fator como proxy da contribuição.

    Parameters
    ----------
    df           : DataFrame com colunas de fatores e métricas
    factor_cols  : colunas de fatores (variáveis categóricas da ablação)
    metric_col   : coluna da métrica alvo

    Returns
    -------
    pl.DataFrame com colunas: factor, ss_between, pct_variance, rank
    """
    grand_mean = df.select(pl.col(metric_col).mean()).item()
    grand_var  = df.select(pl.col(metric_col).var()).item()

    rows = []
    for factor in factor_cols:
        # SS between = soma das variâncias dos grupos × n_grupo
        group_stats = (
            df.group_by(factor)
              .agg([
                  pl.col(metric_col).mean().alias("group_mean"),
                  pl.col(metric_col).count().alias("n"),
              ])
        )
        ss = group_stats.select(
            (pl.col("n") * (pl.col("group_mean") - grand_mean).pow(2)).sum()
        ).item()
        rows.append({"factor": factor, "ss_between": ss})

    total_ss = sum(r["ss_between"] for r in rows)

    result = pl.DataFrame(rows)
    result = result.with_columns(
        (pl.col("ss_between") / total_ss * 100).alias("pct_variance")
    )
    result = result.sort("pct_variance", descending=True)
    result = result.with_columns(
        pl.int_range(1, pl.len() + 1, eager=False).alias("rank")
    )

    return result
