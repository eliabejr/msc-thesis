"""
ablation/volatility_estimators.py
==================================
Estimadores de volatilidade com loops internos compilados via Numba JIT.

Cinco estimadores com diferentes requisitos de dados e eficiências estatísticas:

  1. Close-to-Close  (CC)  – baseline, usa apenas preço de fechamento
  2. Parkinson       (PK)  – High/Low, 5.2× mais eficiente que CC
  3. Garman-Klass    (GK)  – OHLC, 7.4× mais eficiente que CC
  4. Rogers-Satchell (RS)  – OHLC (drift-free), 8.0× mais eficiente que CC
  5. Yang-Zhang      (YZ)  – OHLC + overnight, 14× mais eficiente que CC

Referências:
  Parkinson (1980), Garman & Klass (1980), Rogers & Satchell (1991),
  Yang & Zhang (2000).

Uso:
  >>> import numpy as np
  >>> close = np.array([...])
  >>> vol = rolling_close_to_close(close, window=21)

  >>> df_pl = compute_all_estimators(open_, high, low, close, window=21)
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import numba as nb
import polars as pl

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

ESTIMATOR_NAMES = [
    "CloseToClose",
    "Parkinson",
    "GarmanKlass",
    "RogersSatchell",
    "YangZhang",
]

_LN2   = np.log(2.0)
_2LN2M1 = 2.0 * _LN2 - 1.0   # ≈ 0.3863

# ---------------------------------------------------------------------------
# JIT – Close-to-Close
# ---------------------------------------------------------------------------

@nb.njit(cache=True, fastmath=True)
def _cc_inner(log_ret: np.ndarray, window: int, out: np.ndarray) -> None:
    """
    Variância amostral rolling dos log-retornos (close-to-close).
    Complexidade O(T * window) com acumuladores incrementais.
    """
    T = len(log_ret)
    for t in range(window - 1, T):
        mu = 0.0
        for i in range(t - window + 1, t + 1):
            mu += log_ret[i]
        mu /= window
        var = 0.0
        for i in range(t - window + 1, t + 1):
            d = log_ret[i] - mu
            var += d * d
        out[t] = var / (window - 1)


def rolling_close_to_close(close: np.ndarray, window: int = 21) -> np.ndarray:
    """
    Estimador Close-to-Close: σ² = Var(ln C_t / C_{t-1}).

    Parameters
    ----------
    close  : array de preços de fechamento (ajustados)
    window : janela rolling em dias

    Returns
    -------
    Array de variâncias diárias (annualise × 252 para volatilidade anual).
    Primeiros `window` elementos são NaN.
    """
    log_ret = np.log(close[1:] / close[:-1])
    log_ret = np.concatenate(([np.nan], log_ret))  # alinha com close
    out = np.full(len(close), np.nan)
    _cc_inner(log_ret[1:], window, out[1:])  # ignora primeiro NaN
    return out


# ---------------------------------------------------------------------------
# JIT – Parkinson
# ---------------------------------------------------------------------------

@nb.njit(cache=True, fastmath=True)
def _pk_inner(
    log_hl: np.ndarray, window: int, out: np.ndarray
) -> None:
    """
    Parkinson: σ² = mean[ (ln H/L)² / (4 ln 2) ] sobre janela.
    """
    c = 1.0 / (4.0 * 0.6931471805599453)  # 1 / (4 * ln(2))
    T = len(log_hl)
    for t in range(window - 1, T):
        s = 0.0
        for i in range(t - window + 1, t + 1):
            s += log_hl[i] * log_hl[i]
        out[t] = c * s / window


def rolling_parkinson(
    high: np.ndarray, low: np.ndarray, window: int = 21
) -> np.ndarray:
    """
    Estimador Parkinson (1980): usa apenas High/Low.

    σ² = (1 / 4 ln 2) * E[(ln H/L)²]

    5.2× mais eficiente que Close-to-Close.
    """
    log_hl = np.log(high / low)
    out = np.full(len(high), np.nan)
    _pk_inner(log_hl, window, out)
    return out


# ---------------------------------------------------------------------------
# JIT – Garman-Klass
# ---------------------------------------------------------------------------

@nb.njit(cache=True, fastmath=True)
def _gk_inner(
    log_hl: np.ndarray,
    log_co: np.ndarray,
    window: int,
    out:    np.ndarray,
) -> None:
    """
    Garman-Klass: σ² = 0.5*(h-l)² - (2 ln 2 - 1)*c²
    onde h = ln(H/C_prev), l = ln(L/C_prev), c = ln(C/C_prev).
    """
    c_factor = 2.0 * 0.6931471805599453 - 1.0  # 2 ln2 - 1
    T = len(log_hl)
    for t in range(window - 1, T):
        s = 0.0
        for i in range(t - window + 1, t + 1):
            s += 0.5 * log_hl[i] * log_hl[i] - c_factor * log_co[i] * log_co[i]
        out[t] = s / window


def rolling_garman_klass(
    open_: np.ndarray,
    high:  np.ndarray,
    low:   np.ndarray,
    close: np.ndarray,
    window: int = 21,
) -> np.ndarray:
    """
    Estimador Garman-Klass (1980): usa OHLC.

    σ² = 0.5*(ln H/L)² - (2 ln 2 - 1)*(ln C/O)²

    7.4× mais eficiente que Close-to-Close.
    """
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    out = np.full(len(close), np.nan)
    _gk_inner(log_hl, log_co, window, out)
    return out


# ---------------------------------------------------------------------------
# JIT – Rogers-Satchell
# ---------------------------------------------------------------------------

@nb.njit(cache=True, fastmath=True)
def _rs_inner(
    lh: np.ndarray,
    ll: np.ndarray,
    lc: np.ndarray,
    window: int,
    out:    np.ndarray,
) -> None:
    """
    Rogers-Satchell: σ² = h*(h-c) + l*(l-c)
    onde h = ln(H/O), l = ln(L/O), c = ln(C/O).
    """
    T = len(lh)
    for t in range(window - 1, T):
        s = 0.0
        for i in range(t - window + 1, t + 1):
            s += lh[i] * (lh[i] - lc[i]) + ll[i] * (ll[i] - lc[i])
        out[t] = s / window


def rolling_rogers_satchell(
    open_: np.ndarray,
    high:  np.ndarray,
    low:   np.ndarray,
    close: np.ndarray,
    window: int = 21,
) -> np.ndarray:
    """
    Estimador Rogers-Satchell (1991): drift-free, usa OHLC.

    σ² = E[h(h-c) + l(l-c)]  onde h=ln(H/O), l=ln(L/O), c=ln(C/O)

    8.0× mais eficiente que Close-to-Close.
    """
    lh = np.log(high  / open_)
    ll = np.log(low   / open_)
    lc = np.log(close / open_)
    out = np.full(len(close), np.nan)
    _rs_inner(lh, ll, lc, window, out)
    return out


# ---------------------------------------------------------------------------
# JIT – Yang-Zhang
# ---------------------------------------------------------------------------

@nb.njit(cache=True, fastmath=True)
def _yz_overnight_var(log_oc: np.ndarray, window: int) -> float:
    """Variância overnight: Var(ln O_t / C_{t-1})."""
    n = 0
    mu = 0.0
    for i in range(len(log_oc) - window, len(log_oc)):
        mu += log_oc[i]
        n += 1
    if n < 2:
        return 0.0
    mu /= n
    var = 0.0
    for i in range(len(log_oc) - window, len(log_oc)):
        d = log_oc[i] - mu
        var += d * d
    return var / (n - 1)


@nb.njit(cache=True, fastmath=True)
def _yz_open_close_var(log_oc_norm: np.ndarray, window: int) -> float:
    """Variância open-to-close normalizada."""
    n = 0
    mu = 0.0
    for i in range(len(log_oc_norm) - window, len(log_oc_norm)):
        mu += log_oc_norm[i]
        n += 1
    if n < 2:
        return 0.0
    mu /= n
    var = 0.0
    for i in range(len(log_oc_norm) - window, len(log_oc_norm)):
        d = log_oc_norm[i] - mu
        var += d * d
    return var / (n - 1)


@nb.njit(cache=True, fastmath=True)
def _yz_rs_scalar(
    lh: np.ndarray, ll: np.ndarray, lc: np.ndarray, window: int
) -> float:
    """Rogers-Satchell para uma janela escalar (usado internamente no YZ)."""
    s = 0.0
    for i in range(len(lh) - window, len(lh)):
        s += lh[i] * (lh[i] - lc[i]) + ll[i] * (ll[i] - lc[i])
    return s / window


@nb.njit(cache=True, fastmath=True)
def _yz_inner(
    log_oc:      np.ndarray,  # ln(O_t / C_{t-1})
    log_co_norm: np.ndarray,  # ln(C_t / O_t)
    lh:          np.ndarray,  # ln(H/O)
    ll:          np.ndarray,  # ln(L/O)
    lc:          np.ndarray,  # ln(C/O)
    window:      int,
    k:           float,
    out:         np.ndarray,
) -> None:
    """
    Yang-Zhang: σ² = σ_o² + k*σ_oc² + (1-k)*σ_RS²
    k = 0.34 / (1.34 + (window+1)/(window-1))  [factor de eficiência ótimo]
    """
    T = len(log_oc)
    for t in range(window - 1, T):
        s  = t - window + 1
        e  = t + 1

        # σ_o² (overnight)
        mu_o = 0.0
        for i in range(s, e):
            mu_o += log_oc[i]
        mu_o /= window
        var_o = 0.0
        for i in range(s, e):
            d = log_oc[i] - mu_o
            var_o += d * d
        var_o /= (window - 1)

        # σ_oc² (open-to-close)
        mu_oc = 0.0
        for i in range(s, e):
            mu_oc += log_co_norm[i]
        mu_oc /= window
        var_oc = 0.0
        for i in range(s, e):
            d = log_co_norm[i] - mu_oc
            var_oc += d * d
        var_oc /= (window - 1)

        # σ_RS² (Rogers-Satchell)
        var_rs = 0.0
        for i in range(s, e):
            var_rs += lh[i] * (lh[i] - lc[i]) + ll[i] * (ll[i] - lc[i])
        var_rs /= window

        out[t] = var_o + k * var_oc + (1.0 - k) * var_rs


def rolling_yang_zhang(
    open_: np.ndarray,
    high:  np.ndarray,
    low:   np.ndarray,
    close: np.ndarray,
    window: int = 21,
) -> np.ndarray:
    """
    Estimador Yang-Zhang (2000): drift-independent, usa OHLC + overnight.

    σ² = σ_overnight² + k * σ_open-to-close² + (1-k) * σ_Rogers-Satchell²

    14× mais eficiente que Close-to-Close.

    Parameters
    ----------
    open_, high, low, close : arrays de preços OHLC
    window                  : janela rolling em dias

    Returns
    -------
    Array de variâncias diárias.
    """
    # k ótimo de Yang & Zhang (2000, eq. 20)
    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    # Componentes
    # Nota: log_oc usa close do dia anterior → shift
    log_oc      = np.log(open_[1:] / close[:-1])       # len T-1
    log_oc      = np.concatenate(([np.nan], log_oc))    # alinhar com T

    log_co_norm = np.log(close / open_)                 # ln(C/O)
    lh          = np.log(high  / open_)                 # ln(H/O)
    ll          = np.log(low   / open_)                 # ln(L/O)
    lc          = np.log(close / open_)                 # ln(C/O)

    out = np.full(len(close), np.nan)
    # Começa a partir de índice 1 para ter log_oc válido
    _yz_inner(log_oc[1:], log_co_norm[1:], lh[1:], ll[1:], lc[1:], window, k, out[1:])
    return out


# ---------------------------------------------------------------------------
# Função de conveniência: todos os estimadores → Polars DataFrame
# ---------------------------------------------------------------------------

def compute_all_estimators(
    open_:  np.ndarray,
    high:   np.ndarray,
    low:    np.ndarray,
    close:  np.ndarray,
    dates:  Optional[np.ndarray] = None,
    window: int = 21,
    annualize: bool = True,
) -> pl.DataFrame:
    """
    Calcula todos os estimadores de volatilidade e retorna um Polars DataFrame.

    Parameters
    ----------
    open_, high, low, close : arrays de preços OHLC
    dates                   : array de datas (opcional)
    window                  : janela rolling
    annualize               : se True, multiplica variâncias por 252

    Returns
    -------
    pl.DataFrame com colunas: date (opcional), CloseToClose, Parkinson,
    GarmanKlass, RogersSatchell, YangZhang.
    Valores são volatilidades anualizadas (√(var × 252)) se annualize=True.
    """
    factor = 252.0 if annualize else 1.0

    cc = rolling_close_to_close(close,              window)
    pk = rolling_parkinson     (high, low,          window)
    gk = rolling_garman_klass  (open_, high, low, close, window)
    rs = rolling_rogers_satchell(open_, high, low, close, window)
    yz = rolling_yang_zhang    (open_, high, low, close, window)

    # Converter variância → volatilidade
    def _to_vol(v: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.sqrt(np.where(v > 0, v * factor, np.nan))

    data: dict = {
        "CloseToClose":   _to_vol(cc).tolist(),
        "Parkinson":      _to_vol(pk).tolist(),
        "GarmanKlass":    _to_vol(gk).tolist(),
        "RogersSatchell": _to_vol(rs).tolist(),
        "YangZhang":      _to_vol(yz).tolist(),
    }

    if dates is not None:
        data = {"date": list(dates), **data}

    return pl.DataFrame(data)


# ---------------------------------------------------------------------------
# Download de dados OHLC via yfinance (helper para notebooks)
# ---------------------------------------------------------------------------

def download_ohlc(
    ticker: str,
    start:  str,
    end:    str,
) -> pl.DataFrame:
    """
    Baixa dados OHLC de um ticker via yfinance e retorna Polars DataFrame.

    Returns
    -------
    pl.DataFrame com colunas: date, open, high, low, close, adj_close, volume
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance é necessário. Instale com: pip install yfinance") from exc

    raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if raw.empty:
        raise ValueError(f"Nenhum dado encontrado para ticker='{ticker}'.")

    # Flatten MultiIndex columns se necessário
    if isinstance(raw.columns, type(raw.columns)) and hasattr(raw.columns, "get_level_values"):
        raw.columns = [
            c[0].lower().replace(" ", "_") if isinstance(c, tuple) else c.lower()
            for c in raw.columns
        ]
    else:
        raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]

    raw = raw.reset_index()
    raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]

    # Padronizar nomes
    col_map = {
        "adj_close": "adj_close",
        "adj close": "adj_close",
    }
    raw = raw.rename(columns=col_map)

    return pl.from_pandas(raw[["date", "open", "high", "low", "close"]])
