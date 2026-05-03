"""
Technical indicator feature engineering for ML models.
All features are normalized and NaN-safe.
"""
from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:
    import ta
    HAS_TA = True
except ImportError:
    HAS_TA = False


def compute_features(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """
    Compute a rich set of technical indicators from OHLCV data.
    Returns a new DataFrame with all feature columns.
    """
    feat = pd.DataFrame(index=df.index)
    c = df["close"]
    h = df["high"]
    lo = df["low"]
    v = df["volume"]

    # --- Price action ---
    feat[f"{prefix}returns_1"] = c.pct_change(1)
    feat[f"{prefix}returns_3"] = c.pct_change(3)
    feat[f"{prefix}returns_6"] = c.pct_change(6)
    feat[f"{prefix}returns_12"] = c.pct_change(12)
    feat[f"{prefix}returns_24"] = c.pct_change(24)

    # Log returns
    feat[f"{prefix}log_ret_1"] = np.log(c / c.shift(1))
    feat[f"{prefix}log_ret_5"] = np.log(c / c.shift(5))

    # High-low range
    feat[f"{prefix}hl_range"] = (h - lo) / c
    feat[f"{prefix}body"] = (c - df["open"]) / (h - lo + 1e-10)

    # --- Moving averages ---
    for p in [7, 14, 21, 50]:
        feat[f"{prefix}sma_{p}"] = c.rolling(p).mean() / c - 1
        feat[f"{prefix}ema_{p}"] = c.ewm(span=p, adjust=False).mean() / c - 1

    # EMA crossover signals
    ema9 = c.ewm(span=9, adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    feat[f"{prefix}ema9_21_cross"] = (ema9 - ema21) / c
    feat[f"{prefix}ema21_50_cross"] = (ema21 - ema50) / c

    # --- Momentum ---
    feat[f"{prefix}roc_5"] = c.pct_change(5)
    feat[f"{prefix}roc_10"] = c.pct_change(10)
    feat[f"{prefix}roc_20"] = c.pct_change(20)

    # RSI (manual for no-dependency fallback)
    feat[f"{prefix}rsi_14"] = _rsi(c, 14)
    feat[f"{prefix}rsi_7"] = _rsi(c, 7)

    # Stochastic oscillator
    k, d = _stochastic(h, lo, c, 14, 3)
    feat[f"{prefix}stoch_k"] = k
    feat[f"{prefix}stoch_d"] = d
    feat[f"{prefix}stoch_cross"] = k - d

    # Williams %R
    feat[f"{prefix}williams_r"] = _williams_r(h, lo, c, 14)

    # --- MACD ---
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    feat[f"{prefix}macd"] = macd / c
    feat[f"{prefix}macd_signal"] = macd_signal / c
    feat[f"{prefix}macd_hist"] = (macd - macd_signal) / c
    feat[f"{prefix}macd_cross"] = np.sign(macd - macd_signal)

    # --- Bollinger Bands ---
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    feat[f"{prefix}bb_width"] = (bb_upper - bb_lower) / (bb_mid + 1e-10)
    feat[f"{prefix}bb_pos"] = (c - bb_lower) / (bb_upper - bb_lower + 1e-10)
    feat[f"{prefix}bb_squeeze"] = bb_std.rolling(20).mean() / (bb_std + 1e-10) - 1

    # --- Volatility (ATR) ---
    atr = _atr(h, lo, c, 14)
    feat[f"{prefix}atr_pct"] = atr / c
    feat[f"{prefix}atr_ratio"] = atr / atr.rolling(50).mean()
    # Normalized ATR for TP/SL computation (returned separately via get_atr)
    feat[f"{prefix}atr_raw"] = atr

    # --- Volume features ---
    vol_sma = v.rolling(20).mean()
    feat[f"{prefix}vol_ratio"] = v / (vol_sma + 1e-10)
    feat[f"{prefix}vol_trend"] = v.rolling(5).mean() / (v.rolling(20).mean() + 1e-10)
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    obv_sma = obv.rolling(20).mean()
    feat[f"{prefix}obv_signal"] = (obv - obv_sma) / (obv.rolling(20).std() + 1e-10)

    # On-balance volume slope
    feat[f"{prefix}obv_slope"] = obv.diff(5) / (obv.abs().rolling(5).mean() + 1e-10)

    # --- Market regime features ---
    feat[f"{prefix}adx"] = _adx(h, lo, c, 14)
    feat[f"{prefix}trend_strength"] = feat[f"{prefix}adx"] * feat[f"{prefix}ema9_21_cross"]

    # Price relative to recent range
    rolling_max = h.rolling(20).max()
    rolling_min = lo.rolling(20).min()
    feat[f"{prefix}price_position"] = (c - rolling_min) / (rolling_max - rolling_min + 1e-10)

    # CCI
    feat[f"{prefix}cci"] = _cci(h, lo, c, 20)

    # --- Time features (cyclical encoding) ---
    hours = df.index.hour
    feat[f"{prefix}hour_sin"] = np.sin(2 * np.pi * hours / 24)
    feat[f"{prefix}hour_cos"] = np.cos(2 * np.pi * hours / 24)
    dows = df.index.dayofweek
    feat[f"{prefix}dow_sin"] = np.sin(2 * np.pi * dows / 7)
    feat[f"{prefix}dow_cos"] = np.cos(2 * np.pi * dows / 7)

    feat = feat.replace([np.inf, -np.inf], np.nan)
    return feat


def get_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return _atr(df["high"], df["low"], df["close"], period)


def build_labels(df: pd.DataFrame, lookahead: int = 3, threshold: float = 0.005) -> pd.Series:
    """
    Binary label: 1 if close rises by >= threshold over next `lookahead` candles, else 0.
    Uses a threshold to avoid labeling tiny moves as signal.
    """
    future_ret = df["close"].shift(-lookahead) / df["close"] - 1
    labels = (future_ret >= threshold).astype(int)
    return labels


def align_multi_timeframe(
    base_df: pd.DataFrame,
    higher_tf_df: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """Forward-fill higher-timeframe features onto the base timeframe index."""
    higher_features = compute_features(higher_tf_df, prefix=prefix)
    reindexed = higher_features.reindex(base_df.index, method="ffill")
    return reindexed


# -----------------------------------------------------------------------
# Indicator implementations (no external dependency needed)
# -----------------------------------------------------------------------

def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, k_period: int, d_period: int
) -> Tuple[pd.Series, pd.Series]:
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    d = k.rolling(d_period).mean()
    return k, d


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return -100 * (hh - close) / (hh - ll + 1e-10)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = _atr(high, low, close, period)
    pos_di = 100 * pd.Series(pos_dm, index=close.index).ewm(span=period, adjust=False).mean() / (atr + 1e-10)
    neg_di = 100 * pd.Series(neg_dm, index=close.index).ewm(span=period, adjust=False).mean() / (atr + 1e-10)
    dx = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di + 1e-10)
    return dx.ewm(span=period, adjust=False).mean()


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tp = (high + low + close) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - sma) / (0.015 * mad + 1e-10)
