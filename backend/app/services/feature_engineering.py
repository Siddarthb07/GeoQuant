from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1 / period, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["ret_1"] = frame["close"].pct_change(1)
    frame["ret_3"] = frame["close"].pct_change(3)
    frame["ret_5"] = frame["close"].pct_change(5)
    frame["ret_10"] = frame["close"].pct_change(10)
    frame["fwd_ret_1"] = frame["close"].pct_change(1).shift(-1)

    frame["vol_5"] = frame["ret_1"].rolling(5).std()
    frame["vol_20"] = frame["ret_1"].rolling(20).std()

    frame["mom_21"] = frame["close"].pct_change(21)
    frame["mom_63"] = frame["close"].pct_change(63)
    frame["mom_126"] = frame["close"].pct_change(126)
    # Classic 12-1 month momentum: 252d return excluding the most recent month.
    frame["mom_252_skip21"] = frame["close"].shift(21) / frame["close"].shift(252) - 1.0
    frame["risk_adj_mom"] = frame["mom_126"] / frame["vol_20"].replace(0, np.nan)

    frame["sma_10"] = frame["close"].rolling(10).mean()
    frame["sma_20"] = frame["close"].rolling(20).mean()
    frame["sma_50"] = frame["close"].rolling(50).mean()
    frame["sma_200"] = frame["close"].rolling(200).mean()
    frame["sma_ratio_10_20"] = frame["sma_10"] / frame["sma_20"]
    frame["sma_ratio_20_50"] = frame["sma_20"] / frame["sma_50"]
    frame["trend_ok"] = (frame["close"] > frame["sma_50"]).astype(float)
    frame["trend_long"] = (
        (frame["close"] > frame["sma_50"]) & (frame["sma_50"] > frame["sma_200"])
    ).astype(float)
    # Donchian breakout flags (prior window excludes current bar).
    prior_high_20 = frame["high"].shift(1).rolling(20).max()
    prior_low_10 = frame["low"].shift(1).rolling(10).min()
    prior_high_55 = frame["high"].shift(1).rolling(55).max()
    prior_low_20 = frame["low"].shift(1).rolling(20).min()
    frame["breakout_20"] = (frame["close"] > prior_high_20).astype(float)
    frame["breakdown_10"] = (frame["close"] < prior_low_10).astype(float)
    frame["breakout_55"] = (frame["close"] > prior_high_55).astype(float)
    frame["breakdown_20"] = (frame["close"] < prior_low_20).astype(float)

    frame["rsi_14"] = _rsi(frame["close"], period=14)
    macd, signal = _macd(frame["close"])
    frame["macd"] = macd
    frame["macd_signal"] = signal
    frame["macd_hist"] = macd - signal

    tr1 = (frame["high"] - frame["low"]).abs()
    tr2 = (frame["high"] - frame["close"].shift(1)).abs()
    tr3 = (frame["low"] - frame["close"].shift(1)).abs()
    frame["atr_14"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
    frame["atr_pct"] = frame["atr_14"] / frame["close"] * 100

    vol_roll = frame["volume"].rolling(20)
    frame["volume_z"] = (frame["volume"] - vol_roll.mean()) / vol_roll.std()

    return frame


def make_economic_target(fwd_ret: pd.Series, edge_threshold: float) -> pd.Series:
    """Label 1 only when next-bar return clears a cost/edge hurdle."""
    threshold = float(max(0.0, edge_threshold))
    return (pd.to_numeric(fwd_ret, errors="coerce") > threshold).astype(int)


def make_daily_dataset(
    df: pd.DataFrame,
    news_sentiment: pd.Series,
    symbol_code: int,
) -> pd.DataFrame:
    feats = add_technical_features(df)
    feats["date"] = feats.index.date
    feats["news_sentiment"] = feats["date"].map(news_sentiment).ffill().fillna(0.0)
    feats["symbol_code"] = symbol_code
    feats["target_up"] = (feats["close"].shift(-1) > feats["close"]).astype(int)
    feats = feats.dropna()
    return feats


def make_intraday_dataset(df: pd.DataFrame, symbol_code: int) -> pd.DataFrame:
    feats = add_technical_features(df)
    feats["symbol_code"] = symbol_code
    feats["target_up"] = (feats["close"].shift(-1) > feats["close"]).astype(int)
    feats = feats.dropna()
    return feats


FEATURE_COLUMNS = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "vol_5",
    "vol_20",
    "sma_ratio_10_20",
    "sma_ratio_20_50",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "atr_pct",
    "volume_z",
    "news_sentiment",
    "symbol_code",
]

# Daily research path: no live news feed required; trend flags are filters, not model inputs.
DAILY_FEATURE_COLUMNS = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "vol_5",
    "vol_20",
    "sma_ratio_10_20",
    "sma_ratio_20_50",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "atr_pct",
    "volume_z",
    "symbol_code",
]

INTRADAY_FEATURE_COLUMNS = [
    "ret_1",
    "ret_3",
    "ret_5",
    "vol_5",
    "vol_20",
    "sma_ratio_10_20",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "atr_pct",
    "volume_z",
    "symbol_code",
]
