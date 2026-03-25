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

    frame["vol_5"] = frame["ret_1"].rolling(5).std()
    frame["vol_20"] = frame["ret_1"].rolling(20).std()

    frame["sma_10"] = frame["close"].rolling(10).mean()
    frame["sma_20"] = frame["close"].rolling(20).mean()
    frame["sma_50"] = frame["close"].rolling(50).mean()
    frame["sma_ratio_10_20"] = frame["sma_10"] / frame["sma_20"]
    frame["sma_ratio_20_50"] = frame["sma_20"] / frame["sma_50"]

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
