from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Union

import pandas as pd
import yfinance as yf


@dataclass
class MarketFrame:
    symbol: str
    frame: pd.DataFrame


def _as_series(value: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    if isinstance(value, pd.DataFrame):
        if value.shape[1] == 0:
            return pd.Series(index=value.index, dtype="float64")
        value = value.iloc[:, 0]
    return pd.to_numeric(value, errors="coerce")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    out = pd.DataFrame(index=df.index)

    if isinstance(df.columns, pd.MultiIndex):
        for source, target in mapping.items():
            try:
                extracted = df.xs(source, axis=1, level=0, drop_level=False)
            except KeyError:
                continue
            out[target] = _as_series(extracted)
    else:
        for source, target in mapping.items():
            if source in df.columns:
                out[target] = _as_series(df[source])

    required = ["open", "high", "low", "close"]
    if any(col not in out.columns for col in required):
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    if "volume" not in out.columns:
        out["volume"] = 0.0

    out["volume"] = out["volume"].fillna(0.0)
    out = out.dropna(subset=required).sort_index()
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)
    out.index = idx
    return out


def download_daily_history(symbol: str, start: str, end: str) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    try:
        raw = yf.download(
            symbol,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
            threads=False,
            timeout=4,
        )
        if not raw.empty:
            normalized = _normalize_columns(raw)
            if not normalized.empty:
                return normalized
    except Exception:  # noqa: BLE001
        pass

    try:
        hist = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=False, timeout=4)
        if hist.empty:
            return empty
        return _normalize_columns(hist)
    except Exception:  # noqa: BLE001
        return empty


def download_intraday_history(symbol: str, period: str = "60d", interval: str = "30m") -> pd.DataFrame:
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    try:
        raw = yf.download(
            symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=False,
            timeout=4,
        )
        if not raw.empty:
            normalized = _normalize_columns(raw)
            if not normalized.empty:
                return normalized
    except Exception:  # noqa: BLE001
        pass

    try:
        hist = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False, timeout=4)
        if hist.empty:
            return empty
        return _normalize_columns(hist)
    except Exception:  # noqa: BLE001
        return empty


def latest_price(symbol: str) -> Optional[float]:
    ticker = yf.Ticker(symbol)
    info = ticker.fast_info or {}
    last = info.get("lastPrice")
    if last is not None:
        return float(last)
    hist = ticker.history(period="5d")
    if hist.empty:
        return None
    close = hist["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    if close.empty:
        return None
    return float(close.iloc[-1])


def market_from_symbol(symbol: str) -> str:
    return "INDIA" if symbol.endswith(".NS") else "US"


def bulk_daily(symbols: Iterable[str], start: str, end: str) -> list[MarketFrame]:
    frames: list[MarketFrame] = []
    for symbol in symbols:
        df = download_daily_history(symbol, start=start, end=end)
        if df.empty:
            continue
        frames.append(MarketFrame(symbol=symbol, frame=df))
    return frames
