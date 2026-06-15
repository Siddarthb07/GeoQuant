from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yfinance as yf

from app.core.config import resolve_yfinance_symbol
from app.services.feature_engineering import INTRADAY_FEATURE_COLUMNS, add_technical_features


INTRADAY_PERIOD_BY_INTERVAL = {
    "1m": "7d",
    "2m": "60d",
    "5m": "60d",
    "15m": "60d",
    "30m": "60d",
    "60m": "730d",
    "1h": "730d",
}
MIN_SUCCESSFUL_SYMBOLS = 3
YFINANCE_TIMEOUT_SECONDS = 15
YFINANCE_MAX_RETRIES = 2


@dataclass
class ResearchDataset:
    market_data: pd.DataFrame
    features: pd.DataFrame
    symbol_to_code: Dict[str, int]
    warnings: List[str]


def _as_series(value: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    if isinstance(value, pd.DataFrame):
        if value.shape[1] == 0:
            return pd.Series(index=value.index, dtype="float64")
        value = value.iloc[:, 0]
    return pd.to_numeric(value, errors="coerce")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    mapping = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    if isinstance(df.columns, pd.MultiIndex):
        for src, dst in mapping.items():
            try:
                extracted = df.xs(src, axis=1, level=0, drop_level=False)
            except KeyError:
                continue
            out[dst] = _as_series(extracted)
    else:
        for src, dst in mapping.items():
            if src in df.columns:
                out[dst] = _as_series(df[src])
    if "close" not in out.columns:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    if "open" not in out.columns:
        out["open"] = out["close"]
    if "high" not in out.columns:
        out["high"] = out["close"]
    if "low" not in out.columns:
        out["low"] = out["close"]
    if "volume" not in out.columns:
        out["volume"] = 0.0
    out["volume"] = out["volume"].fillna(0.0)
    out = out.dropna(subset=["open", "high", "low", "close"]).sort_index()
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)
    out.index = idx
    return out


def _ensure_market_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    rename_map = {col: col.strip().lower() for col in out.columns}
    out = out.rename(columns=rename_map)

    if "timestamp" not in out.columns and "date" in out.columns:
        out = out.rename(columns={"date": "timestamp"})
    if "timestamp" not in out.columns:
        raise ValueError("Input market data must contain 'timestamp' column (or 'date').")
    if "symbol" not in out.columns:
        raise ValueError("Input market data must contain 'symbol' column.")
    if "close" not in out.columns:
        raise ValueError("Input market data must contain 'close' column.")

    for col in ["open", "high", "low"]:
        if col not in out.columns:
            out[col] = out["close"]
    if "volume" not in out.columns:
        out["volume"] = 0.0

    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["symbol"] = out["symbol"].astype(str).str.strip()
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["timestamp", "symbol", "close"])
    out = out[out["symbol"] != ""].copy()
    out = out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return out


def _fetch_single_symbol_intraday(
    symbol: str,
    yf_interval: str,
    period: str,
    requested_start_ts: pd.Timestamp,
    requested_end_ts: pd.Timestamp,
) -> Tuple[Optional[pd.DataFrame], List[str]]:
    warnings: List[str] = []
    fetch_symbol = resolve_yfinance_symbol(symbol)
    last_error: Optional[Exception] = None

    for attempt in range(YFINANCE_MAX_RETRIES):
        try:
            raw = yf.download(
                fetch_symbol,
                period=period,
                interval=yf_interval,
                progress=False,
                auto_adjust=False,
                threads=False,
                timeout=YFINANCE_TIMEOUT_SECONDS,
            )
            if raw.empty:
                last_error = RuntimeError("empty frame")
                sleep(1.0 * (attempt + 1))
                continue
            norm = _normalize_columns(raw)
            if norm.empty:
                warnings.append(f"{symbol}: bars returned but normalization produced empty frame.")
                return None, warnings
            norm["symbol"] = symbol
            norm["timestamp"] = pd.to_datetime(norm.index, utc=True, errors="coerce")
            coverage_start = norm["timestamp"].min()
            coverage_end = norm["timestamp"].max()
            if coverage_start is not None and coverage_start > requested_start_ts:
                warnings.append(
                    (
                        f"{symbol}: available intraday coverage starts at {coverage_start.date()} "
                        f"(requested from {requested_start_ts.date()}). "
                        "Free 5m feed has historical limits."
                    )
                )
            if coverage_end is not None and coverage_end < requested_end_ts:
                warnings.append(
                    (
                        f"{symbol}: available intraday coverage ends at {coverage_end.date()} "
                        f"(requested through {requested_end_ts.date()})."
                    )
                )
            frame = norm.reset_index(drop=True)[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]
            return frame, warnings
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            sleep(1.0 * (attempt + 1))

    if last_error is not None:
        warnings.append(f"{symbol}: intraday fetch failed ({last_error}).")
    else:
        warnings.append(f"{symbol}: no intraday bars returned for interval={yf_interval}.")
    return None, warnings


def _fetch_yfinance_intraday(
    symbols: List[str],
    interval: str,
    requested_start: str,
    requested_end: str,
) -> Tuple[pd.DataFrame, List[str]]:
    rows: List[pd.DataFrame] = []
    warnings: List[str] = []
    yf_interval = str(interval or "5m").lower()
    if yf_interval == "1h":
        yf_interval = "60m"
    period = INTRADAY_PERIOD_BY_INTERVAL.get(yf_interval, "60d")

    requested_start_ts = pd.Timestamp(requested_start, tz="UTC")
    requested_end_ts = pd.Timestamp(requested_end, tz="UTC")

    max_workers = min(6, max(1, len(symbols)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _fetch_single_symbol_intraday,
                symbol,
                yf_interval,
                period,
                requested_start_ts,
                requested_end_ts,
            ): symbol
            for symbol in symbols
        }
        for future in as_completed(futures):
            frame, symbol_warnings = future.result()
            warnings.extend(symbol_warnings)
            if frame is not None and not frame.empty:
                rows.append(frame)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"]), warnings

    successful_symbols = {frame["symbol"].iloc[0] for frame in rows if not frame.empty}
    if len(successful_symbols) < MIN_SUCCESSFUL_SYMBOLS:
        raise RuntimeError(
            f"Only {len(successful_symbols)} symbol(s) returned intraday data; need at least "
            f"{MIN_SUCCESSFUL_SYMBOLS}. Retry later, reduce the symbol list, or provide a CSV."
        )

    out = pd.concat(rows, ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return out, warnings


def load_market_data(
    csv_path: Optional[str],
    symbols: List[str],
    interval: str,
    start_date: str,
    end_date: str,
) -> Tuple[pd.DataFrame, List[str]]:
    if csv_path:
        path = Path(csv_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Configured csv_path does not exist: {path}")
        frame = pd.read_csv(path)
        data = _ensure_market_columns(frame)
        return data, []
    data, warnings = _fetch_yfinance_intraday(
        symbols=symbols,
        interval=interval,
        requested_start=start_date,
        requested_end=end_date,
    )
    if data.empty:
        raise RuntimeError(
            "Unable to load intraday market data from free feeds. Provide a CSV with "
            "columns: timestamp,symbol,open,high,low,close,volume."
        )
    return data, warnings


def build_intraday_feature_dataset(market_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    if market_data.empty:
        raise ValueError("Market data is empty; cannot build feature dataset.")

    symbols = sorted(market_data["symbol"].unique().tolist())
    symbol_to_code = {sym: idx for idx, sym in enumerate(symbols)}
    frames: List[pd.DataFrame] = []

    for symbol, group in market_data.groupby("symbol", sort=True):
        local = group.copy().sort_values("timestamp")
        local["timestamp"] = pd.to_datetime(local["timestamp"], utc=True, errors="coerce")
        local = local.dropna(subset=["timestamp"]).copy()
        if local.empty:
            continue
        idx = local["timestamp"].dt.tz_convert(None)
        local = local.set_index(idx)
        local = local[["open", "high", "low", "close", "volume"]]
        feats = add_technical_features(local)
        feats["symbol"] = symbol
        feats["symbol_code"] = symbol_to_code[symbol]
        feats["target_up"] = (feats["close"].shift(-1) > feats["close"]).astype(int)
        feats["timestamp"] = feats.index
        feats = feats.replace([np.inf, -np.inf], np.nan)
        feats = feats.dropna(subset=INTRADAY_FEATURE_COLUMNS + ["target_up", "close"])
        if feats.empty:
            continue
        frames.append(feats.reset_index(drop=True))

    if not frames:
        raise RuntimeError("No valid feature rows after technical feature engineering.")

    all_features = pd.concat(frames, ignore_index=True)
    all_features["timestamp"] = pd.to_datetime(all_features["timestamp"], utc=True, errors="coerce")
    all_features = all_features.dropna(subset=["timestamp", "symbol", "close"]).copy()
    all_features = all_features.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return all_features, symbol_to_code


def load_research_dataset(
    csv_path: Optional[str],
    symbols: List[str],
    interval: str,
    start_date: str,
    end_date: str,
) -> ResearchDataset:
    market_data, warnings = load_market_data(
        csv_path=csv_path,
        symbols=symbols,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
    )
    features, symbol_to_code = build_intraday_feature_dataset(market_data)
    return ResearchDataset(
        market_data=market_data,
        features=features,
        symbol_to_code=symbol_to_code,
        warnings=warnings,
    )
