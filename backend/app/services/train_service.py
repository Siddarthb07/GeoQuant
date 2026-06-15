from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from app.core.config import settings
from app.data.market_data import download_daily_history, download_intraday_history
from app.data.news_data import daily_news_sentiment_series, fetch_global_news
from app.services.feature_engineering import (
    FEATURE_COLUMNS,
    INTRADAY_FEATURE_COLUMNS,
    make_daily_dataset,
    make_intraday_dataset,
)
from app.services.neural_model import train_binary_classifier


STATUS_LOCK = threading.Lock()
TRAIN_LOCK = threading.Lock()
TRAIN_THREAD: Optional[threading.Thread] = None

TRAIN_STATUS: Dict = {
    "state": "idle",
    "message": "No training run yet.",
    "updated_at": datetime.now(timezone.utc),
    "metrics": {},
}


def _set_status(state: str, message: str, metrics: Optional[Dict] = None) -> None:
    with STATUS_LOCK:
        TRAIN_STATUS["state"] = state
        TRAIN_STATUS["message"] = message
        TRAIN_STATUS["updated_at"] = datetime.now(timezone.utc)
        if metrics is not None:
            TRAIN_STATUS["metrics"] = metrics


def get_status() -> Dict:
    with STATUS_LOCK:
        return {
            "state": TRAIN_STATUS["state"],
            "message": TRAIN_STATUS["message"],
            "updated_at": TRAIN_STATUS["updated_at"],
            "metrics": TRAIN_STATUS["metrics"],
        }


def model_paths(prefix: str) -> dict:
    model_dir = settings.model_dir
    return {
        "model": model_dir / f"{prefix}_model.pkl",
        "scaler": model_dir / f"{prefix}_scaler.pkl",
        "meta": model_dir / f"{prefix}_meta.json",
    }


def models_exist() -> bool:
    daily = model_paths("daily")
    intra = model_paths("intraday")
    return all(path.exists() for path in daily.values()) and all(path.exists() for path in intra.values())


def _time_split(df: pd.DataFrame, cutoff: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff_ts = pd.Timestamp(cutoff)
    left = df[df.index < cutoff_ts]
    right = df[df.index >= cutoff_ts]
    if len(right) < 100:
        n = max(100, int(len(df) * 0.2))
        left = df.iloc[:-n]
        right = df.iloc[-n:]
    return left, right


def _prepare_xy(df: pd.DataFrame, feature_columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_columns + ["target_up"])
    if clean.empty:
        raise RuntimeError("Feature set is empty after cleaning. Try adjusting symbols or date ranges.")
    x = clean[feature_columns].astype(float).to_numpy()
    y = clean["target_up"].astype(int).to_numpy()
    return x, y


def _save_bundle(
    prefix: str,
    model: object,
    scaler: StandardScaler,
    feature_columns: list[str],
    symbol_to_code: dict[str, int],
    metrics: Dict,
) -> None:
    paths = model_paths(prefix)
    joblib.dump(model, paths["model"])
    joblib.dump(scaler, paths["scaler"])
    with paths["meta"].open("w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_columns": feature_columns,
                "symbol_to_code": symbol_to_code,
                "input_dim": len(feature_columns),
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "metrics": metrics,
            },
            f,
            indent=2,
        )


def _collect_training_frames(symbol_to_code: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        news_items = fetch_global_news(limit=500)
    except Exception:  # noqa: BLE001
        news_items = []
    news_daily = daily_news_sentiment_series(news_items)

    daily_frames = []
    intraday_frames = []
    failures = 0
    symbols = list(symbol_to_code.keys())

    for idx, symbol in enumerate(symbols, start=1):
        _set_status("running", f"Downloading and engineering features for {symbol} ({idx}/{len(symbols)})")
        code = symbol_to_code[symbol]
        try:
            daily = download_daily_history(symbol, settings.data_start_date, settings.data_end_date)
            if len(daily) >= 300:
                daily_feats = make_daily_dataset(daily, news_daily, code)
                if not daily_feats.empty:
                    daily_feats["symbol"] = symbol
                    daily_frames.append(daily_feats)

            intraday = download_intraday_history(symbol, period=settings.intraday_period, interval=settings.intraday_interval)
            if len(intraday) >= 200:
                intraday_feats = make_intraday_dataset(intraday, code)
                if not intraday_feats.empty:
                    intraday_feats["symbol"] = symbol
                    intraday_feats["news_sentiment"] = 0.0
                    intraday_frames.append(intraday_feats)
        except Exception:  # noqa: BLE001
            failures += 1
            continue

    if not daily_frames:
        raise RuntimeError(
            "No daily data was downloaded. Check internet connectivity or ticker symbols."
        )
    daily_df = pd.concat(daily_frames).sort_index()
    intraday_df = pd.concat(intraday_frames).sort_index() if intraday_frames else pd.DataFrame()
    if failures > 0:
        _set_status(
            "running",
            f"Data build continuing with partial coverage ({failures} symbols skipped due feed errors).",
        )
    return daily_df, intraday_df


def _run_training() -> Dict:
    symbols = settings.default_us_universe + settings.default_india_universe
    symbol_to_code = {sym: i for i, sym in enumerate(symbols)}

    daily_df, intraday_df = _collect_training_frames(symbol_to_code)

    daily_train, daily_val = _time_split(daily_df, "2023-01-01")
    x_train, y_train = _prepare_xy(daily_train, FEATURE_COLUMNS)
    x_val, y_val = _prepare_xy(daily_val, FEATURE_COLUMNS)
    daily_scaler = StandardScaler()
    x_train_scaled = daily_scaler.fit_transform(x_train)
    x_val_scaled = daily_scaler.transform(x_val)
    _set_status("running", "Training daily neural network model")
    daily_result = train_binary_classifier(x_train_scaled, y_train, x_val_scaled, y_val, epochs=40)

    daily_metrics = {
        "accuracy": round(daily_result.accuracy, 4),
        "auc": round(daily_result.auc, 4),
        "train_rows": int(len(x_train)),
        "val_rows": int(len(x_val)),
    }
    _save_bundle("daily", daily_result.model, daily_scaler, FEATURE_COLUMNS, symbol_to_code, daily_metrics)

    intraday_metrics: Dict = {}
    # Some free sources may not provide enough historical intraday bars for all symbols.
    # In that case, fall back to daily-derived features so the intraday head still exists.
    base_intraday_df = intraday_df if not intraday_df.empty else daily_df.copy()
    intra_train, intra_val = _time_split(base_intraday_df, "2025-01-01")
    xi_train, yi_train = _prepare_xy(intra_train, INTRADAY_FEATURE_COLUMNS)
    xi_val, yi_val = _prepare_xy(intra_val, INTRADAY_FEATURE_COLUMNS)
    intra_scaler = StandardScaler()
    xi_train_scaled = intra_scaler.fit_transform(xi_train)
    xi_val_scaled = intra_scaler.transform(xi_val)
    _set_status("running", "Training intraday neural network model")
    intra_result = train_binary_classifier(xi_train_scaled, yi_train, xi_val_scaled, yi_val, epochs=35)
    intraday_metrics = {
        "accuracy": round(intra_result.accuracy, 4),
        "auc": round(intra_result.auc, 4),
        "train_rows": int(len(xi_train)),
        "val_rows": int(len(xi_val)),
    }
    _save_bundle(
        "intraday",
        intra_result.model,
        intra_scaler,
        INTRADAY_FEATURE_COLUMNS,
        symbol_to_code,
        intraday_metrics,
    )

    return {"daily": daily_metrics, "intraday": intraday_metrics}


def _training_worker() -> None:
    with TRAIN_LOCK:
        try:
            _set_status("running", "Starting dataset build and model training")
            metrics = _run_training()
            _set_status("completed", "Training completed successfully", metrics=metrics)
        except Exception as exc:  # noqa: BLE001
            _set_status("failed", f"Training failed: {exc}")


def launch_training(force: bool = False) -> bool:
    global TRAIN_THREAD
    if TRAIN_THREAD is not None and TRAIN_THREAD.is_alive():
        return False
    if models_exist() and not force:
        _set_status("completed", "Models already available. Use force=true to retrain.", metrics=get_status()["metrics"])
        return False
    TRAIN_THREAD = threading.Thread(target=_training_worker, daemon=True)
    TRAIN_THREAD.start()
    return True
