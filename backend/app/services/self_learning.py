from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from app.core.config import settings
from app.core.database import (
    list_unresolved_predictions,
    mark_prediction_resolved,
    prediction_accuracy_stats,
)
from app.data.market_data import download_daily_history
from app.services.train_service import get_status, launch_training


STATE_LOCK = threading.Lock()
THREAD: Optional[threading.Thread] = None
STOP_EVENT = threading.Event()

SELF_STATE: Dict = {
    "last_cycle_at": None,
    "last_retrain_at": None,
    "resolved_signals": 0,
    "rolling_hit_rate": 0.5,
    "sample_size": 0,
    "state": "idle",
    "message": "Self-learning not started.",
}


def _set_state(**kwargs) -> None:
    with STATE_LOCK:
        SELF_STATE.update(kwargs)


def _resolve_prediction_feedback() -> int:
    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    rows = list_unresolved_predictions(cutoff_date=cutoff, limit=400)
    resolved = 0
    for row in rows:
        try:
            prediction_date = datetime.fromisoformat(row["prediction_date"]).date()
            start = (prediction_date - timedelta(days=2)).isoformat()
            end = (prediction_date + timedelta(days=10)).isoformat()
            df = download_daily_history(row["symbol"], start=start, end=end)
            if df.empty:
                continue
            df = df.sort_index()
            if "close" not in df.columns:
                continue

            next_closes = df[df.index.date > prediction_date]["close"]
            if next_closes.empty:
                continue
            next_close = float(next_closes.iloc[0])
            reference_price = float(row["reference_price"])
            actual_up = 1 if next_close > reference_price else 0
            predicted_up = 1 if str(row["direction"]).upper() == "LONG" else 0
            was_correct = 1 if actual_up == predicted_up else 0
            mark_prediction_resolved(int(row["id"]), actual_up, was_correct)
            resolved += 1
        except Exception:  # noqa: BLE001
            continue
    return resolved


def _refresh_accuracy_cache() -> None:
    stats = prediction_accuracy_stats(window=350)
    _set_state(
        rolling_hit_rate=round(float(stats["hit_rate"]), 4),
        sample_size=int(stats["sample_size"]),
    )


def _maybe_retrain() -> None:
    with STATE_LOCK:
        last_retrain = SELF_STATE.get("last_retrain_at")
    last_dt = None
    if isinstance(last_retrain, str):
        try:
            last_dt = datetime.fromisoformat(last_retrain)
        except ValueError:
            last_dt = None
    if last_dt is None:
        last_dt = datetime.now(timezone.utc) - timedelta(days=365)

    age_hours = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600.0
    if age_hours < settings.self_learning_retrain_hours:
        return

    train_status = get_status()
    if train_status["state"] == "running":
        return

    started = launch_training(force=True)
    if started:
        _set_state(last_retrain_at=datetime.now(timezone.utc).isoformat(), message="Auto retrain started.")


def _loop() -> None:
    _set_state(state="running", message="Self-learning active.")
    while not STOP_EVENT.is_set():
        cycle_start = datetime.now(timezone.utc)
        try:
            resolved = _resolve_prediction_feedback()
            _refresh_accuracy_cache()
            _maybe_retrain()
            with STATE_LOCK:
                total = int(SELF_STATE.get("resolved_signals", 0)) + int(resolved)
            _set_state(
                last_cycle_at=cycle_start.isoformat(),
                resolved_signals=total,
                state="running",
                message=f"Cycle complete. Resolved {resolved} new prediction outcomes.",
            )
        except Exception as exc:  # noqa: BLE001
            _set_state(
                last_cycle_at=cycle_start.isoformat(),
                state="running",
                message=f"Cycle warning: {exc}",
            )
        STOP_EVENT.wait(timeout=max(60, settings.self_learning_refresh_minutes * 60))


def start_self_learning_loop() -> None:
    global THREAD
    if not settings.enable_self_learning:
        _set_state(state="disabled", message="Self-learning disabled in settings.")
        return
    if THREAD is not None and THREAD.is_alive():
        return
    with STATE_LOCK:
        if SELF_STATE.get("last_retrain_at") is None:
            SELF_STATE["last_retrain_at"] = datetime.now(timezone.utc).isoformat()
    STOP_EVENT.clear()
    THREAD = threading.Thread(target=_loop, daemon=True, name="self-learning-loop")
    THREAD.start()


def get_self_learning_status() -> Dict:
    _refresh_accuracy_cache()
    with STATE_LOCK:
        payload = dict(SELF_STATE)
    payload["enabled"] = bool(settings.enable_self_learning)
    payload["cycle_minutes"] = int(settings.self_learning_refresh_minutes)
    payload["retrain_hours"] = int(settings.self_learning_retrain_hours)
    return payload
