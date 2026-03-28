from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from app.services.backtest_engine import BacktestConfig, run_backtest
from app.services.neural_model import predict_proba, train_binary_classifier
from app.services.performance_metrics import (
    compute_classification_metrics,
    summarize_performance,
)


@dataclass
class WalkForwardConfig:
    train_start: str = "2000-01-01"
    test_start: str = "2020-01-01"
    test_end: str = "2025-12-31"
    step_months: int = 3
    min_train_rows: int = 800
    min_test_rows: int = 100
    buy_threshold: float = 0.55
    sell_threshold: float = 0.45
    periods_per_year: int = 19656  # 5-minute bars for ~252 sessions * 78 bars/day
    epochs: int = 35
    batch_size: int = 512
    learning_rate: float = 1e-3


def _prepare_frame(frame: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    required = ["timestamp", "symbol", "close", "target_up"] + list(feature_columns)
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Feature dataset missing columns: {', '.join(missing)}")

    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp", "symbol", "close", "target_up"]).copy()
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_columns)
    out["target_up"] = out["target_up"].astype(int)
    out = out.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return out


def _probability_to_signal(probability: np.ndarray, buy_threshold: float, sell_threshold: float) -> np.ndarray:
    signals = np.zeros_like(probability, dtype=int)
    signals[probability >= buy_threshold] = 1
    signals[probability <= sell_threshold] = -1
    return signals


def _train_validate_split(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(train_df) * 0.85)
    split_idx = min(max(split_idx, 1), len(train_df) - 1)
    train_part = train_df.iloc[:split_idx].copy()
    val_part = train_df.iloc[split_idx:].copy()
    return train_part, val_part


def _fold_windows(config: WalkForwardConfig) -> List[tuple[pd.Timestamp, pd.Timestamp]]:
    windows: List[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = pd.Timestamp(config.test_start, tz="UTC")
    end = pd.Timestamp(config.test_end, tz="UTC")
    step = max(1, int(config.step_months))

    while cursor < end:
        next_cursor = min(cursor + pd.DateOffset(months=step), end + pd.Timedelta(days=1))
        windows.append((cursor, next_cursor))
        cursor = next_cursor
    return windows


def run_walk_forward_validation(
    feature_frame: pd.DataFrame,
    feature_columns: List[str],
    walk_cfg: Optional[WalkForwardConfig] = None,
    backtest_cfg: Optional[BacktestConfig] = None,
) -> Dict:
    config = walk_cfg or WalkForwardConfig()
    bt_cfg = backtest_cfg or BacktestConfig()
    data = _prepare_frame(feature_frame, feature_columns=feature_columns)

    train_start = pd.Timestamp(config.train_start, tz="UTC")
    all_predictions: List[pd.DataFrame] = []
    fold_stats: List[Dict] = []

    for fold_id, (window_start, window_end) in enumerate(_fold_windows(config), start=1):
        train_mask = (data["timestamp"] >= train_start) & (data["timestamp"] < window_start)
        test_mask = (data["timestamp"] >= window_start) & (data["timestamp"] < window_end)
        train_rows = data.loc[train_mask]
        test_rows = data.loc[test_mask]

        if len(train_rows) < config.min_train_rows or len(test_rows) < config.min_test_rows:
            fold_stats.append(
                {
                    "fold_id": fold_id,
                    "window_start": window_start.isoformat(),
                    "window_end": window_end.isoformat(),
                    "train_rows": int(len(train_rows)),
                    "test_rows": int(len(test_rows)),
                    "skipped": True,
                    "reason": "insufficient_rows",
                }
            )
            continue

        train_part, val_part = _train_validate_split(train_rows)
        x_train = train_part[feature_columns].astype(float).to_numpy()
        y_train = train_part["target_up"].astype(int).to_numpy()
        x_val = val_part[feature_columns].astype(float).to_numpy()
        y_val = val_part["target_up"].astype(int).to_numpy()
        x_test = test_rows[feature_columns].astype(float).to_numpy()
        y_test = test_rows["target_up"].astype(int).to_numpy()

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)
        x_test_scaled = scaler.transform(x_test)

        model_result = train_binary_classifier(
            x_train=x_train_scaled,
            y_train=y_train,
            x_val=x_val_scaled,
            y_val=y_val,
            epochs=config.epochs,
            batch_size=config.batch_size,
            lr=config.learning_rate,
        )

        prob_up = predict_proba(model_result.model, x_test_scaled).astype(float)
        pred_class = (prob_up >= 0.5).astype(int)
        signals = _probability_to_signal(prob_up, config.buy_threshold, config.sell_threshold)

        pred_frame = test_rows[["timestamp", "symbol", "close", "target_up"]].copy()
        pred_frame["fold_id"] = int(fold_id)
        pred_frame["prob_up"] = prob_up
        pred_frame["pred_class"] = pred_class
        pred_frame["signal"] = signals
        all_predictions.append(pred_frame)

        fold_stats.append(
            {
                "fold_id": fold_id,
                "window_start": window_start.isoformat(),
                "window_end": window_end.isoformat(),
                "train_rows": int(len(train_rows)),
                "test_rows": int(len(test_rows)),
                "skipped": False,
                "accuracy": round(float((pred_class == y_test).mean()), 6),
                "auc": round(float(model_result.auc), 6),
                "signal_count": int((signals != 0).sum()),
            }
        )

    if not all_predictions:
        raise RuntimeError(
            "Walk-forward produced no prediction windows. Provide broader data coverage or reduce minimum row settings."
        )

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions = predictions.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    class_metrics = compute_classification_metrics(
        y_true=predictions["target_up"].astype(int).tolist(),
        y_pred=predictions["pred_class"].astype(int).tolist(),
    )

    directional_mask = predictions["signal"] != 0
    directional_subset = predictions.loc[directional_mask].copy()
    if directional_subset.empty:
        directional_hit_rate = 0.0
        directional_samples = 0
    else:
        directional_correct = (
            ((directional_subset["signal"] == 1) & (directional_subset["target_up"] == 1))
            | ((directional_subset["signal"] == -1) & (directional_subset["target_up"] == 0))
        )
        directional_hit_rate = float(directional_correct.mean())
        directional_samples = int(len(directional_subset))

    backtest_input = predictions[["timestamp", "symbol", "close", "signal"]].copy()
    bt_result = run_backtest(backtest_input, cfg=bt_cfg)
    equity_curve = bt_result["equity_curve"]
    trade_log = bt_result["trade_log"]
    benchmark_curve = bt_result["benchmark_curve"]

    trade_pnl = trade_log["net_pnl"] if not trade_log.empty and "net_pnl" in trade_log.columns else pd.Series(dtype=float)
    perf = summarize_performance(
        equity_curve=equity_curve["equity"] if "equity" in equity_curve.columns else pd.Series(dtype=float),
        trade_pnl=trade_pnl,
        periods_per_year=config.periods_per_year,
    )

    benchmark_return = 0.0
    if not benchmark_curve.empty:
        start_b = float(benchmark_curve["benchmark_equity"].iloc[0])
        end_b = float(benchmark_curve["benchmark_equity"].iloc[-1])
        if start_b > 0:
            benchmark_return = float((end_b / start_b - 1.0) * 100.0)

    return {
        "predictions": predictions,
        "fold_metrics": fold_stats,
        "classification_metrics": class_metrics,
        "directional_trade_metrics": {
            "directional_hit_rate": round(directional_hit_rate, 6),
            "samples": int(directional_samples),
        },
        "backtest": bt_result,
        "performance": perf.to_dict(),
        "benchmark": {
            "buy_and_hold_return_pct": round(benchmark_return, 4),
            "excess_return_pct": round(perf.total_return_pct - benchmark_return, 4),
        },
    }
