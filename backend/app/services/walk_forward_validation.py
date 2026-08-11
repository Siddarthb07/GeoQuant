from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from app.services.backtest_engine import BacktestConfig, run_backtest
from app.services.neural_model import predict_proba, train_binary_classifier
from app.services.performance_metrics import (
    apply_vol_targeted_equity,
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
    allow_short: bool = True
    use_trend_filter: bool = False
    require_long_trend: bool = False
    min_hold_bars: int = 1
    # model | trend_only | trend_primary | breakout
    signal_mode: str = "model"
    # Keep only the top-N long candidates by rank_col each day (0 = disabled).
    top_n: int = 0
    rank_col: str = "risk_adj_mom"
    # Recompute portfolio membership every N bars; hold prior membership between.
    rebalance_every: int = 1
    # Target gross exposure for inverse-vol weights (0 = use fixed position_fraction).
    gross_exposure: float = 0.0
    max_position_fraction: float = 0.35


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
    if "trend_ok" not in out.columns:
        out["trend_ok"] = 1.0
    if "trend_long" not in out.columns:
        out["trend_long"] = out["trend_ok"]
    for col in (
        "mom_21",
        "mom_63",
        "mom_126",
        "mom_252_skip21",
        "risk_adj_mom",
        "vol_20",
        "breakout_20",
        "breakout_55",
        "breakdown_10",
        "breakdown_20",
    ):
        if col not in out.columns:
            out[col] = np.nan
    out = out.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return out


def _probability_to_signal(
    probability: np.ndarray,
    buy_threshold: float,
    sell_threshold: float,
    *,
    allow_short: bool = True,
) -> np.ndarray:
    signals = np.zeros_like(probability, dtype=int)
    signals[probability >= buy_threshold] = 1
    if allow_short:
        signals[probability <= sell_threshold] = -1
    return signals


def _apply_trend_filter(
    signals: np.ndarray,
    trend_ok: np.ndarray,
    trend_long: np.ndarray,
    *,
    use_trend_filter: bool,
    require_long_trend: bool,
) -> np.ndarray:
    if not use_trend_filter:
        return signals
    out = signals.copy()
    gate = trend_long if require_long_trend else trend_ok
    out[(out > 0) & (gate < 0.5)] = 0
    return out


def _compose_signals(
    probability: np.ndarray,
    trend_ok: np.ndarray,
    trend_long: np.ndarray,
    config: WalkForwardConfig,
    *,
    breakout_20: np.ndarray | None = None,
    breakout_55: np.ndarray | None = None,
    breakdown_10: np.ndarray | None = None,
    breakdown_20: np.ndarray | None = None,
) -> np.ndarray:
    mode = str(config.signal_mode or "model").strip().lower()
    gate = trend_long if config.require_long_trend else trend_ok

    if mode == "trend_only":
        return (gate >= 0.5).astype(int)

    if mode == "breakout":
        # Filled later per-symbol after predictions are assembled.
        return np.zeros(len(probability), dtype=int)

    if mode == "trend_primary":
        # Stay long through uptrends; only step aside when the model is clearly bearish.
        signals = np.zeros_like(probability, dtype=int)
        bullish_enough = probability > config.sell_threshold
        signals[(gate >= 0.5) & bullish_enough] = 1
        if config.allow_short:
            signals[(gate < 0.5) & (probability <= config.sell_threshold)] = -1
        return signals

    signals = _probability_to_signal(
        probability,
        config.buy_threshold,
        config.sell_threshold,
        allow_short=config.allow_short,
    )
    return _apply_trend_filter(
        signals,
        trend_ok,
        trend_long,
        use_trend_filter=config.use_trend_filter,
        require_long_trend=config.require_long_trend,
    )


def _apply_min_hold(predictions: pd.DataFrame, min_hold_bars: int) -> pd.DataFrame:
    hold = max(1, int(min_hold_bars))
    if hold <= 1 or predictions.empty:
        return predictions

    out = predictions.copy()
    adjusted: List[pd.Series] = []
    for _, group in out.groupby("symbol", sort=False):
        local = group.sort_values("timestamp").copy()
        signals = local["signal"].to_numpy(dtype=int).copy()
        last_side = 0
        held = 0
        for i, raw in enumerate(signals):
            if last_side == 0:
                last_side = int(raw)
                held = 1 if last_side != 0 else 0
                signals[i] = last_side
                continue
            held += 1
            if held < hold and int(raw) != last_side:
                signals[i] = last_side
                continue
            last_side = int(raw)
            held = 1 if last_side != 0 else 0
            signals[i] = last_side
        local["signal"] = signals
        adjusted.append(local)
    return pd.concat(adjusted, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _apply_breakout_state(predictions: pd.DataFrame) -> pd.DataFrame:
    """Stateful Donchian long-only: enter on 20/55 breakout, exit on 10/20 breakdown."""
    if predictions.empty:
        return predictions
    required = {"breakout_20", "breakout_55", "breakdown_10", "breakdown_20"}
    if not required.issubset(predictions.columns):
        return predictions

    pieces: List[pd.DataFrame] = []
    for _, group in predictions.groupby("symbol", sort=False):
        local = group.sort_values("timestamp").copy()
        signals = np.zeros(len(local), dtype=int)
        in_pos = 0
        b20 = local["breakout_20"].to_numpy(dtype=float)
        b55 = local["breakout_55"].to_numpy(dtype=float)
        d10 = local["breakdown_10"].to_numpy(dtype=float)
        d20 = local["breakdown_20"].to_numpy(dtype=float)
        for i in range(len(local)):
            if in_pos == 0:
                if b20[i] >= 0.5 or b55[i] >= 0.5:
                    in_pos = 1
            elif d10[i] >= 0.5 or d20[i] >= 0.5:
                in_pos = 0
            signals[i] = in_pos
        local["signal"] = signals
        pieces.append(local)
    return pd.concat(pieces, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _apply_top_n_rank(
    predictions: pd.DataFrame,
    *,
    top_n: int,
    rank_col: str,
) -> pd.DataFrame:
    n = int(top_n)
    if n <= 0 or predictions.empty or rank_col not in predictions.columns:
        return predictions

    out = predictions.copy()
    score = pd.to_numeric(out[rank_col], errors="coerce")
    out["_rank_score"] = score
    kept_rows: List[pd.DataFrame] = []
    for _, group in out.groupby("timestamp", sort=True):
        local = group.copy()
        long_mask = local["signal"] > 0
        if not long_mask.any():
            kept_rows.append(local)
            continue
        candidates = local.loc[long_mask].sort_values("_rank_score", ascending=False, na_position="last")
        winners = set(candidates.head(n)["symbol"].astype(str).tolist())
        local.loc[long_mask & ~local["symbol"].astype(str).isin(winners), "signal"] = 0
        kept_rows.append(local)
    ranked = pd.concat(kept_rows, ignore_index=True)
    return ranked.drop(columns=["_rank_score"], errors="ignore")


def _apply_rebalance_schedule(predictions: pd.DataFrame, rebalance_every: int) -> pd.DataFrame:
    every = max(1, int(rebalance_every))
    if every <= 1 or predictions.empty:
        return predictions

    out = predictions.sort_values(["timestamp", "symbol"]).copy()
    timestamps = sorted(out["timestamp"].unique())
    rebalance_set = set(timestamps[::every])
    # Always include first timestamp.
    if timestamps:
        rebalance_set.add(timestamps[0])

    held: Dict[str, int] = {}
    adjusted: List[pd.DataFrame] = []
    for ts, group in out.groupby("timestamp", sort=True):
        local = group.copy()
        if ts in rebalance_set:
            held = {
                str(row["symbol"]): int(row["signal"])
                for _, row in local.iterrows()
                if int(row["signal"]) != 0
            }
            # Flat names stay flat unless held.
            local["signal"] = local["symbol"].astype(str).map(lambda s: held.get(s, 0)).astype(int)
        else:
            local["signal"] = local["symbol"].astype(str).map(lambda s: held.get(s, 0)).astype(int)
            # Drop names that disappeared from the panel but keep held membership for symbols present.
        adjusted.append(local)
    return pd.concat(adjusted, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _assign_inverse_vol_weights(
    predictions: pd.DataFrame,
    *,
    gross_exposure: float,
    max_position_fraction: float,
) -> pd.DataFrame:
    """Risk-parity style weights: w_i ∝ 1/vol, scaled to target gross exposure."""
    out = predictions.copy()
    if out.empty:
        out["weight"] = []
        return out

    gross = float(max(0.0, gross_exposure))
    cap = float(max(1e-6, max_position_fraction))
    out["weight"] = 0.0
    if gross <= 0:
        return out

    pieces: List[pd.DataFrame] = []
    for _, group in out.groupby("timestamp", sort=True):
        local = group.copy()
        active = local["signal"] != 0
        if not active.any():
            pieces.append(local)
            continue
        vol = pd.to_numeric(local.loc[active, "vol_20"], errors="coerce").replace(0, np.nan)
        inv = 1.0 / vol
        if not np.isfinite(inv.to_numpy(dtype=float)).any():
            n = int(active.sum())
            local.loc[active, "weight"] = min(cap, gross / max(n, 1))
            pieces.append(local)
            continue
        inv = inv.fillna(inv.median() if np.isfinite(inv.median()) else 1.0)
        raw = inv / inv.sum() * gross
        local.loc[active, "weight"] = np.minimum(raw.to_numpy(dtype=float), cap)
        # Renormalize if caps bind and we still have room.
        used = float(local.loc[active, "weight"].sum())
        if used > 1e-12 and used < gross - 1e-9:
            scale = gross / used
            local.loc[active, "weight"] = np.minimum(
                local.loc[active, "weight"].to_numpy(dtype=float) * scale,
                cap,
            )
        pieces.append(local)
    return pd.concat(pieces, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)


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

        mode = str(config.signal_mode or "model").strip().lower()
        y_test = test_rows["target_up"].astype(int).to_numpy()
        if mode in {"trend_only", "breakout"}:
            prob_up = np.full(len(test_rows), 0.5, dtype=float)
            pred_class = np.zeros(len(test_rows), dtype=int)
            model_auc = 0.5
            model_acc = float((y_test == 0).mean()) if len(y_test) else 0.5
        else:
            train_part, val_part = _train_validate_split(train_rows)
            x_train = train_part[feature_columns].astype(float).to_numpy()
            y_train = train_part["target_up"].astype(int).to_numpy()
            x_val = val_part[feature_columns].astype(float).to_numpy()
            y_val = val_part["target_up"].astype(int).to_numpy()
            x_test = test_rows[feature_columns].astype(float).to_numpy()

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
            model_auc = float(model_result.auc)
            model_acc = float((pred_class == y_test).mean())

        signals = _compose_signals(
            prob_up,
            test_rows["trend_ok"].to_numpy(dtype=float),
            test_rows["trend_long"].to_numpy(dtype=float),
            config,
        )

        pred_frame = test_rows[
            [
                "timestamp",
                "symbol",
                "close",
                "target_up",
                "vol_20",
                "mom_63",
                "mom_126",
                "mom_252_skip21",
                "risk_adj_mom",
                "breakout_20",
                "breakout_55",
                "breakdown_10",
                "breakdown_20",
            ]
        ].copy()
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
                "accuracy": round(float(model_acc), 6),
                "auc": round(float(model_auc), 6),
                "signal_count": int((signals != 0).sum()),
            }
        )

    if not all_predictions:
        raise RuntimeError(
            "Walk-forward produced no prediction windows. Provide broader data coverage or reduce minimum row settings."
        )

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions = predictions.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    if str(config.signal_mode or "").strip().lower() == "breakout":
        predictions = _apply_breakout_state(predictions)
    predictions = _apply_top_n_rank(
        predictions,
        top_n=config.top_n,
        rank_col=config.rank_col,
    )
    predictions = _apply_rebalance_schedule(predictions, config.rebalance_every)
    predictions = _apply_min_hold(predictions, config.min_hold_bars)
    if float(config.gross_exposure) > 0:
        predictions = _assign_inverse_vol_weights(
            predictions,
            gross_exposure=config.gross_exposure,
            max_position_fraction=config.max_position_fraction,
        )
    else:
        predictions = predictions.copy()
        predictions["weight"] = float("nan")

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

    backtest_cols = ["timestamp", "symbol", "close", "signal", "weight", "vol_20"]
    backtest_input = predictions[backtest_cols].copy()
    # Entry-level portfolio vol scaling is coarse; prefer return-level targeting below.
    bt_cfg_local = bt_cfg
    if getattr(bt_cfg, "portfolio_vol_target", 0.0) and float(bt_cfg.portfolio_vol_target) > 0:
        from dataclasses import replace

        bt_cfg_local = replace(bt_cfg, portfolio_vol_target=0.0)

    bt_result = run_backtest(backtest_input, cfg=bt_cfg_local)
    equity_curve = bt_result["equity_curve"]
    trade_log = bt_result["trade_log"]
    benchmark_curve = bt_result["benchmark_curve"]

    if getattr(bt_cfg, "portfolio_vol_target", 0.0) and float(bt_cfg.portfolio_vol_target) > 0 and not equity_curve.empty:
        round_trip_bps = float(bt_cfg.brokerage_fee_bps + bt_cfg.slippage_bps)
        equity_curve = apply_vol_targeted_equity(
            equity_curve,
            target_vol=float(bt_cfg.portfolio_vol_target),
            lookback=int(getattr(bt_cfg, "portfolio_vol_lookback", 20) or 20),
            max_leverage=float(getattr(bt_cfg, "max_leverage", 2.0) or 2.0),
            cost_bps=round_trip_bps,
        )
        bt_result = {
            **bt_result,
            "equity_curve": equity_curve,
            "final_equity": float(equity_curve["equity"].iloc[-1]),
        }

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
