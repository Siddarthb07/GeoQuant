from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score


@dataclass
class PerformanceSummary:
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float

    def to_dict(self) -> Dict:
        return {
            "total_return_pct": self.total_return_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown_pct": self.max_drawdown_pct,
            "win_rate_pct": self.win_rate_pct,
            "profit_factor": self.profit_factor,
        }


def safe_series(values: Iterable[float]) -> pd.Series:
    series = pd.Series(list(values), dtype="float64")
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def compute_total_return_pct(equity_curve: pd.Series) -> float:
    clean = safe_series(equity_curve)
    if clean.empty:
        return 0.0
    start = float(clean.iloc[0])
    end = float(clean.iloc[-1])
    if start <= 0:
        return 0.0
    return float((end / start - 1.0) * 100.0)


def compute_sharpe_ratio(returns: pd.Series, periods_per_year: int) -> float:
    clean = safe_series(returns)
    if clean.empty:
        return 0.0
    std = float(clean.std(ddof=0))
    if std <= 1e-12:
        return 0.0
    annualization = float(np.sqrt(max(1, periods_per_year)))
    return float((clean.mean() / std) * annualization)


def compute_max_drawdown_pct(equity_curve: pd.Series) -> float:
    clean = safe_series(equity_curve)
    if clean.empty:
        return 0.0
    running_max = clean.cummax().replace(0, np.nan)
    drawdown = clean / running_max - 1.0
    return float(drawdown.min() * 100.0)


def compute_win_rate_pct(trade_pnl: pd.Series) -> float:
    clean = safe_series(trade_pnl)
    if clean.empty:
        return 0.0
    wins = float((clean > 0).sum())
    total = float(len(clean))
    return float((wins / total) * 100.0)


def compute_profit_factor(trade_pnl: pd.Series) -> float:
    clean = safe_series(trade_pnl)
    if clean.empty:
        return 0.0
    gross_profit = float(clean[clean > 0].sum())
    gross_loss = float(-clean[clean < 0].sum())
    if gross_loss <= 1e-12:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def summarize_performance(
    equity_curve: pd.Series,
    trade_pnl: pd.Series,
    periods_per_year: int,
) -> PerformanceSummary:
    returns = safe_series(equity_curve).pct_change().dropna()
    profit_factor = compute_profit_factor(trade_pnl)
    return PerformanceSummary(
        total_return_pct=round(compute_total_return_pct(equity_curve), 4),
        sharpe_ratio=round(compute_sharpe_ratio(returns, periods_per_year), 6),
        max_drawdown_pct=round(compute_max_drawdown_pct(equity_curve), 4),
        win_rate_pct=round(compute_win_rate_pct(trade_pnl), 4),
        profit_factor=round(profit_factor, 6) if np.isfinite(profit_factor) else float("inf"),
    )


def apply_vol_targeted_equity(
    equity_curve: pd.DataFrame,
    *,
    target_vol: float,
    lookback: int = 20,
    max_leverage: float = 2.0,
    cost_bps: float = 15.0,
    timestamp_col: str = "timestamp",
    equity_col: str = "equity",
) -> pd.DataFrame:
    """
    Rebuild equity by applying lagged constant-vol leverage to daily strategy returns.

    leverage_t = clip(target_vol / realized_vol_t, 0, max_leverage)
    r_vt_t = leverage_{t-1} * r_t - |leverage_t - leverage_{t-1}| * cost
    """
    if equity_curve is None or equity_curve.empty or target_vol <= 0:
        return equity_curve

    frame = equity_curve[[timestamp_col, equity_col]].copy().sort_values(timestamp_col).reset_index(drop=True)
    rets = frame[equity_col].pct_change()
    realized = rets.rolling(max(5, int(lookback))).std(ddof=0) * float(np.sqrt(252.0))
    leverage = (float(target_vol) / realized.replace(0, np.nan)).clip(lower=0.0, upper=float(max_leverage))
    leverage = leverage.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    lagged = leverage.shift(1).fillna(1.0)
    turnover = (leverage - lagged).abs().fillna(0.0)
    cost = turnover * (float(max(0.0, cost_bps)) / 10000.0)
    vt_rets = lagged * rets.fillna(0.0) - cost
    vt_rets.iloc[0] = 0.0

    start = float(frame[equity_col].iloc[0])
    vt_equity = start * (1.0 + vt_rets).cumprod()
    out = frame.copy()
    out[equity_col] = vt_equity.to_numpy(dtype=float)
    if "drawdown_pct" in equity_curve.columns or True:
        running_max = out[equity_col].cummax().replace(0, np.nan)
        out["drawdown_pct"] = (out[equity_col] / running_max - 1.0) * 100.0
    return out


def compute_classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    if not y_true:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "support": 0,
        }

    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    return {
        "accuracy": round(float(accuracy_score(y_true_arr, y_pred_arr)), 6),
        "precision": round(float(precision_score(y_true_arr, y_pred_arr, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true_arr, y_pred_arr, zero_division=0)), 6),
        "confusion_matrix": cm.tolist(),
        "support": int(len(y_true_arr)),
    }
