from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    initial_capital: float = 100000.0
    position_fraction: float = 0.10
    brokerage_fee_bps: float = 10.0
    slippage_bps: float = 5.0
    allow_short: bool = True
    timestamp_col: str = "timestamp"
    symbol_col: str = "symbol"
    price_col: str = "close"
    signal_col: str = "signal"

    @property
    def fee_rate(self) -> float:
        return max(0.0, self.brokerage_fee_bps) / 10000.0

    @property
    def slippage_rate(self) -> float:
        return max(0.0, self.slippage_bps) / 10000.0


def _ensure_required_columns(frame: pd.DataFrame, cfg: BacktestConfig) -> None:
    required = [cfg.timestamp_col, cfg.symbol_col, cfg.price_col, cfg.signal_col]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Backtest input is missing required columns: {', '.join(missing)}")


def _normalize_signals(series: pd.Series, allow_short: bool) -> pd.Series:
    normalized = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
    normalized = normalized.clip(lower=-1, upper=1)
    if not allow_short:
        normalized = normalized.clip(lower=0, upper=1)
    return normalized


def _mark_to_market(cash: float, positions: Dict[str, Dict], last_prices: Dict[str, float]) -> float:
    value = cash
    for symbol, pos in positions.items():
        px = float(last_prices.get(symbol, pos["last_price"]))
        value += float(pos["qty"]) * px
    return float(value)


def _entry_execution_price(price: float, side: int, slippage_rate: float) -> float:
    if side > 0:
        return float(price * (1.0 + slippage_rate))
    return float(price * (1.0 - slippage_rate))


def _exit_execution_price(price: float, side: int, slippage_rate: float) -> float:
    if side > 0:
        return float(price * (1.0 - slippage_rate))
    return float(price * (1.0 + slippage_rate))


def _open_position(
    symbol: str,
    timestamp: pd.Timestamp,
    side: int,
    price: float,
    equity_now: float,
    cash: float,
    cfg: BacktestConfig,
    positions: Dict[str, Dict],
    trade_log: List[Dict],
) -> float:
    if equity_now <= 0:
        return cash
    notional = float(max(0.0, equity_now * cfg.position_fraction))
    if notional <= 0:
        return cash

    entry_price = _entry_execution_price(price, side, cfg.slippage_rate)
    if entry_price <= 0:
        return cash

    qty = float(notional / entry_price)
    if qty <= 0:
        return cash

    fee = float(notional * cfg.fee_rate)
    signed_qty = qty if side > 0 else -qty

    if side > 0:
        cash = float(cash - qty * entry_price - fee)
    else:
        cash = float(cash + qty * entry_price - fee)

    positions[symbol] = {
        "symbol": symbol,
        "side": side,
        "qty": signed_qty,
        "entry_price": entry_price,
        "entry_notional": notional,
        "entry_fee": fee,
        "entry_time": timestamp,
        "last_price": price,
    }
    trade_log.append(
        {
            "symbol": symbol,
            "entry_time": timestamp.isoformat(),
            "exit_time": None,
            "side": "LONG" if side > 0 else "SHORT",
            "qty": round(float(abs(signed_qty)), 8),
            "entry_price": round(entry_price, 8),
            "exit_price": None,
            "gross_pnl": None,
            "fees": round(fee, 8),
            "net_pnl": None,
            "return_pct": None,
        }
    )
    return cash


def _close_position(
    symbol: str,
    timestamp: pd.Timestamp,
    price: float,
    cash: float,
    cfg: BacktestConfig,
    positions: Dict[str, Dict],
    trade_log: List[Dict],
) -> float:
    pos = positions.pop(symbol, None)
    if pos is None:
        return cash

    side = int(pos["side"])
    qty = float(abs(pos["qty"]))
    entry_price = float(pos["entry_price"])
    entry_fee = float(pos["entry_fee"])
    exit_price = _exit_execution_price(price, side, cfg.slippage_rate)
    exit_notional = float(qty * exit_price)
    exit_fee = float(exit_notional * cfg.fee_rate)

    if side > 0:
        cash = float(cash + qty * exit_price - exit_fee)
        gross_pnl = float((exit_price - entry_price) * qty)
    else:
        cash = float(cash - qty * exit_price - exit_fee)
        gross_pnl = float((entry_price - exit_price) * qty)

    fees_total = float(entry_fee + exit_fee)
    net_pnl = float(gross_pnl - fees_total)
    invested = float(max(1e-12, qty * entry_price))
    trade_return = float((net_pnl / invested) * 100.0)

    for row in range(len(trade_log) - 1, -1, -1):
        item = trade_log[row]
        if item["symbol"] != symbol or item["exit_time"] is not None:
            continue
        item["exit_time"] = timestamp.isoformat()
        item["exit_price"] = round(exit_price, 8)
        item["gross_pnl"] = round(gross_pnl, 8)
        item["fees"] = round(float(item["fees"]) + exit_fee, 8)
        item["net_pnl"] = round(net_pnl, 8)
        item["return_pct"] = round(trade_return, 8)
        break

    return cash


def _prepare_input(frame: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    _ensure_required_columns(frame, cfg)
    out = frame.copy()
    out[cfg.timestamp_col] = pd.to_datetime(out[cfg.timestamp_col], utc=True, errors="coerce")
    out[cfg.price_col] = pd.to_numeric(out[cfg.price_col], errors="coerce")
    out = out.dropna(subset=[cfg.timestamp_col, cfg.symbol_col, cfg.price_col]).copy()
    out[cfg.symbol_col] = out[cfg.symbol_col].astype(str).str.strip()
    out = out[out[cfg.symbol_col] != ""].copy()
    out[cfg.signal_col] = _normalize_signals(out[cfg.signal_col], allow_short=cfg.allow_short)
    out = out.sort_values([cfg.timestamp_col, cfg.symbol_col]).reset_index(drop=True)
    return out


def build_buy_hold_benchmark(frame: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    data = _prepare_input(frame, cfg)
    symbols = sorted(data[cfg.symbol_col].unique().tolist())
    if not symbols:
        return pd.DataFrame(columns=[cfg.timestamp_col, "benchmark_equity"])

    alloc = float(cfg.initial_capital / len(symbols))
    panel = data.pivot_table(
        index=cfg.timestamp_col,
        columns=cfg.symbol_col,
        values=cfg.price_col,
        aggfunc="last",
    ).sort_index()
    panel = panel.ffill().dropna(how="all")
    if panel.empty:
        return pd.DataFrame(columns=[cfg.timestamp_col, "benchmark_equity"])

    first_prices = panel.iloc[0]
    shares: Dict[str, float] = {}
    for symbol in symbols:
        px = float(first_prices.get(symbol, np.nan))
        if np.isfinite(px) and px > 0:
            shares[symbol] = alloc / px
        else:
            shares[symbol] = 0.0

    values = []
    for ts, row in panel.iterrows():
        equity = 0.0
        for symbol in symbols:
            px = float(row.get(symbol, np.nan))
            if not np.isfinite(px):
                continue
            equity += shares[symbol] * px
        values.append({"timestamp": ts, "benchmark_equity": float(equity)})

    return pd.DataFrame(values)


def run_backtest(frame: pd.DataFrame, cfg: Optional[BacktestConfig] = None) -> Dict:
    config = cfg or BacktestConfig()
    data = _prepare_input(frame, config)
    if data.empty:
        return {
            "equity_curve": pd.DataFrame(columns=["timestamp", "equity"]),
            "trade_log": pd.DataFrame(),
            "benchmark_curve": pd.DataFrame(columns=["timestamp", "benchmark_equity"]),
            "final_equity": float(config.initial_capital),
        }

    cash = float(config.initial_capital)
    positions: Dict[str, Dict] = {}
    last_prices: Dict[str, float] = {}
    equity_points: List[Dict] = []
    trade_log: List[Dict] = []

    grouped = data.groupby(config.timestamp_col, sort=True)
    last_timestamp: Optional[pd.Timestamp] = None
    for timestamp, rows in grouped:
        last_timestamp = timestamp
        for _, row in rows.iterrows():
            symbol = str(row[config.symbol_col])
            px = float(row[config.price_col])
            signal = int(row[config.signal_col])
            last_prices[symbol] = px

            current = positions.get(symbol)
            target_side = signal

            if current is not None:
                current_side = int(current["side"])
                current["last_price"] = px
                if target_side != current_side:
                    cash = _close_position(symbol, timestamp, px, cash, config, positions, trade_log)
                    current = None

            if current is None and target_side != 0:
                equity_now = _mark_to_market(cash, positions, last_prices)
                cash = _open_position(
                    symbol=symbol,
                    timestamp=timestamp,
                    side=target_side,
                    price=px,
                    equity_now=equity_now,
                    cash=cash,
                    cfg=config,
                    positions=positions,
                    trade_log=trade_log,
                )

        equity_now = _mark_to_market(cash, positions, last_prices)
        equity_points.append({"timestamp": timestamp, "equity": float(equity_now)})

    if last_timestamp is not None:
        for symbol in list(positions.keys()):
            px = float(last_prices.get(symbol, positions[symbol]["last_price"]))
            cash = _close_position(symbol, last_timestamp, px, cash, config, positions, trade_log)
        equity_points.append({"timestamp": last_timestamp, "equity": float(cash)})

    equity_curve = pd.DataFrame(equity_points).drop_duplicates(subset=["timestamp"], keep="last")
    equity_curve = equity_curve.sort_values("timestamp").reset_index(drop=True)
    if not equity_curve.empty:
        running_max = equity_curve["equity"].cummax().replace(0, np.nan)
        equity_curve["drawdown_pct"] = (equity_curve["equity"] / running_max - 1.0) * 100.0
    else:
        equity_curve["drawdown_pct"] = []

    trades = pd.DataFrame(trade_log)
    if not trades.empty:
        trades = trades.dropna(subset=["exit_time"]).reset_index(drop=True)

    benchmark_curve = build_buy_hold_benchmark(data, config)
    return {
        "equity_curve": equity_curve,
        "trade_log": trades,
        "benchmark_curve": benchmark_curve,
        "final_equity": float(cash),
    }
