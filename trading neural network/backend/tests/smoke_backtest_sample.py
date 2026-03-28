from __future__ import annotations

import pandas as pd

from app.services.backtest_engine import BacktestConfig, run_backtest
from app.services.performance_metrics import summarize_performance


def main() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01 09:15:00", periods=5, freq="5min", tz="UTC"),
            "symbol": ["TEST"] * 5,
            "close": [100.0, 102.0, 104.0, 103.0, 101.0],
            "signal": [1, 1, 0, -1, 0],
        }
    )
    cfg = BacktestConfig(
        initial_capital=10000.0,
        position_fraction=1.0,
        brokerage_fee_bps=0.0,
        slippage_bps=0.0,
        allow_short=True,
    )
    result = run_backtest(frame, cfg=cfg)
    equity = result["equity_curve"]["equity"]
    trades = result["trade_log"]
    summary = summarize_performance(
        equity_curve=equity,
        trade_pnl=trades["net_pnl"],
        periods_per_year=19656,
    )

    assert len(trades) == 2, f"expected 2 closed trades, got {len(trades)}"
    assert float(result["final_equity"]) > 10600.0, f"unexpected final equity {result['final_equity']}"
    assert summary.total_return_pct > 6.0, f"expected >6% return, got {summary.total_return_pct}"
    print("smoke_backtest_sample: PASS")


if __name__ == "__main__":
    main()
