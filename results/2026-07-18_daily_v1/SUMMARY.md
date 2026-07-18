# GeoQuant measured run — 2026-07-18 · daily_v1

First committed walk-forward result for the public case file.

## Setup

| Item | Value |
|------|-------|
| Config | `config.results_daily.yaml` (copy in this folder) |
| Interval | `1d` |
| Symbols | AAPL, MSFT, NVDA, AMZN, GOOGL, META, RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS |
| Train / test | 2018-01-30 → train; **2022-01-01 → 2025-12-31** test |
| Walk-forward | 3-month steps · 16 folds |
| Costs | 10 bps brokerage + 5 bps slippage per side · 10% position fraction |
| Seed | 42 |
| Model | sklearn `MLPClassifier` (96, 48) · max 40 epochs · early stopping |

## Headline metrics (from `report.json`)

| Chip (site) | Field | Value |
|-------------|-------|-------|
| SHARPE | `performance_metrics.sharpe_ratio` | **−0.47** |
| MDD | `performance_metrics.max_drawdown_pct` | **37.1%** (reported as −37.1484) |
| HIT | `directional_trade_metrics.directional_hit_rate` | **51.1%** |
| YR | test span | **4** |

Also: total return **−32.9%** vs buy-and-hold **+197.9%** on the same window (`benchmark_comparison`). Classification accuracy on walk-forward predictions ≈ **51.2%**.

## Caveats (read these)

- **Weak / negative risk-adjusted return.** Costs are inside the optimizer; the strategy still underperforms a dumb long basket over this window.
- Yahoo daily bars + free feed — no proprietary tick data.
- MLP folds hit the epoch cap with ConvergenceWarnings on some folds; early stopping still applied when validation plateaued.
- This is **paper research**, not live PnL. Alpaca paper routing is separate from this backtest.

## Reproduce

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py --config config.results_daily.yaml --no-persist-model
```

Artifacts land in `backend/artifacts/latest/` (gitignored). Diff `report.json` against this folder.

## Files

- `report.json` — full pipeline export
- `equity_curve.csv` — strategy equity + drawdown
- `benchmark_curve.csv` — buy-and-hold comparison
- `equity_spark.json` — 80-point downsample used on the portfolio site chart
- `config.results_daily.yaml` — exact config for this run
