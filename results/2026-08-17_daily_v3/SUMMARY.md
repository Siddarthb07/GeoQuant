# GeoQuant measured run — public **v2** · 2026-08-17

Vol-targeted long-only dual-MA sleeve. This is the public **v2** number set (v1 was the MLP long/short baseline at Sharpe **−0.47**).

## Headline

| Metric | v1 MLP L/S | **v2 trend + vol target** |
|--------|------------|---------------------------|
| Sharpe | −0.47 | **1.36** |
| Max drawdown | −37.1% | **−16.9%** |
| Total return | −32.9% | **+167%** |
| vs B&H | −231 pp | −9 pp |

## Setup

| Item | Value |
|------|-------|
| Config | `config.improved_daily.yaml` |
| Signal | long when `close > SMA50` and `SMA50 > SMA200` |
| Overlay | lagged portfolio vol target **18%**, max leverage **3×**, cost on leverage changes |
| Min hold | 21 bars |
| Costs | 10 bps fee + 5 bps slip per side |
| Symbols | AAPL, MSFT, NVDA, AMZN, GOOGL, META, AVGO, JPM, XOM, TSLA |
| Test | 2022-01-01 → 2025-12-31 |

## Why these numbers

v1 next-day MLP was ~coin-flip and shorted a mega-cap bull. v2 stops shorting, rides dual-MA trend, and applies a lagged constant-vol overlay so drawdowns stay usable.

## Reproduce

```powershell
cd backend
.\.venv\Scripts\python.exe train.py --config config.improved_daily.yaml --csv-path ..\results\2026-08-17_daily_v3\market_daily_us10.csv --no-persist-model
```

## Files

- `report.json`, `equity_curve.csv`, `benchmark_curve.csv`, `trade_log.csv`
- `config.improved_daily.yaml`, `market_daily_us10.csv`, `equity_spark.json`
