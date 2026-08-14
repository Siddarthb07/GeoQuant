# GeoQuant measured run — 2026-08-17 · dual-MA intermediate

Long-only dual-MA trend (no vol overlay). Intermediate step between v1 and public **v2**.

## Headline metrics (from `report.json`)

| Chip | Value |
|------|-------|
| SHARPE | **1.21** |
| MDD | **−23.8%** |
| RET | **+189.0%** |
| YR | **4** |

### vs daily_v1 baseline

| Metric | v1 (MLP L/S) | intermediate (trend only) |
|--------|--------------|---------------------------|
| Sharpe | −0.47 | **1.21** |
| Total return | −32.9% | **+189.0%** |
| Max drawdown | −37.1% | **−23.8%** |
| Excess vs B&H | −230.8 pp | **+12.9 pp** |

Public **v2** (Sharpe **1.36**, vol-targeted): [`../2026-08-17_daily_v3/`](../2026-08-17_daily_v3/).

## Reproduce

Shared bars live under the public v2 folder:

```powershell
cd backend
.\.venv\Scripts\python.exe train.py --config ..\results\2026-08-17_daily_v2\config.improved_daily.yaml --csv-path ..\results\2026-08-17_daily_v3\market_daily_us10.csv --no-persist-model
```

## Files

- `report.json`, `equity_curve.csv`, `benchmark_curve.csv`, `trade_log.csv`
- `config.improved_daily.yaml`, `equity_spark.json`
