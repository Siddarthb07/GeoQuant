# Measured results

Committed walk-forward outputs from the research pipeline. `backend/artifacts/` stays gitignored; this folder is the public, reviewable evidence.

## Latest run

**[`2026-07-18_daily_v1/`](./2026-07-18_daily_v1/)** — daily bars, costs inside the optimizer.

| Metric | Value |
|--------|-------|
| Sharpe | **−0.47** |
| Max drawdown | **37.1%** |
| Directional hit rate | **51.1%** |
| Test window | **2022 → 2025** (~4 yr) |
| Total return | −32.9% |
| Buy-and-hold (same window) | +197.9% |

Honest read: this baseline **loses money** after fees/slippage and underperforms buy-and-hold. That is expected for a first public cost-aware walk-forward on a mixed US/India mega-cap basket — the point is a **measured** number, not a marketing curve.

### Reproduce

```powershell
cd backend
.\.venv\Scripts\python.exe train.py --config config.results_daily.yaml --no-persist-model
```

Then compare `artifacts/latest/report.json` to `results/2026-07-18_daily_v1/report.json`.
