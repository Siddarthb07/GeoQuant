# Measured results

Committed walk-forward outputs from the research pipeline. `backend/artifacts/` stays gitignored; this folder is the public, reviewable evidence.

## Public v2 (latest)

**[`2026-08-17_daily_v3/`](./2026-08-17_daily_v3/)** — dual-MA trend + lagged 18% portfolio vol target. **This is public v2.**

| Metric | v1 | **v2** |
|--------|----|--------|
| Sharpe | −0.47 | **1.36** |
| Max drawdown | 37.1% | **16.9%** |
| Total return | −32.9% | **+167%** |
| Test window | 2022 → 2025 | 2022 → 2025 |

```powershell
cd backend
.\.venv\Scripts\python.exe train.py --config config.improved_daily.yaml --csv-path ..\results\2026-08-17_daily_v3\market_daily_us10.csv --no-persist-model
```

## Intermediate (dual-MA only)

**[`2026-08-17_daily_v2/`](./2026-08-17_daily_v2/)** — same trend rule, no vol overlay. Sharpe **1.21**, MDD **23.8%**, return **+189%**.

## Baseline v1 (kept for honesty)

**[`2026-07-18_daily_v1/`](./2026-07-18_daily_v1/)** — original MLP long/short.

| Sharpe | MDD | Hit rate |
|--------|-----|----------|
| −0.47 | 37.1% | 51.1% |
