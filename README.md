# GeoQuant Neural Trader

[![Release](https://img.shields.io/github/v/release/Siddarthb07/GeoQuant?label=v1.0)](https://github.com/Siddarthb07/GeoQuant/releases/tag/v1.0)

**Keywords:** quantitative trading · algorithmic trading · FastAPI · PyTorch · Alpaca · backtesting · sentiment analysis · walk-forward validation

Full-stack **quantitative / algorithmic trading** platform for **US + India** equities. GeoQuant combines neural-network signal models, global news sentiment analysis, walk-forward backtesting research, and a live trading dashboard (Alpaca paper/live) in one FastAPI app.

## v1.0 highlights

- **Live dashboard** — news flow, candlestick charts, top 15 long/short candidates, paper/live order ticket
- **Reproducible research pipeline** — walk-forward validation, backtest with costs, benchmark comparison, Plotly artifacts
- **Self-learning loop** — resolved-signal feedback and scheduled retraining
- **Resilient market data** — parallel Yahoo Finance downloads, retries, ticker aliases (e.g. `TATAMOTORS.NS` → `TMPV.NS`)
- **Fast API startup** — background candidate cache warming; non-blocking chart/candidate endpoints
- **One-command launcher** — `run_all.ps1` runs research then starts the server

## Architecture

```
run_all.ps1
  ├── backend/train.py          # walk-forward research + model export
  └── uvicorn app.main:app      # dashboard + REST API

backend/app/
  ├── routers/api.py            # /api/candidates, /api/chart, /api/research/run, …
  ├── services/
  │   ├── research_pipeline.py
  │   ├── signal_service.py
  │   ├── walk_forward_validation.py
  │   └── backtest_engine.py
  ├── data/market_data.py       # yfinance daily + intraday
  └── templates/index.html      # single-page trading UI
```

## Quick start

### Prerequisites

- Python 3.10+
- Windows PowerShell (or use manual commands below)

### One command (recommended)

From the project root:

```powershell
.\run_all.ps1
```

This installs dependencies, runs the research pipeline, and starts the API at **http://127.0.0.1:8020**.

From the `backend` folder:

```powershell
..\run_all.ps1
# or
.\run_all.ps1
```

### Skip research (API only)

```powershell
.\run_all.ps1 -SkipResearch
```

### Manual setup

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --host 127.0.0.1 --port 8020
```

Open: **http://127.0.0.1:8020**

## Launcher options

```powershell
.\run_all.ps1 -BindHost 0.0.0.0 -Port 8020
.\run_all.ps1 -SkipInstall
.\run_all.ps1 -Reload
.\run_all.ps1 -InitialCapital 150000 -PositionFraction 0.12 -BrokerageFeeBps 8 -SlippageBps 6
.\run_all.ps1 -CsvPath "C:\data\geoquant_5m.csv"
.\run_all.ps1 -SkipResearch
```

If PowerShell blocks scripts, use `.\run_all.bat` or run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

## Measured walk-forward (public)

### Public v2 — vol-targeted trend sleeve

**[`results/2026-08-17_daily_v3/`](./results/2026-08-17_daily_v3/)** — long-only dual-MA + lagged 18% portfolio vol target.

| | v1 (MLP L/S) | **v2 (trend + vol)** |
|--|--------------|----------------------|
| Sharpe | **−0.47** | **1.36** |
| Max drawdown | **37.1%** | **16.9%** |
| Total return | **−32.9%** | **+167%** |
| Test window | 2022 → 2025 | 2022 → 2025 |

Repro: `cd backend` then `python train.py --config config.improved_daily.yaml --csv-path ..\results\2026-08-17_daily_v3\market_daily_us10.csv --no-persist-model`.

### Intermediate — trend only

**[`results/2026-08-17_daily_v2/`](./results/2026-08-17_daily_v2/)** — Sharpe **1.21**, MDD **23.8%**, return **+189%**.

### Baseline v1 — original MLP long/short

**[`results/2026-07-18_daily_v1/`](./results/2026-07-18_daily_v1/)** — kept so the failure mode stays visible.

| Sharpe | Max drawdown | Directional hit | Test window |
|--------|--------------|-----------------|-------------|
| **−0.47** | **37.1%** | **51.1%** | 2022 → 2025 |

Near-random next-day classification traded long/short into a mega-cap bull — fees and shorts dominate. Full writeup: [`results/README.md`](./results/README.md).

## Reproducible research

```powershell
cd backend
.\.venv\Scripts\python.exe train.py --config config.yaml
# baseline daily MLP long/short (v1):
.\.venv\Scripts\python.exe train.py --config config.results_daily.yaml --no-persist-model
# public v2 — trend + vol target:
.\.venv\Scripts\python.exe train.py --config config.improved_daily.yaml --csv-path ..\results\2026-08-17_daily_v3\market_daily_us10.csv --no-persist-model
```

Outputs land in `backend/artifacts/latest/` (gitignored). Public copies of measured runs live under [`results/`](./results/).

| File | Description |
|------|-------------|
| `report.json` | Full run summary + metrics |
| `predictions.csv` | Walk-forward model predictions |
| `trade_log.csv` | Simulated trades |
| `equity_curve.csv` | Strategy equity |
| `benchmark_curve.csv` | Buy-and-hold benchmark |
| `plots/*.html` | Equity, drawdown, trade markers |

Trigger via API:

```http
POST /api/research/run
Content-Type: application/json

{
  "config_path": "config.yaml",
  "initial_capital": 100000,
  "position_fraction": 0.10,
  "brokerage_fee_bps": 10,
  "slippage_bps": 5
}
```

### CSV data format

For full historical intraday backtests (beyond Yahoo's ~60-day 5m limit):

```text
timestamp,symbol,open,high,low,close,volume
```

`open`, `high`, `low`, and `volume` are optional; missing fields are inferred from `close`.

Set in `backend/config.yaml`:

```yaml
data:
  csv_path: "path/to/intraday.csv"
```

## Configuration

Edit `backend/config.yaml` for symbols, walk-forward windows, model hyperparameters, and backtest costs.

Environment variables (`backend/.env`):

```env
APP_HOST=127.0.0.1
APP_PORT=8020
ENABLE_SELF_LEARNING=true
SELF_LEARNING_REFRESH_MINUTES=20
SELF_LEARNING_RETRAIN_HOURS=24
ALPACA_KEY_ID=
ALPACA_SECRET_KEY=
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

## API overview

| Endpoint | Description |
|----------|-------------|
| `GET /` | Trading dashboard UI |
| `GET /api/health` | Health check |
| `GET /api/candidates/split` | Top long + short candidates |
| `GET /api/chart/{symbol}` | Candlestick + indicators |
| `GET /api/news` | Global RSS news feed |
| `POST /api/research/run` | Run full research pipeline |
| `GET /api/research/latest` | Latest research report |
| `POST /api/train` | Background model retrain |
| `POST /api/order` | Paper or live order |

## Data limitations

- Free Yahoo Finance **5m** history is limited to roughly **60 days**. The research pipeline auto-adjusts train/test windows to available coverage and logs warnings in `report.json`.
- For strict multi-year intraday validation, supply your own CSV.
- India live execution uses paper fallback unless a broker API is integrated.
- This is a **decision-support** system, not a profit guarantee. Use strict risk controls before live deployment.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Port 8020 already in use | `run_all.ps1` stops the old process automatically; or `Stop-Process` the PID on that port |
| `run_all.ps1` not found in `backend` | Use `..\run_all.ps1` or `backend\run_all.ps1` wrapper |
| Candidate API slow on first load | Wait ~30–60s for background cache warm-up, or click **Refresh Signals** |
| Research fails on all symbols | Check network/DNS; retry or provide a CSV via `-CsvPath` |

## License

MIT — see [LICENSE](LICENSE).

## Author

[Siddarthb07](https://github.com/Siddarthb07) — GeoQuant v1.0
