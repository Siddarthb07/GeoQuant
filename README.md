# GeoQuant Neural Trader

[![Release](https://img.shields.io/github/v/release/Siddarthb07/GeoQuant?label=v1.0)](https://github.com/Siddarthb07/GeoQuant/releases/tag/v1.0)

Full-stack trading intelligence platform for **US + India** equities. GeoQuant combines neural-network signal models, global news sentiment, walk-forward research, and a live trading dashboard in one FastAPI app.

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

First committed cost-aware run: **[`results/2026-07-18_daily_v1/`](./results/2026-07-18_daily_v1/)**.

| Sharpe | Max drawdown | Directional hit | Test window |
|--------|--------------|-----------------|-------------|
| **−0.47** | **37.1%** | **51.1%** | 2022 → 2025 |

Negative risk-adjusted return vs buy-and-hold on this basket — expected for a first public baseline with fees/slippage inside the optimizer. Full writeup: [`results/README.md`](./results/README.md). Repro: `python train.py --config config.results_daily.yaml --no-persist-model`.

## Reproducible research

```powershell
cd backend
.\.venv\Scripts\python.exe train.py --config config.yaml
# public measured daily run:
.\.venv\Scripts\python.exe train.py --config config.results_daily.yaml --no-persist-model
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
