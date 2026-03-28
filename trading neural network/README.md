# GeoQuant Neural Trader

Full-stack trading intelligence app that:

- Ingests **US + India** stocks from free market feeds (`yfinance`)
- Pulls free global/world news (including war/geopolitical coverage) from public RSS feeds
- Builds neural-network direction models for:
  - Next-day move probability
  - Intraday move probability (best available intraday window from free feeds)
- Ranks top **15 trade candidates** by risk-adjusted expected edge
- Shows **Top 15 Long** and **Top 15 Short** opportunities separately
- Estimates stocks most impacted by current headlines/news flow
- Runs a self-learning loop (resolved-signal feedback + scheduled retrain)
- Shows:
  - News flow (left panel)
  - Stock graph and setup details (center)
  - Ranked opportunities + order ticket + order history (right panel)
- Supports paper trading by default and optional live US routing via Alpaca keys
- Includes production-style quant research pipeline:
  - Walk-forward validation (chronological only)
  - Backtesting with position sizing + capital tracking
  - Realistic transaction costs (brokerage + slippage)
  - Strategy performance + benchmark comparison
  - Classification quality vs trading outcomes
  - Plot artifacts (equity, drawdown, trade markers)

## Quick start

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --port 8020
```

Open: `http://127.0.0.1:8020`

## One-command launcher

From project root:

```powershell
.\run_all.ps1
```

By default this now runs the full reproducible research pipeline first (`backend/train.py`) and then starts the API/UI.

If PowerShell blocks script execution, use:

```powershell
.\run_all.bat
```

If you see a Python error, install Python 3.10+ and reopen terminal.

Optional:

```powershell
.\run_all.ps1 -BindHost 0.0.0.0 -Port 8020
.\run_all.ps1 -SkipInstall
.\run_all.ps1 -Reload
.\run_all.ps1 -InitialCapital 150000 -PositionFraction 0.12 -BrokerageFeeBps 8 -SlippageBps 6
.\run_all.ps1 -CsvPath "C:\data\geoquant_5m.csv"
.\run_all.ps1 -SkipResearch
```

`-Reload` is optional and mainly for active development; default run avoids extra reloader subprocess noise on Windows.
`-SkipResearch` starts backend/UI without running the walk-forward + backtest pipeline.

## Reproducible quant run

You can run the full quant pipeline directly:

```powershell
cd backend
.venv\Scripts\python.exe train.py --config config.yaml
```

Artifacts are generated in `backend/artifacts/latest/`:

- `report.json`
- `predictions.csv`
- `trade_log.csv`
- `equity_curve.csv`
- `benchmark_curve.csv`
- `plots/equity_curve.html`
- `plots/drawdown_curve.html`
- `plots/trade_markers.html`

FastAPI trigger (optional):

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

## Expected market-data format (CSV)

`timestamp,symbol,open,high,low,close,volume`

`open/high/low/volume` are optional; if missing, pipeline auto-fills from `close` (or zero volume).

## Smoke test

```powershell
cd backend
.venv\Scripts\python.exe tests\smoke_backtest_sample.py
```

Expected output:

```text
smoke_backtest_sample: PASS
```

## Environment variables (optional for live US orders)

Create `backend/.env`:

```env
ENABLE_SELF_LEARNING=true
SELF_LEARNING_REFRESH_MINUTES=20
SELF_LEARNING_RETRAIN_HOURS=24
ALPACA_KEY_ID=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

## Notes

- Historical training window is configured for `2000-01-01` through `2025-12-31`.
- Free sources do not provide full-depth institutional data (order book, full newswire history, alt-data, etc.).
- Free intraday feeds (5m) have historical limits. For strict 2000-2025 intraday validation, provide your own CSV data.
- India live execution is left in paper fallback unless broker-specific authenticated APIs are added.
- This is a decision-support system, not a profit guarantee. Use strict risk controls before live deployment.
