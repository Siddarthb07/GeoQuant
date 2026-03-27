# GeoQuant

GeoQuant is a full-stack trading intelligence platform built around FastAPI + neural classification + news-driven context scoring.

It ranks long/short opportunities, renders chart + rationale, supports paper/live order routing, and runs a background self-learning loop that resolves prediction outcomes and retrains periodically.

## Core Capabilities

- Multi-market universe (US and India ticker sets)
- Daily and intraday direction models
- News ingestion and sentiment/relevance scoring
- Top-ranked trade candidates with:
  - directional probability
  - confidence
  - expected profit proxy
  - stop/target/risk-reward
  - expected accuracy and reasoning text
- Split view for Top Long and Top Short candidates
- Plotly candlestick chart panel with SMA20/SMA50/VWAP overlays
- Trade ticket + order history + PnL tracker
- Paper trading by default, optional Alpaca live routing for US
- Self-learning background thread:
  - resolves prior prediction outcomes
  - computes rolling hit-rate
  - triggers periodic retraining

## Runtime Architecture

### Backend

- Framework: FastAPI (`backend/app/main.py`)
- API routes: `backend/app/routers/api.py`
- Data store: SQLite (`trading.db`) for orders and signal outcomes
- Startup behavior:
  - init DB tables
  - start self-learning loop
  - auto-train models if missing

### Model Training

`backend/app/services/train_service.py`:

- Downloads market data with `yfinance`
- Pulls global RSS news and converts to daily sentiment features
- Engineers technical features (`feature_engineering.py`)
- Trains MLP classifiers (`neural_model.py`) for:
  - daily horizon
  - intraday horizon
- Saves model/scaler/meta bundles in `models_store/`

### Signal Engine

`backend/app/services/signal_service.py`:

- Loads bundles and latest feature rows per symbol
- Combines daily/intraday probabilities into blended score
- Generates direction, trade levels, risk metrics, and rationale
- Persists prediction snapshots for later feedback resolution
- Has multi-layer fallback modes:
  - heuristic mode when model/feed is incomplete
  - deterministic synthetic mode when live feed fails

### Self-Learning

`backend/app/services/self_learning.py`:

- Pulls unresolved predictions from DB
- Checks next-day market move to mark correctness
- Updates rolling hit-rate
- Retrains when retrain-hour threshold is reached

### Broker Routing

`backend/app/services/broker_service.py`:

- Paper execution for all markets
- Live US execution via Alpaca if credentials are configured
- India live path intentionally falls back to paper until broker API is integrated

## Frontend UI

Single-page dashboard from:

- `backend/app/templates/index.html`
- `backend/app/static/app.js`
- `backend/app/static/style.css`

Panels:

- Global News Flow + "Most Affected" stocks
- Chart and trade rationale center panel
- Top 15 Long + Top 15 Short ranking
- Trade ticket + order list + portfolio PnL

## API Surface (Major Endpoints)

Under `/api`:

- `GET /health`
- `GET /news`
- `GET /news-impact`
- `GET /candidates`
- `GET /candidates/split`
- `GET /chart/{symbol}`
- `POST /train`
- `GET /train/status`
- `POST /reload-models`
- `POST /order`
- `GET /orders`
- `GET /portfolio`
- `GET /model-metrics`
- `GET /self-learning/status`

## Setup

### Quick start (recommended)

From `trading neural network/` inside this repo:

```powershell
.\run_all.ps1
```

If script execution is blocked on Windows:

```powershell
.\run_all.bat
```

### Manual start

```bash
cd "trading neural network/backend"
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --port 8020
```

Open: `http://127.0.0.1:8020`

## Optional Environment Variables

Set in `backend/.env`:

- `ENABLE_SELF_LEARNING=true`
- `SELF_LEARNING_REFRESH_MINUTES=20`
- `SELF_LEARNING_RETRAIN_HOURS=24`
- `ALPACA_KEY_ID=...`
- `ALPACA_SECRET_KEY=...`
- `ALPACA_BASE_URL=https://paper-api.alpaca.markets`

## Important Operational Notes

- Intraday market data from free feeds is exchange-delayed and sometimes sparse.
- The system already includes fallback logic, but generated opportunities should still be treated as decision support.
- This project does not guarantee profitability and should be used with strict risk controls.

