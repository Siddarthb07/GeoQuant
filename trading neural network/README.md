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
```

`-Reload` is optional and mainly for active development; default run avoids extra reloader subprocess noise on Windows.

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
- India live execution is left in paper fallback unless broker-specific authenticated APIs are added.
- This is a decision-support system, not a profit guarantee. Use strict risk controls before live deployment.
