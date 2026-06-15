from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict

import requests

from app.core.config import settings
from app.core.database import save_order
from app.data.market_data import latest_price


def _paper_order(symbol: str, side: str, qty: float, market: str, note: str = "") -> Dict:
    fill_price = latest_price(symbol)
    order = {
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "market": market,
        "mode": "paper",
        "status": "filled" if fill_price else "queued",
        "fill_price": fill_price,
        "note": note or "Paper order execution",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    db_id = save_order(order)
    return {
        "id": db_id,
        "symbol": symbol,
        "status": order["status"],
        "mode": "paper",
        "fill_price": fill_price,
        "note": order["note"],
    }


def _alpaca_live_order(symbol: str, side: str, qty: float) -> Dict:
    if not settings.alpaca_key_id or not settings.alpaca_secret_key:
        return _paper_order(
            symbol,
            side,
            qty,
            market="US",
            note="Live requested but Alpaca credentials missing. Executed in paper mode.",
        )
    url = f"{settings.alpaca_base_url.rstrip('/')}/v2/orders"
    payload = {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "type": "market",
        "time_in_force": "day",
    }
    headers = {
        "APCA-API-KEY-ID": settings.alpaca_key_id,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
        "Content-Type": "application/json",
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=20)
    if resp.status_code >= 400:
        return _paper_order(
            symbol,
            side,
            qty,
            market="US",
            note=f"Live order failed ({resp.status_code}). Executed in paper mode.",
        )
    data = resp.json()
    order = {
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "market": "US",
        "mode": "live",
        "status": data.get("status", "accepted"),
        "broker_order_id": data.get("id"),
        "note": "Live order sent to Alpaca",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    db_id = save_order(order)
    return {
        "id": db_id,
        "symbol": symbol,
        "status": order["status"],
        "mode": "live",
        "fill_price": None,
        "note": order["note"],
    }


def place_trade(symbol: str, side: str, qty: float, market: str, live: bool = False) -> Dict:
    if not live:
        return _paper_order(symbol, side, qty, market=market)

    if market == "US":
        return _alpaca_live_order(symbol, side, qty)

    # Free public APIs do not provide authenticated India order placement.
    return _paper_order(
        symbol,
        side,
        qty,
        market=market,
        note="India live broker integration needs broker credentials/API setup. Executed in paper mode.",
    )

