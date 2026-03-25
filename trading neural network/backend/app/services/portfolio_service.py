from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from app.core.database import list_orders
from app.data.market_data import latest_price

PRICE_CACHE: Dict[str, Tuple[datetime, Optional[float]]] = {}


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return default


def _order_sort_key(item: Dict) -> Tuple[str, int]:
    created = str(item.get("created_at", ""))
    oid = int(_coerce_float(item.get("id", 0)))
    return created, oid


def _cached_latest_price(symbol: str, ttl_seconds: int = 45) -> Optional[float]:
    now = datetime.now(timezone.utc)
    cached = PRICE_CACHE.get(symbol)
    if cached is not None:
        ts, value = cached
        if (now - ts).total_seconds() <= ttl_seconds:
            return value
    try:
        value = latest_price(symbol)
    except Exception:  # noqa: BLE001
        value = None
    PRICE_CACHE[symbol] = (now, value)
    return value


def _symbol_metrics(symbol: str, market: str, rows: List[Dict]) -> Dict:
    long_lots: List[List[float]] = []  # [qty, price]
    short_lots: List[List[float]] = []  # [qty, price]
    realized = 0.0
    bought_qty = 0.0
    sold_qty = 0.0
    buy_value = 0.0
    sell_value = 0.0
    executed_orders = 0

    for row in rows:
        side = str(row.get("side", "")).lower().strip()
        qty = _coerce_float(row.get("qty"))
        fill = row.get("fill_price")
        fill_price = _coerce_float(fill, default=-1.0)
        if qty <= 0 or fill is None or fill_price <= 0:
            continue
        executed_orders += 1

        if side == "buy":
            bought_qty += qty
            buy_value += qty * fill_price
            qty_left = qty

            while qty_left > 1e-12 and short_lots:
                lot_qty, lot_price = short_lots[0]
                matched = min(qty_left, lot_qty)
                # For short coverage, profit if buy back lower than short sale price.
                realized += (lot_price - fill_price) * matched
                qty_left -= matched
                lot_qty -= matched
                if lot_qty <= 1e-12:
                    short_lots.pop(0)
                else:
                    short_lots[0][0] = lot_qty

            if qty_left > 1e-12:
                long_lots.append([qty_left, fill_price])

        elif side == "sell":
            sold_qty += qty
            sell_value += qty * fill_price
            qty_left = qty

            while qty_left > 1e-12 and long_lots:
                lot_qty, lot_price = long_lots[0]
                matched = min(qty_left, lot_qty)
                realized += (fill_price - lot_price) * matched
                qty_left -= matched
                lot_qty -= matched
                if lot_qty <= 1e-12:
                    long_lots.pop(0)
                else:
                    long_lots[0][0] = lot_qty

            if qty_left > 1e-12:
                short_lots.append([qty_left, fill_price])

    open_long_qty = sum(lot[0] for lot in long_lots)
    open_short_qty = sum(lot[0] for lot in short_lots)
    net_qty = open_long_qty - open_short_qty

    avg_entry_price: Optional[float] = None
    open_side = "FLAT"
    if net_qty > 1e-12:
        long_cost = sum(lot[0] * lot[1] for lot in long_lots)
        avg_entry_price = long_cost / max(open_long_qty, 1e-12)
        open_side = "LONG"
    elif net_qty < -1e-12:
        short_value = sum(lot[0] * lot[1] for lot in short_lots)
        avg_entry_price = short_value / max(open_short_qty, 1e-12)
        open_side = "SHORT"

    current_price = _cached_latest_price(symbol)

    unrealized = 0.0
    if current_price is not None and avg_entry_price is not None:
        if open_side == "LONG":
            unrealized = (current_price - avg_entry_price) * open_long_qty
        elif open_side == "SHORT":
            unrealized = (avg_entry_price - current_price) * open_short_qty

    total_pnl = realized + unrealized
    capital = max(1e-9, buy_value)
    realized_return_pct = (realized / capital) * 100.0
    total_return_pct = (total_pnl / capital) * 100.0

    return {
        "symbol": symbol,
        "market": market,
        "executed_orders": int(executed_orders),
        "total_bought_qty": round(bought_qty, 6),
        "total_sold_qty": round(sold_qty, 6),
        "net_qty": round(net_qty, 6),
        "open_side": open_side,
        "avg_entry_price": None if avg_entry_price is None else round(float(avg_entry_price), 4),
        "current_price": None if current_price is None else round(float(current_price), 4),
        "realized_pnl": round(realized, 2),
        "unrealized_pnl": round(unrealized, 2),
        "total_pnl": round(total_pnl, 2),
        "realized_return_pct": round(realized_return_pct, 2),
        "total_return_pct": round(total_return_pct, 2),
    }


def get_portfolio_performance(limit: int = 2000) -> Dict:
    rows = list_orders(limit=limit)
    if not rows:
        return {
            "summary": {
                "symbols_traded": 0,
                "open_positions": 0,
                "total_realized_pnl": 0.0,
                "total_unrealized_pnl": 0.0,
                "total_pnl": 0.0,
            },
            "items": [],
        }

    grouped: Dict[Tuple[str, str], List[Dict]] = {}
    for row in rows:
        symbol = str(row.get("symbol", "")).strip()
        market = str(row.get("market", "")).strip().upper() or "US"
        if not symbol:
            continue
        key = (symbol, market)
        grouped.setdefault(key, []).append(row)

    metrics: List[Dict] = []
    for (symbol, market), entries in grouped.items():
        metrics.append(_symbol_metrics(symbol, market, sorted(entries, key=_order_sort_key)))

    metrics.sort(key=lambda x: abs(float(x.get("total_pnl", 0.0))), reverse=True)
    open_positions = sum(1 for x in metrics if abs(float(x.get("net_qty", 0.0))) > 1e-12)
    total_realized = sum(float(x.get("realized_pnl", 0.0)) for x in metrics)
    total_unrealized = sum(float(x.get("unrealized_pnl", 0.0)) for x in metrics)

    return {
        "summary": {
            "symbols_traded": len(metrics),
            "open_positions": int(open_positions),
            "total_realized_pnl": round(total_realized, 2),
            "total_unrealized_pnl": round(total_unrealized, 2),
            "total_pnl": round(total_realized + total_unrealized, 2),
        },
        "items": metrics,
    }
