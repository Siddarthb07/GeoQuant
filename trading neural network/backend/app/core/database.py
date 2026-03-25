from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.core.config import settings


def _conn() -> sqlite3.Connection:
    connection = sqlite3.connect(settings.sqlite_path)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with _conn() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                qty REAL NOT NULL,
                market TEXT NOT NULL,
                mode TEXT NOT NULL,
                status TEXT NOT NULL,
                broker_order_id TEXT,
                fill_price REAL,
                note TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS signal_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                market TEXT NOT NULL,
                direction TEXT NOT NULL,
                prediction_date TEXT NOT NULL,
                predicted_at TEXT NOT NULL,
                reference_price REAL NOT NULL,
                predicted_up_prob REAL NOT NULL,
                confidence REAL NOT NULL,
                expected_accuracy REAL NOT NULL,
                horizon TEXT NOT NULL DEFAULT '1d',
                resolved INTEGER NOT NULL DEFAULT 0,
                actual_up INTEGER,
                was_correct INTEGER,
                resolved_at TEXT,
                UNIQUE(symbol, prediction_date, horizon, direction)
            )
            """
        )
        con.commit()


def save_order(order: Dict[str, Any]) -> int:
    with _conn() as con:
        cursor = con.execute(
            """
            INSERT INTO orders (
                symbol, side, qty, market, mode, status,
                broker_order_id, fill_price, note, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order["symbol"],
                order["side"],
                order["qty"],
                order["market"],
                order["mode"],
                order["status"],
                order.get("broker_order_id"),
                order.get("fill_price"),
                order.get("note"),
                order.get("created_at", datetime.now(timezone.utc).isoformat()),
            ),
        )
        con.commit()
        return int(cursor.lastrowid)


def list_orders(limit: int = 100) -> List[Dict[str, Any]]:
    with _conn() as con:
        rows = con.execute(
            """
            SELECT id, symbol, side, qty, market, mode, status,
                   broker_order_id, fill_price, note, created_at
            FROM orders
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def save_prediction_rows(rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0
    payload = []
    for row in rows:
        payload.append(
            (
                row["symbol"],
                row["market"],
                row["direction"],
                row["prediction_date"],
                row.get("predicted_at", datetime.now(timezone.utc).isoformat()),
                row["reference_price"],
                row["predicted_up_prob"],
                row["confidence"],
                row["expected_accuracy"],
                row.get("horizon", "1d"),
            )
        )
    with _conn() as con:
        before = con.total_changes
        con.executemany(
            """
            INSERT OR IGNORE INTO signal_predictions (
                symbol, market, direction, prediction_date, predicted_at,
                reference_price, predicted_up_prob, confidence, expected_accuracy, horizon
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        con.commit()
        return int(con.total_changes - before)


def list_unresolved_predictions(cutoff_date: str, limit: int = 500) -> List[Dict[str, Any]]:
    with _conn() as con:
        rows = con.execute(
            """
            SELECT id, symbol, market, direction, prediction_date, predicted_at,
                   reference_price, predicted_up_prob, confidence, expected_accuracy, horizon
            FROM signal_predictions
            WHERE resolved = 0
              AND prediction_date <= ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (cutoff_date, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def mark_prediction_resolved(
    prediction_id: int,
    actual_up: int,
    was_correct: int,
    resolved_at: Optional[str] = None,
) -> None:
    with _conn() as con:
        con.execute(
            """
            UPDATE signal_predictions
            SET resolved = 1,
                actual_up = ?,
                was_correct = ?,
                resolved_at = ?
            WHERE id = ?
            """,
            (
                int(actual_up),
                int(was_correct),
                resolved_at or datetime.now(timezone.utc).isoformat(),
                int(prediction_id),
            ),
        )
        con.commit()


def prediction_accuracy_stats(window: int = 300) -> Dict[str, float]:
    with _conn() as con:
        row = con.execute(
            """
            SELECT COUNT(*) AS n,
                   AVG(CASE WHEN was_correct IS NULL THEN NULL ELSE was_correct END) AS hit_rate
            FROM (
                SELECT was_correct
                FROM signal_predictions
                WHERE resolved = 1
                ORDER BY id DESC
                LIMIT ?
            )
            """,
            (window,),
        ).fetchone()
    n = int(row["n"] or 0)
    hit = float(row["hit_rate"]) if row["hit_rate"] is not None else 0.5
    return {"sample_size": n, "hit_rate": hit}
