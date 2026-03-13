from __future__ import annotations

import os
from typing import Any, Dict

from flask import Flask, jsonify, redirect, render_template, request
import psycopg


app = Flask(__name__)


def get_db_connection() -> psycopg.Connection:
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")
    return psycopg.connect(database_url)


def init_db() -> None:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS bot_control_state (
                    id INTEGER PRIMARY KEY,
                    trading_enabled BOOLEAN NOT NULL,
                    new_entries_enabled BOOLEAN NOT NULL,
                    scale_ins_enabled BOOLEAN NOT NULL,
                    prematch_window_hours DOUBLE PRECISION NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            cur.execute(
                """
                INSERT INTO bot_control_state (
                    id,
                    trading_enabled,
                    new_entries_enabled,
                    scale_ins_enabled,
                    prematch_window_hours
                )
                VALUES (1, TRUE, TRUE, TRUE, 24.0)
                ON CONFLICT (id) DO NOTHING;
                """
            )
        conn.commit()


def get_control_state() -> Dict[str, Any]:
    init_db()
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    trading_enabled,
                    new_entries_enabled,
                    scale_ins_enabled,
                    prematch_window_hours,
                    updated_at
                FROM bot_control_state
                WHERE id = 1;
                """
            )
            row = cur.fetchone()
    if row is None:
        raise RuntimeError("bot_control_state row with id=1 is missing")
    return {
        "id": row[0],
        "trading_enabled": bool(row[1]),
        "new_entries_enabled": bool(row[2]),
        "scale_ins_enabled": bool(row[3]),
        "prematch_window_hours": float(row[4]),
        "updated_at": row[5].isoformat() if hasattr(row[5], "isoformat") else str(row[5]),
    }


@app.get("/")
def index() -> str:
    return render_template("index.html", controls=get_control_state())


@app.post("/update-controls")
def update_controls():
    trading_enabled = "trading_enabled" in request.form
    new_entries_enabled = "new_entries_enabled" in request.form
    scale_ins_enabled = "scale_ins_enabled" in request.form
    try:
        prematch_window_hours = float(request.form.get("prematch_window_hours", "24"))
    except (TypeError, ValueError):
        prematch_window_hours = 24.0

    init_db()
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE bot_control_state
                SET
                    trading_enabled = %s,
                    new_entries_enabled = %s,
                    scale_ins_enabled = %s,
                    prematch_window_hours = %s,
                    updated_at = NOW()
                WHERE id = 1;
                """,
                (
                    trading_enabled,
                    new_entries_enabled,
                    scale_ins_enabled,
                    prematch_window_hours,
                ),
            )
        conn.commit()
    return redirect("/")


@app.get("/healthz")
def healthz():
    return jsonify({"ok": True})


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
