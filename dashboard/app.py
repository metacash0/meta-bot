from __future__ import annotations

import json
import os
import secrets
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Iterable, List

from flask import Flask, Response, jsonify, redirect, render_template, request
import psycopg


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
OPEN_POSITIONS_PATH = BASE_DIR / "data" / "paper_open_positions.json"
SCAN_SUMMARY_PATH = BASE_DIR / "data" / "logs" / "scan_summary.jsonl"
SIGNAL_EVENTS_PATH = BASE_DIR / "data" / "logs" / "signal_events.jsonl"
PAPER_TRADES_PATH = BASE_DIR / "data" / "logs" / "paper_trades.jsonl"
PAPER_SETTLEMENTS_PATH = BASE_DIR / "data" / "logs" / "paper_settlements.jsonl"


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


def read_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def read_jsonl_tail(path: Path, limit: int = 10) -> List[dict]:
    if not path.exists():
        return []
    rows: List[dict] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except Exception:
        return []
    if limit <= 0:
        return []
    return rows[-limit:]


def sum_open_positions(positions_payload: dict) -> dict:
    positions = positions_payload.get("positions", []) if isinstance(positions_payload, dict) else []
    open_positions = 0
    open_yes_positions = 0
    open_no_positions = 0
    open_total_notional = 0.0
    open_yes_notional = 0.0
    open_no_notional = 0.0

    for row in positions:
        if not isinstance(row, dict) or str(row.get("status", "") or "") != "open":
            continue
        open_positions += 1
        side = str(row.get("side", "") or "").upper()
        try:
            total_notional = float(row.get("total_notional", 0.0) or 0.0)
        except (TypeError, ValueError):
            total_notional = 0.0
        open_total_notional += total_notional
        if side == "YES":
            open_yes_positions += 1
            open_yes_notional += total_notional
        elif side == "NO":
            open_no_positions += 1
            open_no_notional += total_notional

    return {
        "open_positions": open_positions,
        "open_yes_positions": open_yes_positions,
        "open_no_positions": open_no_positions,
        "open_total_notional": open_total_notional,
        "open_yes_notional": open_yes_notional,
        "open_no_notional": open_no_notional,
    }


def sum_realized_pnl(settlement_rows: Iterable[dict]) -> dict:
    settled_trades = 0
    realized_gross_pnl = 0.0
    roi_values: List[float] = []

    for row in settlement_rows:
        if not isinstance(row, dict):
            continue
        settled_trades += 1
        try:
            realized_gross_pnl += float(row.get("gross_pnl", 0.0) or 0.0)
        except (TypeError, ValueError):
            pass
        try:
            roi_values.append(float(row.get("roi", 0.0) or 0.0))
        except (TypeError, ValueError):
            pass

    return {
        "settled_trades": settled_trades,
        "realized_gross_pnl": realized_gross_pnl,
        "avg_roi": (sum(roi_values) / len(roi_values)) if roi_values else 0.0,
    }


def enrich_open_positions(positions_payload: dict) -> List[dict]:
    positions = positions_payload.get("positions", []) if isinstance(positions_payload, dict) else []
    enriched: List[dict] = []

    for row in positions:
        if not isinstance(row, dict):
            continue
        row_copy = dict(row)
        unrealized_pnl = None
        try:
            avg_price = float(row_copy.get("avg_price", 0.0) or 0.0)
            shares = float(row_copy.get("total_shares", 0.0) or 0.0)
            side = str(row_copy.get("side", "") or "").upper()
            fair_yes = row_copy.get("fair_yes")
            fair_no = row_copy.get("fair_no")
            if side == "YES" and fair_yes is not None:
                mark_price = float(fair_yes)
                unrealized_pnl = shares * (mark_price - avg_price)
            elif side == "NO" and fair_no is not None:
                mark_price = float(fair_no)
                unrealized_pnl = shares * (mark_price - avg_price)
        except (TypeError, ValueError):
            unrealized_pnl = None
        row_copy["unrealized_pnl"] = unrealized_pnl
        enriched.append(row_copy)
    return enriched


def check_auth(username: str, password: str) -> bool:
    expected_username = os.getenv("DASHBOARD_USERNAME", "admin")
    expected_password = os.getenv("DASHBOARD_PASSWORD", "changeme")
    return secrets.compare_digest(username, expected_username) and secrets.compare_digest(
        password, expected_password
    )


def _unauthorized_response() -> Response:
    return Response(
        "Authentication required",
        401,
        {"WWW-Authenticate": 'Basic realm="Paper Bot Dashboard"'},
    )


def requires_auth(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        auth = request.authorization
        if auth is None or not check_auth(auth.username or "", auth.password or ""):
            return _unauthorized_response()
        return view(*args, **kwargs)

    return wrapped


@app.get("/")
@requires_auth
def index() -> str:
    controls = get_control_state()
    open_positions_payload = read_json_file(OPEN_POSITIONS_PATH, {"positions": []})
    open_positions = enrich_open_positions(open_positions_payload)
    open_positions_summary = sum_open_positions(open_positions_payload)
    latest_scan_rows = read_jsonl_tail(SCAN_SUMMARY_PATH, limit=1)
    latest_scan_summary = latest_scan_rows[-1] if latest_scan_rows else {}
    recent_paper_trades = list(reversed(read_jsonl_tail(PAPER_TRADES_PATH, limit=10)))
    recent_paper_settlements = list(reversed(read_jsonl_tail(PAPER_SETTLEMENTS_PATH, limit=10)))
    recent_signal_events = list(reversed(read_jsonl_tail(SIGNAL_EVENTS_PATH, limit=10)))
    realized_pnl_summary = sum_realized_pnl(read_jsonl_tail(PAPER_SETTLEMENTS_PATH, limit=10000))

    return render_template(
        "index.html",
        controls=controls,
        open_positions=open_positions,
        open_positions_summary=open_positions_summary,
        latest_scan_summary=latest_scan_summary,
        recent_paper_trades=recent_paper_trades,
        recent_paper_settlements=recent_paper_settlements,
        recent_signal_events=recent_signal_events,
        realized_pnl_summary=realized_pnl_summary,
    )


@app.post("/update-controls")
@requires_auth
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
