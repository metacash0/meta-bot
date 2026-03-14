from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timezone
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
RANKED_CANDIDATES_HISTORY_PATH = BASE_DIR / "data" / "ranked_candidates_history.jsonl"
MAX_TOTAL_OPEN_NOTIONAL = 300.0
MAX_PER_FIXTURE_NOTIONAL = 200.0
MAX_OPEN_POSITIONS = 5
MAX_PER_LEAGUE_NOTIONAL = 300.0
DAILY_REALIZED_LOSS_STOP = -100.0


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


def extract_latest_ranked_candidates(rows: Iterable[dict]) -> dict:
    latest: dict = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("event_type", "") or "") != "ranked_candidates":
            continue
        candidates = row.get("candidates", [])
        if not isinstance(candidates, list):
            candidates = []
        latest = {
            "timestamp": row.get("timestamp"),
            "candidates": [item for item in candidates if isinstance(item, dict)],
        }
    return latest


def format_bucket_counts(bucket_counts: Any) -> List[dict]:
    if not isinstance(bucket_counts, dict):
        return []
    label_map = {
        "0_1h": "0-1h",
        "1_3h": "1-3h",
        "3_6h": "3-6h",
        "6_24h": "6-24h",
    }
    rows: List[dict] = []
    for key in ("0_1h", "1_3h", "3_6h", "6_24h"):
        try:
            count = int(bucket_counts.get(key, 0) or 0)
        except (TypeError, ValueError):
            count = 0
        rows.append({"label": label_map[key], "count": count})
    return rows


def format_scan_timestamp(value: Any) -> str:
    dt = _parse_utc_datetime(value)
    if dt is None:
        return str(value or "")
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def build_latest_scan_cards(scan_summary: dict) -> List[dict]:
    if not isinstance(scan_summary, dict):
        return []
    cards: List[dict] = []
    for key in (
        "fixtures_eligible",
        "fixtures_scanned",
        "live_fixtures_scanned",
        "signals_found",
        "research_signals_found",
        "prematch_window_hours",
        "trading_enabled",
        "new_entries_enabled",
        "scale_ins_enabled",
        "include_live_fixtures",
    ):
        if key in scan_summary:
            cards.append({"label": key, "value": scan_summary.get(key)})
    return cards


def _decision_edge(row: dict) -> float | None:
    side = str(row.get("side", "") or "").upper()
    try:
        if "edge" in row and row.get("edge") is not None:
            return float(row.get("edge"))
        if side == "YES" and row.get("yes_edge") is not None:
            return float(row.get("yes_edge"))
        if side == "NO" and row.get("no_edge") is not None:
            return float(row.get("no_edge"))
    except (TypeError, ValueError):
        return None
    return None


def load_decision_audit(signal_event_rows: Iterable[dict], limit: int = 10) -> List[dict]:
    decisions: List[dict] = []
    for row in signal_event_rows:
        if not isinstance(row, dict) or str(row.get("event_type", "") or "") != "buy_signal":
            continue
        data = row.get("data", {})
        if not isinstance(data, dict):
            continue
        if data.get("executed") is not True:
            continue
        decisions.append(
            {
                "timestamp": data.get("timestamp") or row.get("timestamp"),
                "league": data.get("league"),
                "market_name": data.get("market_name"),
                "fixture_id": data.get("fixture_id"),
                "side": data.get("side"),
                "execution_reason": data.get("execution_reason"),
                "entry_mode": data.get("entry_mode"),
                "edge": _decision_edge(data),
                "rank_at_decision": data.get("rank_at_decision"),
                "was_top_ranked": data.get("was_top_ranked"),
                "candidate_count": data.get("candidate_count"),
                "priority_score": data.get("priority_score"),
                "recommended_notional": data.get("recommended_notional"),
                "ask_price": data.get("ask_price"),
                "spread": data.get("spread"),
                "minute": data.get("minute"),
                "status": data.get("status"),
                "minute_bucket": data.get("minute_bucket"),
                "projected_open_total_notional_after": data.get("projected_open_total_notional_after"),
            }
        )
    decisions.sort(
        key=lambda row: _parse_utc_datetime(row.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return decisions[:limit]


def build_signal_quality_summary(
    signal_event_rows: Iterable[dict],
    paper_trade_rows: Iterable[dict],
) -> dict:
    today_utc = datetime.now(timezone.utc).date()
    executed_entry_count = 0
    blocked_by_risk = 0
    blocked_by_liquidity = 0
    blocked_by_below_min_notional = 0
    blocked_by_scale_in_guard = 0
    executed_edges: List[float] = []
    executed_priority_scores: List[float] = []

    for row in paper_trade_rows:
        if not isinstance(row, dict):
            continue
        event_dt = _parse_utc_datetime(row.get("timestamp"))
        if event_dt is None or event_dt.date() != today_utc:
            continue
        executed_entry_count += 1
        try:
            executed_edges.append(float(row.get("entry_edge", 0.0) or 0.0))
        except (TypeError, ValueError):
            pass
        try:
            executed_priority_scores.append(float(row.get("priority_score", 0.0) or 0.0))
        except (TypeError, ValueError):
            pass

    for row in signal_event_rows:
        if not isinstance(row, dict) or str(row.get("event_type", "") or "") != "buy_signal":
            continue
        event_dt = _parse_utc_datetime(row.get("timestamp"))
        if event_dt is None or event_dt.date() != today_utc:
            continue
        data = row.get("data", {})
        if not isinstance(data, dict):
            continue
        execution_reason = str(data.get("execution_reason", "") or "")
        sizing_reason = str(data.get("sizing_reason", "") or "")
        if execution_reason.startswith("risk_limit_"):
            blocked_by_risk += 1
        if execution_reason == "scale_in_edge_not_improved":
            blocked_by_scale_in_guard += 1
        if execution_reason == "below_min_notional" or sizing_reason == "below_min_notional":
            blocked_by_below_min_notional += 1
        if execution_reason in {"missing_price", "missing_size"} or sizing_reason in {"missing_price", "missing_size"}:
            blocked_by_liquidity += 1

    return {
        "today_executed_entry_count": executed_entry_count,
        "today_blocked_by_risk_count": blocked_by_risk,
        "today_blocked_by_liquidity_count": blocked_by_liquidity,
        "today_blocked_by_below_min_notional_count": blocked_by_below_min_notional,
        "today_blocked_by_scale_in_guard_count": blocked_by_scale_in_guard,
        "avg_edge_executed_entries_today": (
            sum(executed_edges) / len(executed_edges) if executed_edges else 0.0
        ),
        "avg_priority_score_executed_entries_today": (
            sum(executed_priority_scores) / len(executed_priority_scores)
            if executed_priority_scores
            else 0.0
        ),
    }


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


def _parse_utc_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def build_risk_snapshot(positions_payload: dict, settlement_rows: Iterable[dict]) -> dict:
    positions = positions_payload.get("positions", []) if isinstance(positions_payload, dict) else []
    open_positions = 0
    open_total_notional = 0.0
    per_fixture_notional: Dict[str, float] = {}
    per_league_notional: Dict[str, float] = {}

    for row in positions:
        if not isinstance(row, dict) or str(row.get("status", "") or "") != "open":
            continue
        open_positions += 1
        try:
            total_notional = float(
                row.get("position_total_notional", row.get("total_notional", 0.0)) or 0.0
            )
        except (TypeError, ValueError):
            total_notional = 0.0
        open_total_notional += total_notional

        fixture_key = str(row.get("fixture_id", "") or "")
        if fixture_key:
            per_fixture_notional[fixture_key] = per_fixture_notional.get(fixture_key, 0.0) + total_notional

        league_key = str(row.get("league", "") or "")
        if league_key:
            per_league_notional[league_key] = per_league_notional.get(league_key, 0.0) + total_notional

    today_utc = datetime.now(timezone.utc).date()
    today_realized_pnl = 0.0
    for row in settlement_rows:
        if not isinstance(row, dict):
            continue
        event_dt = _parse_utc_datetime(row.get("closed_at")) or _parse_utc_datetime(row.get("timestamp"))
        if event_dt is None or event_dt.date() != today_utc:
            continue
        try:
            today_realized_pnl += float(row.get("gross_pnl", 0.0) or 0.0)
        except (TypeError, ValueError):
            pass

    return {
        "open_positions": open_positions,
        "open_total_notional": open_total_notional,
        "per_fixture_notional": per_fixture_notional,
        "per_league_notional": per_league_notional,
        "today_realized_pnl": today_realized_pnl,
    }


def build_risk_summary(risk_snapshot: dict) -> dict:
    try:
        open_total_notional = float(risk_snapshot.get("open_total_notional", 0.0) or 0.0)
    except (TypeError, ValueError):
        open_total_notional = 0.0
    try:
        open_positions = int(risk_snapshot.get("open_positions", 0) or 0)
    except (TypeError, ValueError):
        open_positions = 0
    try:
        today_realized_pnl = float(risk_snapshot.get("today_realized_pnl", 0.0) or 0.0)
    except (TypeError, ValueError):
        today_realized_pnl = 0.0

    return {
        "max_total_open_notional": MAX_TOTAL_OPEN_NOTIONAL,
        "max_per_fixture_notional": MAX_PER_FIXTURE_NOTIONAL,
        "max_open_positions": MAX_OPEN_POSITIONS,
        "max_per_league_notional": MAX_PER_LEAGUE_NOTIONAL,
        "daily_realized_loss_stop": DAILY_REALIZED_LOSS_STOP,
        "open_positions": open_positions,
        "open_total_notional": open_total_notional,
        "remaining_total_notional": max(0.0, MAX_TOTAL_OPEN_NOTIONAL - open_total_notional),
        "today_realized_pnl": today_realized_pnl,
        "remaining_loss_buffer": today_realized_pnl - DAILY_REALIZED_LOSS_STOP,
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
    signal_event_rows = read_jsonl_tail(SIGNAL_EVENTS_PATH, limit=2000)
    recent_signal_events = list(reversed(signal_event_rows[-10:]))
    ranked_candidate_rows = read_jsonl_tail(RANKED_CANDIDATES_HISTORY_PATH, limit=500)
    ranked_candidates_snapshot = extract_latest_ranked_candidates(ranked_candidate_rows)
    paper_trade_rows = read_jsonl_tail(PAPER_TRADES_PATH, limit=5000)
    all_settlement_rows = read_jsonl_tail(PAPER_SETTLEMENTS_PATH, limit=10000)
    realized_pnl_summary = sum_realized_pnl(all_settlement_rows)
    risk_snapshot = build_risk_snapshot(open_positions_payload, all_settlement_rows)
    risk_summary = build_risk_summary(risk_snapshot)
    signal_quality_summary = build_signal_quality_summary(signal_event_rows, paper_trade_rows)
    latest_scan_cards = build_latest_scan_cards(latest_scan_summary)
    latest_scan_bucket_counts = format_bucket_counts(latest_scan_summary.get("bucket_counts"))
    latest_scan_timestamp = format_scan_timestamp(latest_scan_summary.get("timestamp"))
    decision_audit_rows = load_decision_audit(signal_event_rows, limit=15)

    return render_template(
        "index.html",
        controls=controls,
        open_positions=open_positions,
        open_positions_summary=open_positions_summary,
        latest_scan_summary=latest_scan_summary,
        recent_paper_trades=recent_paper_trades,
        recent_paper_settlements=recent_paper_settlements,
        recent_signal_events=recent_signal_events,
        ranked_candidates_snapshot=ranked_candidates_snapshot,
        realized_pnl_summary=realized_pnl_summary,
        risk_summary=risk_summary,
        signal_quality_summary=signal_quality_summary,
        latest_scan_cards=latest_scan_cards,
        latest_scan_bucket_counts=latest_scan_bucket_counts,
        latest_scan_timestamp=latest_scan_timestamp,
        decision_audit_rows=decision_audit_rows,
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
