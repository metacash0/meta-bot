from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    import psycopg  # type: ignore
except Exception:
    psycopg = None

from binary_bot.paper_executor import (
    find_open_position,
    maybe_execute_paper_trade,
    read_open_positions,
)
from binary_bot.paper_settlement import settle_open_positions
from shared.market_signal_snapshot import build_market_signal_snapshot
from shared.order_sizing import size_from_signal_snapshot


FIXTURE_MAPPING_INDEX_PATH = "data/fixture_mapping_index.json"
MARKET_MAP_PATH = "data/market_map.json"
SCAN_LOG_PATH = "data/logs/scan_summary.jsonl"
SIGNAL_LOG_PATH = "data/logs/signal_events.jsonl"
SCAN_SLEEP_SEC = 10.0
DEFAULT_BANKROLL = 5000.0
RESEARCH_EFFECTIVE_MIN_EDGE = 0.025
INCLUDE_LIVE_FIXTURES = True
TRADING_ENABLED = os.getenv("TRADING_ENABLED", "true").lower() == "true"
NEW_ENTRIES_ENABLED = os.getenv("NEW_ENTRIES_ENABLED", "true").lower() == "true"
SCALE_INS_ENABLED = os.getenv("SCALE_INS_ENABLED", "true").lower() == "true"
PREMATCH_WINDOW_HOURS = float(os.getenv("PREMATCH_WINDOW_HOURS", "6"))
DASHBOARD_DATABASE_URL = os.getenv("DASHBOARD_DATABASE_URL", "").strip()
OPEN_POSITIONS_PATH = "data/paper_open_positions.json"
PAPER_SETTLEMENTS_PATH = "data/logs/paper_settlements.jsonl"
MAX_TOTAL_OPEN_NOTIONAL = 300.0
MAX_PER_FIXTURE_NOTIONAL = 200.0
MAX_OPEN_POSITIONS = 5
MAX_PER_LEAGUE_NOTIONAL = 300.0
DAILY_REALIZED_LOSS_STOP = -100.0

os.makedirs("data/logs", exist_ok=True)


def _read_json_rows(path: str, key: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []

    rows = payload.get(key, []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def append_jsonl(path: str, payload: dict) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
    except Exception:
        pass


def load_open_positions_payload() -> dict:
    if not os.path.exists(OPEN_POSITIONS_PATH):
        return {"positions": []}
    try:
        with open(OPEN_POSITIONS_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {"positions": []}
    if not isinstance(payload, dict):
        return {"positions": []}
    positions = payload.get("positions")
    if not isinstance(positions, list):
        return {"positions": []}
    return {"positions": [row for row in positions if isinstance(row, dict)]}


def _parse_utc_date(value: Any) -> Optional[datetime]:
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


def load_today_realized_pnl() -> float:
    if not os.path.exists(PAPER_SETTLEMENTS_PATH):
        return 0.0
    today_utc = datetime.now(timezone.utc).date()
    total = 0.0
    try:
        with open(PAPER_SETTLEMENTS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except Exception:
                    continue
                if not isinstance(row, dict):
                    continue
                event_dt = _parse_utc_date(row.get("closed_at")) or _parse_utc_date(row.get("timestamp"))
                if event_dt is None or event_dt.date() != today_utc:
                    continue
                try:
                    total += float(row.get("gross_pnl", 0.0) or 0.0)
                except (TypeError, ValueError):
                    pass
    except Exception:
        return 0.0
    return total


def build_risk_snapshot() -> dict:
    open_positions_payload = load_open_positions_payload()
    positions = open_positions_payload.get("positions", [])
    open_positions = 0
    open_total_notional = 0.0
    per_fixture_notional: Dict[str, float] = {}
    per_league_notional: Dict[str, float] = {}

    for row in positions:
        if not isinstance(row, dict):
            continue
        if str(row.get("status", "") or "") != "open":
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

    return {
        "open_positions": open_positions,
        "open_total_notional": open_total_notional,
        "per_fixture_notional": per_fixture_notional,
        "per_league_notional": per_league_notional,
        "today_realized_pnl": load_today_realized_pnl(),
        "_open_positions_payload": open_positions_payload,
    }


def check_risk_limits(signal_row: dict, risk_snapshot: dict) -> str | None:
    try:
        today_realized_pnl = float(risk_snapshot.get("today_realized_pnl", 0.0) or 0.0)
    except (TypeError, ValueError):
        today_realized_pnl = 0.0
    if today_realized_pnl <= DAILY_REALIZED_LOSS_STOP:
        return "risk_limit_daily_loss_stop"

    try:
        proposed_notional = float(signal_row.get("recommended_notional", 0.0) or 0.0)
    except (TypeError, ValueError):
        proposed_notional = 0.0
    if proposed_notional <= 0.0:
        return None

    open_positions_payload = risk_snapshot.get("_open_positions_payload", {"positions": []})
    existing_position = find_open_position(
        open_positions_payload,
        int(signal_row.get("fixture_id")),
        str(signal_row.get("side") or ""),
    )
    if existing_position is None:
        try:
            open_positions = int(risk_snapshot.get("open_positions", 0) or 0)
        except (TypeError, ValueError):
            open_positions = 0
        if open_positions >= MAX_OPEN_POSITIONS:
            return "risk_limit_max_open_positions"

    try:
        open_total_notional = float(risk_snapshot.get("open_total_notional", 0.0) or 0.0)
    except (TypeError, ValueError):
        open_total_notional = 0.0
    if open_total_notional + proposed_notional > MAX_TOTAL_OPEN_NOTIONAL:
        return "risk_limit_total_open_notional"

    fixture_key = str(signal_row.get("fixture_id", "") or "")
    fixture_notional = 0.0
    try:
        fixture_notional = float(risk_snapshot.get("per_fixture_notional", {}).get(fixture_key, 0.0) or 0.0)
    except (TypeError, ValueError, AttributeError):
        fixture_notional = 0.0
    if fixture_notional + proposed_notional > MAX_PER_FIXTURE_NOTIONAL:
        return "risk_limit_fixture_notional"

    league_key = str(signal_row.get("league", "") or "")
    league_notional = 0.0
    try:
        league_notional = float(risk_snapshot.get("per_league_notional", {}).get(league_key, 0.0) or 0.0)
    except (TypeError, ValueError, AttributeError):
        league_notional = 0.0
    if league_notional + proposed_notional > MAX_PER_LEAGUE_NOTIONAL:
        return "risk_limit_league_notional"

    return None


def clone_risk_snapshot(risk_snapshot: dict) -> dict:
    open_payload = risk_snapshot.get("_open_positions_payload", {"positions": []})
    positions = open_payload.get("positions", []) if isinstance(open_payload, dict) else []
    return {
        "open_positions": int(risk_snapshot.get("open_positions", 0) or 0),
        "open_total_notional": float(risk_snapshot.get("open_total_notional", 0.0) or 0.0),
        "per_fixture_notional": dict(risk_snapshot.get("per_fixture_notional", {}) or {}),
        "per_league_notional": dict(risk_snapshot.get("per_league_notional", {}) or {}),
        "today_realized_pnl": float(risk_snapshot.get("today_realized_pnl", 0.0) or 0.0),
        "_open_positions_payload": {
            "positions": [dict(row) for row in positions if isinstance(row, dict)],
        },
    }


def apply_execution_to_risk_snapshot(
    risk_snapshot: dict,
    signal_row: dict,
    proposed_notional: float,
    is_new_position: bool,
) -> None:
    try:
        notional = float(proposed_notional or 0.0)
    except (TypeError, ValueError):
        notional = 0.0
    if notional <= 0.0:
        return

    risk_snapshot["open_total_notional"] = float(risk_snapshot.get("open_total_notional", 0.0) or 0.0) + notional
    if is_new_position:
        risk_snapshot["open_positions"] = int(risk_snapshot.get("open_positions", 0) or 0) + 1

    fixture_key = str(signal_row.get("fixture_id", "") or "")
    per_fixture_notional = risk_snapshot.setdefault("per_fixture_notional", {})
    if fixture_key:
        per_fixture_notional[fixture_key] = float(per_fixture_notional.get(fixture_key, 0.0) or 0.0) + notional

    league_key = str(signal_row.get("league", "") or "")
    per_league_notional = risk_snapshot.setdefault("per_league_notional", {})
    if league_key:
        per_league_notional[league_key] = float(per_league_notional.get(league_key, 0.0) or 0.0) + notional

    open_payload = risk_snapshot.setdefault("_open_positions_payload", {"positions": []})
    positions = open_payload.setdefault("positions", [])
    existing_position = find_open_position(
        open_payload,
        int(signal_row.get("fixture_id")),
        str(signal_row.get("side") or ""),
    )
    if existing_position is None and is_new_position:
        positions.append(
            {
                "fixture_id": signal_row.get("fixture_id"),
                "market_name": signal_row.get("market_name"),
                "league": signal_row.get("league"),
                "side": signal_row.get("side"),
                "status": "open",
                "position_total_notional": notional,
                "total_notional": notional,
            }
        )
    elif existing_position is not None:
        updated_total = float(
            existing_position.get("position_total_notional", existing_position.get("total_notional", 0.0)) or 0.0
        ) + notional
        existing_position["position_total_notional"] = updated_total
        existing_position["total_notional"] = updated_total


def compute_priority_score(signal_row: dict) -> float:
    side = str(signal_row.get("side", "") or "").upper()
    if side == "YES":
        edge_value = signal_row.get("yes_edge")
    elif side == "NO":
        edge_value = signal_row.get("no_edge")
    else:
        return 0.0

    try:
        edge = float(edge_value or 0.0)
        recommended_notional = float(signal_row.get("recommended_notional", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if edge <= 0.0 or recommended_notional <= 0.0:
        return 0.0
    return edge * recommended_notional


def load_remote_control_state() -> Dict[str, Any] | None:
    if not DASHBOARD_DATABASE_URL or psycopg is None:
        return None
    try:
        with psycopg.connect(DASHBOARD_DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        trading_enabled,
                        new_entries_enabled,
                        scale_ins_enabled,
                        prematch_window_hours
                    FROM bot_control_state
                    WHERE id = 1
                    """
                )
                row = cur.fetchone()
    except Exception:
        return None
    if row is None:
        return None
    try:
        return {
            "trading_enabled": bool(row[0]),
            "new_entries_enabled": bool(row[1]),
            "scale_ins_enabled": bool(row[2]),
            "prematch_window_hours": float(row[3]),
        }
    except (TypeError, ValueError, IndexError):
        return None


def get_runtime_controls() -> Dict[str, Any]:
    local_controls = {
        "trading_enabled": TRADING_ENABLED,
        "new_entries_enabled": NEW_ENTRIES_ENABLED,
        "scale_ins_enabled": SCALE_INS_ENABLED,
        "prematch_window_hours": PREMATCH_WINDOW_HOURS,
    }
    remote_controls = load_remote_control_state()
    if remote_controls is not None:
        return remote_controls
    return local_controls


def read_fixture_mapping_index() -> List[Dict[str, Any]]:
    return _read_json_rows(FIXTURE_MAPPING_INDEX_PATH, "fixtures")


def read_market_map() -> List[Dict[str, Any]]:
    return _read_json_rows(MARKET_MAP_PATH, "markets")


def find_market_row(fixture_id: int, rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for row in rows:
        try:
            row_fixture_id = int(row.get("fixture_id"))
        except (TypeError, ValueError):
            continue
        if row_fixture_id == int(fixture_id):
            return row
    return None


def _hours_to_kickoff(mapping_row: dict, now_utc: datetime) -> float | None:
    raw_commence_time = str(mapping_row.get("commence_time", "") or "").strip()
    if not raw_commence_time:
        return None

    try:
        commence_time = datetime.fromisoformat(raw_commence_time.replace("Z", "+00:00"))
    except ValueError:
        return None

    if commence_time.tzinfo is None:
        commence_time = commence_time.replace(tzinfo=timezone.utc)
    else:
        commence_time = commence_time.astimezone(timezone.utc)

    return (commence_time - now_utc).total_seconds() / 3600.0


def kickoff_bucket(hours_to_kickoff: float) -> str | None:
    hours_to_kickoff = float(hours_to_kickoff)
    if 0.0 <= hours_to_kickoff < 1.0:
        return "0_1h"
    if 1.0 <= hours_to_kickoff < 3.0:
        return "1_3h"
    if 3.0 <= hours_to_kickoff < 6.0:
        return "3_6h"
    if 6.0 <= hours_to_kickoff <= 24.0:
        return "6_24h"
    return None


def should_scan_fixture(
    mapping_row: dict,
    now_utc: datetime,
    prematch_window_hours: float,
    include_live: bool = True,
) -> bool:
    _ = include_live
    hours_to_kickoff = _hours_to_kickoff(mapping_row, now_utc)
    if hours_to_kickoff is None:
        return False
    return 0.0 <= hours_to_kickoff <= float(prematch_window_hours)


def build_signal_row(mapping_row: Dict[str, Any], market_row: Dict[str, Any]) -> Dict[str, Any]:
    fixture_id = int(mapping_row["fixture_id"])
    snapshot = build_market_signal_snapshot(fixture_id=fixture_id)
    return {
        "timestamp": str(snapshot.get("timestamp", datetime.now(timezone.utc).isoformat()) or datetime.now(timezone.utc).isoformat()),
        "fixture_id": fixture_id,
        "market_name": str(snapshot.get("market_name", mapping_row.get("market_name", market_row.get("name", ""))) or ""),
        "league": str(snapshot.get("league", mapping_row.get("league", market_row.get("league", ""))) or ""),
        "minute": int(snapshot.get("minute", 0) or 0),
        "home_team": str(snapshot.get("home_team", market_row.get("home_team", "")) or ""),
        "away_team": str(snapshot.get("away_team", market_row.get("away_team", "")) or ""),
        "status": str(snapshot.get("status", "") or ""),
        "home_yes_fair": float(snapshot.get("home_yes_fair", 0.0) or 0.0),
        "yes_bid": snapshot.get("yes_bid"),
        "yes_ask": snapshot.get("yes_ask"),
        "yes_bid_size": snapshot.get("yes_bid_size"),
        "yes_ask_size": snapshot.get("yes_ask_size"),
        "yes_spread": snapshot.get("yes_spread"),
        "yes_edge": snapshot.get("yes_edge"),
        "no_bid": snapshot.get("no_bid"),
        "no_ask": snapshot.get("no_ask"),
        "no_bid_size": snapshot.get("no_bid_size"),
        "no_ask_size": snapshot.get("no_ask_size"),
        "no_spread": snapshot.get("no_spread"),
        "no_edge": snapshot.get("no_edge"),
        "effective_min_edge": snapshot.get("effective_min_edge"),
        "yes_tradable": snapshot.get("yes_tradable"),
        "no_tradable": snapshot.get("no_tradable"),
        "action": str(snapshot.get("action", "HOLD") or "HOLD"),
        "side": snapshot.get("side"),
    }


def _build_diagnostic_candidate(signal_row: Dict[str, Any]) -> Dict[str, Any] | None:
    effective_min_edge = signal_row.get("effective_min_edge")
    try:
        effective_min_edge_val = float(effective_min_edge) if effective_min_edge is not None else None
    except (TypeError, ValueError):
        effective_min_edge_val = None

    yes_edge = signal_row.get("yes_edge")
    no_edge = signal_row.get("no_edge")

    yes_gap = None
    no_gap = None
    if effective_min_edge_val is not None and yes_edge is not None:
        try:
            yes_gap = effective_min_edge_val - float(yes_edge)
        except (TypeError, ValueError):
            yes_gap = None
    if effective_min_edge_val is not None and no_edge is not None:
        try:
            no_gap = effective_min_edge_val - float(no_edge)
        except (TypeError, ValueError):
            no_gap = None

    valid_gaps = [gap for gap in (yes_gap, no_gap) if gap is not None]
    if not valid_gaps:
        return None

    return {
        "market_name": signal_row.get("market_name"),
        "league": signal_row.get("league"),
        "fixture_id": signal_row.get("fixture_id"),
        "home_team": signal_row.get("home_team"),
        "away_team": signal_row.get("away_team"),
        "minute": signal_row.get("minute"),
        "yes_edge": yes_edge,
        "no_edge": no_edge,
        "yes_gap": yes_gap,
        "no_gap": no_gap,
        "yes_tradable": signal_row.get("yes_tradable"),
        "no_tradable": signal_row.get("no_tradable"),
        "yes_spread": signal_row.get("yes_spread"),
        "no_spread": signal_row.get("no_spread"),
        "yes_ask_size": signal_row.get("yes_ask_size"),
        "no_ask_size": signal_row.get("no_ask_size"),
        "effective_min_edge": effective_min_edge,
        "best_gap": min(valid_gaps),
    }


def _build_research_candidate(signal_row: Dict[str, Any]) -> Dict[str, Any] | None:
    yes_tradable = signal_row.get("yes_tradable")
    no_tradable = signal_row.get("no_tradable")
    yes_edge = signal_row.get("yes_edge")
    no_edge = signal_row.get("no_edge")

    yes_qualifies = (
        yes_tradable is True
        and yes_edge is not None
        and float(yes_edge) >= RESEARCH_EFFECTIVE_MIN_EDGE
    )
    no_qualifies = (
        no_tradable is True
        and no_edge is not None
        and float(no_edge) >= RESEARCH_EFFECTIVE_MIN_EDGE
    )
    if not yes_qualifies and not no_qualifies:
        return None

    research_side = None
    research_edge = None
    if yes_qualifies and (not no_qualifies or float(yes_edge) >= float(no_edge)):
        research_side = "YES"
        research_edge = yes_edge
    elif no_qualifies:
        research_side = "NO"
        research_edge = no_edge

    return {
        "market_name": signal_row.get("market_name"),
        "league": signal_row.get("league"),
        "fixture_id": signal_row.get("fixture_id"),
        "home_team": signal_row.get("home_team"),
        "away_team": signal_row.get("away_team"),
        "minute": signal_row.get("minute"),
        "research_effective_min_edge": RESEARCH_EFFECTIVE_MIN_EDGE,
        "production_effective_min_edge": signal_row.get("effective_min_edge"),
        "yes_edge": yes_edge,
        "no_edge": no_edge,
        "yes_tradable": yes_tradable,
        "no_tradable": no_tradable,
        "yes_spread": signal_row.get("yes_spread"),
        "no_spread": signal_row.get("no_spread"),
        "yes_ask_size": signal_row.get("yes_ask_size"),
        "no_ask_size": signal_row.get("no_ask_size"),
        "research_side": research_side,
        "research_edge": research_edge,
    }


def run_loop() -> None:
    while True:
        runtime_controls = get_runtime_controls()
        trading_enabled = bool(runtime_controls.get("trading_enabled", TRADING_ENABLED))
        new_entries_enabled = bool(runtime_controls.get("new_entries_enabled", NEW_ENTRIES_ENABLED))
        scale_ins_enabled = bool(runtime_controls.get("scale_ins_enabled", SCALE_INS_ENABLED))
        prematch_window_hours = float(
            runtime_controls.get("prematch_window_hours", PREMATCH_WINDOW_HOURS)
        )
        settlement_summary = settle_open_positions()
        if int(settlement_summary.get("positions_settled", 0) or 0) > 0:
            print(
                json.dumps(
                    {
                        "event_type": "paper_settlement_summary",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "positions_checked": int(settlement_summary.get("positions_checked", 0) or 0),
                        "positions_settled": int(settlement_summary.get("positions_settled", 0) or 0),
                        "positions_still_open": int(settlement_summary.get("positions_still_open", 0) or 0),
                    },
                    sort_keys=True,
                )
            )
        risk_snapshot = build_risk_snapshot()
        projected_risk_snapshot = clone_risk_snapshot(risk_snapshot)
        mapping_rows = read_fixture_mapping_index()
        market_rows = read_market_map()
        now_utc = datetime.now(timezone.utc)

        fixtures_eligible = 0
        fixtures_scanned = 0
        live_fixtures_scanned = 0
        signals_found = 0
        research_signals_found = 0
        diagnostic_candidates: List[Dict[str, Any]] = []
        research_candidates: List[Dict[str, Any]] = []
        execution_candidates: List[Dict[str, Any]] = []
        bucket_counts = {"0_1h": 0, "1_3h": 0, "3_6h": 0, "6_24h": 0}

        for mapping_row in mapping_rows:
            hours_to_kickoff = _hours_to_kickoff(mapping_row, now_utc)
            if hours_to_kickoff is None:
                continue
            is_prematch_eligible = should_scan_fixture(
                mapping_row=mapping_row,
                now_utc=now_utc,
                prematch_window_hours=prematch_window_hours,
            )
            if is_prematch_eligible:
                fixtures_eligible += 1
                bucket = kickoff_bucket(hours_to_kickoff)
                if bucket is not None:
                    bucket_counts[bucket] += 1
            elif not INCLUDE_LIVE_FIXTURES:
                continue

            try:
                fixture_id = int(mapping_row.get("fixture_id"))
            except (TypeError, ValueError):
                continue

            market_row = find_market_row(fixture_id, market_rows)
            if market_row is None:
                continue

            try:
                signal_row = build_signal_row(mapping_row, market_row)
            except Exception:
                continue

            if not is_prematch_eligible:
                if signal_row.get("status") not in {"1H", "HT", "2H"}:
                    continue
                fixtures_eligible += 1
                live_fixtures_scanned += 1

            fixtures_scanned += 1
            diagnostic_candidate = _build_diagnostic_candidate(signal_row)
            if diagnostic_candidate is not None:
                diagnostic_candidates.append(diagnostic_candidate)
            research_candidate = _build_research_candidate(signal_row)
            if research_candidate is not None:
                research_candidates.append(research_candidate)
                research_signals_found += 1
            if signal_row.get("action") != "HOLD":
                sizing_snapshot = size_from_signal_snapshot(signal_row, bankroll=DEFAULT_BANKROLL)
                signal_row["recommended_notional"] = sizing_snapshot.get("recommended_notional")
                signal_row["recommended_shares"] = sizing_snapshot.get("recommended_shares")
                side = str(signal_row.get("side") or "").upper()
                action = str(signal_row.get("action", "") or "")
                if action in {"BUY_YES", "BUY_NO"} and side in {"YES", "NO"}:
                    try:
                        recommended_notional = float(signal_row.get("recommended_notional", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        recommended_notional = 0.0
                    if recommended_notional > 0.0:
                        priority_score = compute_priority_score(signal_row)
                        if side == "YES":
                            edge_value = signal_row.get("yes_edge")
                        else:
                            edge_value = signal_row.get("no_edge")
                        execution_candidates.append(
                            {
                                "signal_row": dict(signal_row),
                                "fixture_id": signal_row.get("fixture_id"),
                                "league": signal_row.get("league"),
                                "market_name": signal_row.get("market_name"),
                                "side": side,
                                "recommended_notional": recommended_notional,
                                "edge": edge_value,
                                "timestamp": signal_row.get("timestamp"),
                                "priority_score": priority_score,
                                "mapping_row": dict(mapping_row),
                                "market_row": dict(market_row),
                            }
                        )

        execution_candidates.sort(
            key=lambda row: (
                float(row.get("priority_score", 0.0) or 0.0),
                float(row.get("edge", 0.0) or 0.0),
                float(row.get("recommended_notional", 0.0) or 0.0),
            ),
            reverse=True,
        )
        for candidate in execution_candidates:
            try:
                refreshed_signal_row = build_signal_row(candidate["mapping_row"], candidate["market_row"])
            except Exception:
                continue

            if str(refreshed_signal_row.get("action", "") or "") == "HOLD":
                continue

            refreshed_side = str(refreshed_signal_row.get("side") or "").upper()
            refreshed_action = str(refreshed_signal_row.get("action", "") or "")
            if refreshed_action not in {"BUY_YES", "BUY_NO"} or refreshed_side not in {"YES", "NO"}:
                continue

            sizing_snapshot = size_from_signal_snapshot(refreshed_signal_row, bankroll=DEFAULT_BANKROLL)
            refreshed_signal_row["recommended_notional"] = sizing_snapshot.get("recommended_notional")
            refreshed_signal_row["recommended_shares"] = sizing_snapshot.get("recommended_shares")
            priority_score = compute_priority_score(refreshed_signal_row)
            try:
                refreshed_notional = float(refreshed_signal_row.get("recommended_notional", 0.0) or 0.0)
            except (TypeError, ValueError):
                refreshed_notional = 0.0
            if refreshed_notional <= 0.0:
                continue

            execution_result = {"executed": False, "reason": "trading_disabled"}
            existing_position = None
            if trading_enabled:
                open_positions_payload = projected_risk_snapshot.get(
                    "_open_positions_payload",
                    {"positions": []},
                )
                existing_position = find_open_position(
                    open_positions_payload,
                    int(refreshed_signal_row.get("fixture_id")),
                    str(refreshed_signal_row.get("side") or ""),
                )
                if existing_position is None and not new_entries_enabled:
                    execution_result = {"executed": False, "reason": "new_entries_disabled"}
                elif existing_position is not None and not scale_ins_enabled:
                    execution_result = {"executed": False, "reason": "scale_ins_disabled"}
                else:
                    risk_block_reason = check_risk_limits(refreshed_signal_row, projected_risk_snapshot)
                    if risk_block_reason is not None:
                        execution_result = {"executed": False, "reason": risk_block_reason}
                    else:
                        execution_result = maybe_execute_paper_trade(refreshed_signal_row, sizing_snapshot)
                        if execution_result.get("executed") is True:
                            apply_execution_to_risk_snapshot(
                                projected_risk_snapshot,
                                refreshed_signal_row,
                                refreshed_notional,
                                existing_position is None,
                            )

            signal_event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "fixture_id": refreshed_signal_row.get("fixture_id"),
                "market_name": refreshed_signal_row.get("market_name"),
                "league": refreshed_signal_row.get("league"),
                "side": refreshed_signal_row.get("side"),
                "status": refreshed_signal_row.get("status"),
                "minute": refreshed_signal_row.get("minute"),
                "yes_edge": refreshed_signal_row.get("yes_edge"),
                "no_edge": refreshed_signal_row.get("no_edge"),
                "yes_ask": refreshed_signal_row.get("yes_ask"),
                "no_ask": refreshed_signal_row.get("no_ask"),
                "yes_ask_size": refreshed_signal_row.get("yes_ask_size"),
                "no_ask_size": refreshed_signal_row.get("no_ask_size"),
                "recommended_notional": sizing_snapshot.get("recommended_notional"),
                "recommended_shares": sizing_snapshot.get("recommended_shares"),
                "sizing_reason": sizing_snapshot.get("reason"),
                "risk_cap_notional": sizing_snapshot.get("risk_cap_notional"),
                "book_cap_notional": sizing_snapshot.get("book_cap_notional"),
                "edge_scale": sizing_snapshot.get("edge_scale"),
                "executed": execution_result.get("executed"),
                "execution_reason": execution_result.get("reason"),
                "priority_score": priority_score,
            }
            if execution_result.get("executed") is True:
                signal_event["projected_open_total_notional_after"] = float(
                    projected_risk_snapshot.get("open_total_notional", 0.0) or 0.0
                )
            print(json.dumps(signal_event, sort_keys=True))
            append_jsonl(
                SIGNAL_LOG_PATH,
                {
                    "event_type": "buy_signal",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": signal_event,
                },
            )
            signals_found += 1

        diagnostic_candidates.sort(key=lambda row: row["best_gap"])
        near_signals = []
        for row in diagnostic_candidates[:3]:
            row_copy = dict(row)
            row_copy.pop("best_gap", None)
            near_signals.append(row_copy)
        research_candidates.sort(
            key=lambda row: float(row.get("research_edge", 0.0) or 0.0),
            reverse=True,
        )

        summary_payload = {
            "scan_complete": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fixtures_eligible": fixtures_eligible,
            "fixtures_scanned": fixtures_scanned,
            "live_fixtures_scanned": live_fixtures_scanned,
            "signals_found": signals_found,
            "research_signals_found": research_signals_found,
            "prematch_window_hours": prematch_window_hours,
            "include_live_fixtures": INCLUDE_LIVE_FIXTURES,
            "trading_enabled": trading_enabled,
            "new_entries_enabled": new_entries_enabled,
            "scale_ins_enabled": scale_ins_enabled,
            "bucket_counts": bucket_counts,
            "risk": {
                "max_total_open_notional": MAX_TOTAL_OPEN_NOTIONAL,
                "max_per_fixture_notional": MAX_PER_FIXTURE_NOTIONAL,
                "max_open_positions": MAX_OPEN_POSITIONS,
                "max_per_league_notional": MAX_PER_LEAGUE_NOTIONAL,
                "daily_realized_loss_stop": DAILY_REALIZED_LOSS_STOP,
                "open_positions": int(projected_risk_snapshot.get("open_positions", 0) or 0),
                "open_total_notional": float(projected_risk_snapshot.get("open_total_notional", 0.0) or 0.0),
                "today_realized_pnl": float(projected_risk_snapshot.get("today_realized_pnl", 0.0) or 0.0),
            },
        }
        print(
            json.dumps(
                summary_payload,
                sort_keys=True,
            )
        )
        append_jsonl(SCAN_LOG_PATH, summary_payload)
        near_signals_payload = {
            "near_signals": near_signals,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        print(
            json.dumps(
                near_signals_payload,
                sort_keys=True,
            )
        )
        for row in near_signals:
            append_jsonl(
                SIGNAL_LOG_PATH,
                {
                    "event_type": "near_signal",
                    "timestamp": near_signals_payload["timestamp"],
                    "data": row,
                },
            )
        research_signals_payload = {
            "research_signals": research_candidates[:5],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        print(
            json.dumps(
                research_signals_payload,
                sort_keys=True,
            )
        )
        for row in research_signals_payload["research_signals"]:
            append_jsonl(
                SIGNAL_LOG_PATH,
                {
                    "event_type": "research_signal",
                    "timestamp": research_signals_payload["timestamp"],
                    "data": row,
                },
            )
        time.sleep(10)


def main() -> None:
    try:
        run_loop()
    except KeyboardInterrupt:
        print("signal loop stopped")


if __name__ == "__main__":
    main()
