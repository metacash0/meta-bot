from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from binary_bot.paper_executor import maybe_execute_paper_trade
from binary_bot.paper_settlement import settle_open_positions
from shared.market_signal_snapshot import build_market_signal_snapshot
from shared.order_sizing import size_from_signal_snapshot


FIXTURE_MAPPING_INDEX_PATH = "data/fixture_mapping_index.json"
MARKET_MAP_PATH = "data/market_map.json"
SCAN_LOG_PATH = "data/logs/scan_summary.jsonl"
SIGNAL_LOG_PATH = "data/logs/signal_events.jsonl"
SCAN_SLEEP_SEC = 10.0
DEFAULT_BANKROLL = 5000.0
PREMATCH_WINDOW_HOURS = float(os.getenv("PREMATCH_WINDOW_HOURS", "6"))
RESEARCH_EFFECTIVE_MIN_EDGE = 0.025
INCLUDE_LIVE_FIXTURES = True

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
        bucket_counts = {"0_1h": 0, "1_3h": 0, "3_6h": 0, "6_24h": 0}

        for mapping_row in mapping_rows:
            hours_to_kickoff = _hours_to_kickoff(mapping_row, now_utc)
            if hours_to_kickoff is None:
                continue
            is_prematch_eligible = should_scan_fixture(
                mapping_row=mapping_row,
                now_utc=now_utc,
                prematch_window_hours=PREMATCH_WINDOW_HOURS,
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
                execution_result = maybe_execute_paper_trade(signal_row, sizing_snapshot)
                signal_event = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "fixture_id": signal_row.get("fixture_id"),
                    "market_name": signal_row.get("market_name"),
                    "league": signal_row.get("league"),
                    "side": signal_row.get("side"),
                    "status": signal_row.get("status"),
                    "minute": signal_row.get("minute"),
                    "yes_edge": signal_row.get("yes_edge"),
                    "no_edge": signal_row.get("no_edge"),
                    "yes_ask": signal_row.get("yes_ask"),
                    "no_ask": signal_row.get("no_ask"),
                    "yes_ask_size": signal_row.get("yes_ask_size"),
                    "no_ask_size": signal_row.get("no_ask_size"),
                    "recommended_notional": sizing_snapshot.get("recommended_notional"),
                    "recommended_shares": sizing_snapshot.get("recommended_shares"),
                    "sizing_reason": sizing_snapshot.get("reason"),
                    "risk_cap_notional": sizing_snapshot.get("risk_cap_notional"),
                    "book_cap_notional": sizing_snapshot.get("book_cap_notional"),
                    "edge_scale": sizing_snapshot.get("edge_scale"),
                    "executed": execution_result.get("executed"),
                    "execution_reason": execution_result.get("reason"),
                }
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
            "prematch_window_hours": PREMATCH_WINDOW_HOURS,
            "include_live_fixtures": INCLUDE_LIVE_FIXTURES,
            "bucket_counts": bucket_counts,
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
