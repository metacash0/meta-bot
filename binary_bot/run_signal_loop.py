from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from shared.market_signal_snapshot import build_market_signal_snapshot
from shared.order_sizing import size_from_signal_snapshot


FIXTURE_MAPPING_INDEX_PATH = "data/fixture_mapping_index.json"
MARKET_MAP_PATH = "data/market_map.json"
SCAN_SLEEP_SEC = 10.0
DEFAULT_BANKROLL = 5000.0


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


def build_signal_row(mapping_row: Dict[str, Any], market_row: Dict[str, Any]) -> Dict[str, Any]:
    fixture_id = int(mapping_row["fixture_id"])
    snapshot = build_market_signal_snapshot(fixture_id=fixture_id)
    signal_row = {
        "timestamp": str(snapshot.get("timestamp", datetime.now(timezone.utc).isoformat()) or datetime.now(timezone.utc).isoformat()),
        "fixture_id": fixture_id,
        "market_name": str(snapshot.get("market_name", mapping_row.get("market_name", market_row.get("name", ""))) or ""),
        "league": str(snapshot.get("league", mapping_row.get("league", market_row.get("league", ""))) or ""),
        "minute": int(snapshot.get("minute", 0) or 0),
        "home_team": str(snapshot.get("home_team", market_row.get("home_team", "")) or ""),
        "away_team": str(snapshot.get("away_team", market_row.get("away_team", "")) or ""),
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
        "action": str(snapshot.get("action", "HOLD") or "HOLD"),
        "side": snapshot.get("side"),
    }
    if signal_row["action"] != "HOLD":
        sizing = size_from_signal_snapshot(snapshot, bankroll=DEFAULT_BANKROLL)
        signal_row.update(
            {
                "recommended_notional": sizing.get("recommended_notional"),
                "recommended_shares": sizing.get("recommended_shares"),
                "reason": sizing.get("reason"),
                "risk_cap_notional": sizing.get("risk_cap_notional"),
                "book_cap_notional": sizing.get("book_cap_notional"),
                "edge_scale": sizing.get("edge_scale"),
            }
        )
    return signal_row


def run_loop() -> None:
    while True:
        mapping_rows = read_fixture_mapping_index()
        market_rows = read_market_map()

        fixtures_scanned = 0
        signals_found = 0

        for mapping_row in mapping_rows:
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

            fixtures_scanned += 1
            if signal_row.get("action") != "HOLD":
                print(json.dumps(signal_row, sort_keys=True))
                signals_found += 1

        print(
            json.dumps(
                {
                    "scan_complete": True,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "fixtures_scanned": fixtures_scanned,
                    "signals_found": signals_found,
                },
                sort_keys=True,
            )
        )
        time.sleep(10)


def main() -> None:
    try:
        run_loop()
    except KeyboardInterrupt:
        print("signal loop stopped")


if __name__ == "__main__":
    main()
