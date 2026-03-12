from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from shared.live_fair_snapshot import build_live_fair_snapshot
from shared.market_edge import evaluate_home_win_market
from shared.polymarket_quotes import get_binary_quotes


FIXTURE_MAPPING_INDEX_PATH = "data/fixture_mapping_index.json"
MARKET_MAP_PATH = "data/market_map.json"
SCAN_SLEEP_SEC = 10.0


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
    fair_snapshot = build_live_fair_snapshot(fixture_id)
    quotes = get_binary_quotes(
        yes_asset_id=str(market_row.get("yes_asset_id", "") or ""),
        no_asset_id=str(market_row.get("no_asset_id", "") or ""),
    )
    edge = evaluate_home_win_market(
        fair_snapshot=fair_snapshot,
        yes_bid=quotes.get("yes_bid"),
        yes_ask=quotes.get("yes_ask"),
        no_bid=quotes.get("no_bid"),
        no_ask=quotes.get("no_ask"),
    )

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fixture_id": fixture_id,
        "market_name": str(mapping_row.get("market_name", market_row.get("name", "")) or ""),
        "league": str(mapping_row.get("league", fair_snapshot.get("league", "")) or ""),
        "minute": int(fair_snapshot.get("minute", 0) or 0),
        "home_team": str(fair_snapshot.get("home_team", market_row.get("home_team", "")) or ""),
        "away_team": str(fair_snapshot.get("away_team", market_row.get("away_team", "")) or ""),
        "home_yes_fair": float(fair_snapshot.get("home_yes_fair", 0.0) or 0.0),
        "yes_bid": quotes.get("yes_bid"),
        "yes_ask": quotes.get("yes_ask"),
        "yes_edge": edge.get("yes_edge"),
        "action": str(edge.get("action", "HOLD") or "HOLD"),
    }


def run_loop() -> None:
    while True:
        loop_timestamp = datetime.now(timezone.utc).isoformat()
        mapping_rows = read_fixture_mapping_index()
        market_rows = read_market_map()

        scanned = 0
        signaled = 0

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

            scanned += 1
            if signal_row.get("action") != "HOLD":
                signaled += 1
            print(json.dumps(signal_row, sort_keys=True))

        print(
            json.dumps(
                {
                    "timestamp": loop_timestamp,
                    "fixtures_scanned": scanned,
                    "signals_triggered": signaled,
                },
                sort_keys=True,
            )
        )
        time.sleep(SCAN_SLEEP_SEC)


def main() -> None:
    run_loop()


if __name__ == "__main__":
    main()
