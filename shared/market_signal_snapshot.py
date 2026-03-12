from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

from shared.live_fair_snapshot import build_live_fair_snapshot
from shared.market_edge import evaluate_home_win_market


MARKET_MAP_PATH = "data/market_map.json"


def _read_market_rows() -> List[Dict[str, Any]]:
    if not os.path.exists(MARKET_MAP_PATH):
        return []

    try:
        with open(MARKET_MAP_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []

    rows = payload.get("markets", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def load_market_row_by_fixture_id(fixture_id: int) -> dict:
    fixture_id = int(fixture_id)
    for row in _read_market_rows():
        try:
            row_fixture_id = int(row.get("fixture_id"))
        except (TypeError, ValueError):
            continue
        if row_fixture_id == fixture_id:
            return row
    raise RuntimeError("market row not found for fixture_id=%s" % fixture_id)


def build_market_signal_snapshot(
    fixture_id: int,
    yes_bid: float | None = None,
    yes_ask: float | None = None,
    no_bid: float | None = None,
    no_ask: float | None = None,
    min_edge: float = 0.03,
) -> dict:
    market_row = load_market_row_by_fixture_id(fixture_id)
    fair_snapshot = build_live_fair_snapshot(fixture_id)
    edge_eval = evaluate_home_win_market(
        fair_snapshot=fair_snapshot,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
        min_edge=min_edge,
    )

    return {
        "fixture_id": int(fixture_id),
        "market_name": str(market_row.get("name", "") or ""),
        "league": str(fair_snapshot.get("league", market_row.get("league", "")) or ""),
        "home_team": str(fair_snapshot.get("home_team", market_row.get("home_team", "")) or ""),
        "away_team": str(fair_snapshot.get("away_team", market_row.get("away_team", "")) or ""),
        "minute": int(fair_snapshot.get("minute", 0) or 0),
        "score_home": int(fair_snapshot.get("score_home", 0) or 0),
        "score_away": int(fair_snapshot.get("score_away", 0) or 0),
        "red_home": int(fair_snapshot.get("red_home", 0) or 0),
        "red_away": int(fair_snapshot.get("red_away", 0) or 0),
        "status": str(fair_snapshot.get("status", "") or ""),
        "status_long": str(fair_snapshot.get("status_long", "") or ""),
        "home_yes_fair": float(fair_snapshot.get("home_yes_fair", 0.0) or 0.0),
        "home_no_fair": float(fair_snapshot.get("home_no_fair", 0.0) or 0.0),
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "yes_mid": edge_eval.get("yes_mid"),
        "no_bid": no_bid,
        "no_ask": no_ask,
        "no_mid": edge_eval.get("no_mid"),
        "yes_edge": edge_eval.get("yes_edge"),
        "no_edge": edge_eval.get("no_edge"),
        "min_edge": float(edge_eval.get("min_edge", min_edge) or min_edge),
        "action": str(edge_eval.get("action", "HOLD") or "HOLD"),
        "side": edge_eval.get("side"),
        "yes_asset_id": str(market_row.get("yes_asset_id", "") or ""),
        "no_asset_id": str(market_row.get("no_asset_id", "") or ""),
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python3 -m shared.market_signal_snapshot <fixture_id>")
        return

    fixture_id = int(sys.argv[1])
    payload = build_market_signal_snapshot(
        fixture_id=fixture_id,
        yes_bid=0.47,
        yes_ask=0.49,
        no_bid=0.51,
        no_ask=0.53,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
