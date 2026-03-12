from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

from shared.live_match_state import get_live_match_state
from shared.live_soccer_fair import estimate_live_probs


PREMATCH_LAMBDAS_PATH = "data/prematch_lambdas.json"


def _read_prematch_rows() -> List[Dict[str, Any]]:
    if not os.path.exists(PREMATCH_LAMBDAS_PATH):
        return []

    try:
        with open(PREMATCH_LAMBDAS_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []

    rows = payload.get("fixtures", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def load_prematch_lambda_row(fixture_id: int) -> dict:
    fixture_id = int(fixture_id)
    for row in _read_prematch_rows():
        try:
            row_fixture_id = int(row.get("fixture_id"))
        except (TypeError, ValueError):
            continue
        if row_fixture_id == fixture_id:
            return row
    raise RuntimeError("prematch lambda row not found for fixture_id=%s" % fixture_id)


def build_live_fair_snapshot(fixture_id: int) -> dict:
    prematch = load_prematch_lambda_row(fixture_id)
    live_state = get_live_match_state(fixture_id)

    lambda_home = float(prematch.get("lambda_home", 0.0) or 0.0)
    lambda_away = float(prematch.get("lambda_away", 0.0) or 0.0)
    lambda_total = float(prematch.get("lambda_total", lambda_home + lambda_away) or 0.0)

    fair = estimate_live_probs(
        lambda_home=lambda_home,
        lambda_away=lambda_away,
        minute=int(live_state.get("minute", 0) or 0),
        score_home=int(live_state.get("score_home", 0) or 0),
        score_away=int(live_state.get("score_away", 0) or 0),
        red_home=int(live_state.get("red_home", 0) or 0),
        red_away=int(live_state.get("red_away", 0) or 0),
        status=str(live_state.get("status", "") or ""),
    )

    home_win_prob = float(fair.get("home_win_prob", 0.0) or 0.0)
    home_yes_fair = home_win_prob
    home_no_fair = 1.0 - home_win_prob

    return {
        "fixture_id": int(fixture_id),
        "home_team": str(live_state.get("home_team", prematch.get("home_team", "")) or ""),
        "away_team": str(live_state.get("away_team", prematch.get("away_team", "")) or ""),
        "league": str(live_state.get("league", prematch.get("league", "")) or ""),
        "minute": int(live_state.get("minute", 0) or 0),
        "score_home": int(live_state.get("score_home", 0) or 0),
        "score_away": int(live_state.get("score_away", 0) or 0),
        "red_home": int(live_state.get("red_home", 0) or 0),
        "red_away": int(live_state.get("red_away", 0) or 0),
        "status": str(live_state.get("status", "") or ""),
        "status_long": str(live_state.get("status_long", "") or ""),
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
        "lambda_total": lambda_total,
        "home_win_prob": home_win_prob,
        "draw_prob": float(fair.get("draw_prob", 0.0) or 0.0),
        "away_win_prob": float(fair.get("away_win_prob", 0.0) or 0.0),
        "home_yes_fair": home_yes_fair,
        "home_no_fair": home_no_fair,
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python3 -m shared.live_fair_snapshot <fixture_id>")
        return

    fixture_id = int(sys.argv[1])
    payload = build_live_fair_snapshot(fixture_id)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
