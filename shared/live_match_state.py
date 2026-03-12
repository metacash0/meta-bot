from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import requests


APIFOOTBALL_BASE_URL = os.getenv("APIFOOTBALL_BASE_URL", "https://v3.football.api-sports.io").rstrip("/")
APIFOOTBALL_API_KEY = os.getenv("APIFOOTBALL_API_KEY", "")
FINISHED_SHORT_STATUSES = {"FT", "AET", "PEN", "CANC", "ABD", "AWD", "WO", "FINISHED"}
FINISHED_LONG_HINTS = (
    "finished",
    "after extra time",
    "penalties",
    "penalty",
    "cancelled",
    "canceled",
    "abandoned",
    "awarded",
    "walkover",
)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _event_is_red_card(event: Dict[str, Any]) -> bool:
    event_type = str(event.get("type", "") or "").strip().lower()
    detail = str(event.get("detail", "") or "").strip().lower()
    if "card" not in event_type:
        return False
    return "red card" in detail or "second yellow card" in detail


def is_fixture_finished(status: str, status_long: str = "") -> bool:
    short = str(status or "").strip().upper()
    long_text = str(status_long or "").strip().lower()
    if short in FINISHED_SHORT_STATUSES:
        return True
    return any(hint in long_text for hint in FINISHED_LONG_HINTS)


def get_live_match_state(fixture_id: int) -> dict:
    if not APIFOOTBALL_API_KEY:
        raise RuntimeError("APIFOOTBALL_API_KEY missing")

    fixture_id = _safe_int(fixture_id, default=0)
    if fixture_id <= 0:
        raise RuntimeError("fixture_id must be positive")

    url = f"{APIFOOTBALL_BASE_URL}/fixtures"
    headers = {"x-apisports-key": APIFOOTBALL_API_KEY}
    params = {"id": fixture_id}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10.0)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise RuntimeError("API-Football request failed for fixture_id=%s: %s" % (fixture_id, exc)) from exc

    rows = payload.get("response", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("fixture not found for fixture_id=%s" % fixture_id)

    row = _safe_dict(rows[0])
    fixture = _safe_dict(row.get("fixture"))
    status_info = _safe_dict(fixture.get("status"))
    teams = _safe_dict(row.get("teams"))
    home_team_info = _safe_dict(teams.get("home"))
    away_team_info = _safe_dict(teams.get("away"))
    goals = _safe_dict(row.get("goals"))
    league = _safe_dict(row.get("league"))
    events = _safe_list(row.get("events"))

    home_team = str(home_team_info.get("name", "") or "")
    away_team = str(away_team_info.get("name", "") or "")
    status = str(status_info.get("short", "") or "")
    status_long = str(status_info.get("long", "") or "")
    minute = max(0, _safe_int(status_info.get("elapsed"), default=0))
    score_home = max(0, _safe_int(goals.get("home"), default=0))
    score_away = max(0, _safe_int(goals.get("away"), default=0))

    red_home = 0
    red_away = 0
    for raw_event in events:
        event = _safe_dict(raw_event)
        if not _event_is_red_card(event):
            continue
        team = _safe_dict(event.get("team"))
        team_name = str(team.get("name", "") or "")
        if team_name == home_team:
            red_home += 1
        elif team_name == away_team:
            red_away += 1

    return {
        "fixture_id": fixture_id,
        "minute": minute,
        "score_home": score_home,
        "score_away": score_away,
        "red_home": red_home,
        "red_away": red_away,
        "status": status,
        "status_long": status_long,
        "home_team": home_team,
        "away_team": away_team,
        "league": str(league.get("name", "") or ""),
        "raw_event_count": len(events),
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python3 -m shared.live_match_state <fixture_id>")
        return

    fixture_id = _safe_int(sys.argv[1], default=0)
    result = get_live_match_state(fixture_id)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
