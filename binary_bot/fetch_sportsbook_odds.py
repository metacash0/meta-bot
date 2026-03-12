from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import requests


ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_API_BASE_URL = os.getenv("ODDS_API_BASE_URL", "https://api.the-odds-api.com/v4").rstrip("/")
SPORT_KEYS = [
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_italy_serie_a",
    "soccer_germany_bundesliga",
    "soccer_france_ligue_one",
    "soccer_uefa_champs_league",
]


def fetch_league_odds(sport_key: str) -> list[dict]:
    if not ODDS_API_KEY:
        raise RuntimeError("ODDS_API_KEY missing")

    url = f"{ODDS_API_BASE_URL}/sports/{sport_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "uk,eu",
        "markets": "h2h,totals",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }

    try:
        response = requests.get(url, params=params, timeout=10.0)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise RuntimeError("failed to fetch odds for sport_key=%s: %s" % (sport_key, exc)) from exc

    if not isinstance(payload, list):
        return []

    rows: List[dict] = []
    for fixture in payload:
        if not isinstance(fixture, dict):
            continue
        row = dict(fixture)
        row["_source_sport_key"] = sport_key
        rows.append(row)
    return rows


def fetch_all_selected_leagues() -> dict:
    if not ODDS_API_KEY:
        raise RuntimeError("ODDS_API_KEY missing")

    fixtures: List[dict] = []
    succeeded: List[str] = []

    for sport_key in SPORT_KEYS:
        try:
            league_rows = fetch_league_odds(sport_key)
        except RuntimeError as exc:
            print("warning: %s" % exc)
            continue
        fixtures.extend(league_rows)
        succeeded.append(sport_key)

    return {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "sport_keys": succeeded,
        "fixtures": fixtures,
    }


def write_raw_snapshot(payload: dict, path: str = "data/sportsbook_odds_raw.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")


def main() -> None:
    payload = fetch_all_selected_leagues()
    write_raw_snapshot(payload)
    print("leagues attempted: %d" % len(SPORT_KEYS))
    print("leagues succeeded: %d" % len(payload.get("sport_keys", [])))
    print("total fixtures written: %d" % len(payload.get("fixtures", [])))


if __name__ == "__main__":
    main()
