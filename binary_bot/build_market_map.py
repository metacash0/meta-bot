from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional, Set

import requests

from binary_bot.discover_soccer_markets import LEAGUE_TAGS, discover_for_league


APIFOOTBALL_BASE_URL = os.getenv("APIFOOTBALL_BASE_URL", "https://v3.football.api-sports.io").rstrip("/")
APIFOOTBALL_API_KEY = os.getenv("APIFOOTBALL_API_KEY", "")
MARKET_MAP_PATH = "market_map.json"
TARGET_LEAGUES = {
    "Premier League",
    "La Liga",
    "Serie A",
    "Bundesliga",
    "Ligue 1",
    "Champions League",
    "Europa League",
}
SUFFIX_TOKENS = {"fc", "cf", "afc", "sc", "sk", "cd", "ca", "jk"}


def normalize_team_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    parts = [part for part in text.split() if part]
    while parts and parts[-1] in SUFFIX_TOKENS:
        parts.pop()
    return " ".join(parts)


def fetch_fixtures_for_date(match_date: str) -> List[Dict[str, Any]]:
    if not APIFOOTBALL_API_KEY:
        return []

    url = f"{APIFOOTBALL_BASE_URL}/fixtures"
    headers = {"x-apisports-key": APIFOOTBALL_API_KEY}
    params = {"date": match_date}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10.0)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []

    rows = payload.get("response", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def find_matching_fixture(home_team: str, match_date: str) -> Optional[Dict[str, Any]]:
    normalized_home = normalize_team_name(home_team)
    if not normalized_home or not match_date:
        return None

    for row in fetch_fixtures_for_date(match_date):
        teams = row.get("teams", {})
        if not isinstance(teams, dict):
            continue
        home = teams.get("home", {})
        away = teams.get("away", {})
        if not isinstance(home, dict) or not isinstance(away, dict):
            continue

        fixture = row.get("fixture", {})
        if not isinstance(fixture, dict):
            continue

        api_home_name = str(home.get("name", "") or "")
        if normalize_team_name(api_home_name) != normalized_home:
            continue

        fixture_id = fixture.get("id")
        try:
            fixture_id = int(fixture_id)
        except (TypeError, ValueError):
            continue

        return {
            "fixture_id": fixture_id,
            "home_team": api_home_name,
            "away_team": str(away.get("name", "") or ""),
        }

    return None


def discover_candidate_markets() -> List[Dict[str, str]]:
    all_rows: List[Dict[str, str]] = []
    for tag_id, league_name in LEAGUE_TAGS.items():
        rows, _ = discover_for_league(tag_id, league_name)
        all_rows.extend(rows)
    return all_rows


def write_market_map(rows: List[Dict[str, Any]]) -> None:
    payload = {"markets": rows}
    with open(MARKET_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")


def main() -> None:
    if not APIFOOTBALL_API_KEY:
        write_market_map([])
        print("APIFOOTBALL_API_KEY missing; wrote empty market_map.json")
        print("total discovered markets considered: 0")
        print("total eligible after league + home_win filtering: 0")
        print("total successfully matched to fixtures: 0")
        print("total written to market_map.json: 0")
        print("counts by league written: {}")
        return

    discovered = discover_candidate_markets()
    eligible = [
        row
        for row in discovered
        if row.get("league") in TARGET_LEAGUES
        and row.get("market_kind") == "home_win"
        and str(row.get("yes_token", "")).strip() != ""
        and str(row.get("no_token", "")).strip() != ""
    ]

    written: List[Dict[str, Any]] = []
    seen_slugs: Set[str] = set()
    seen_yes_assets: Set[str] = set()
    counts_by_league: Counter = Counter()
    matched_count = 0

    for row in eligible:
        slug = str(row.get("slug", "") or "")
        yes_token = str(row.get("yes_token", "") or "")
        if not slug or not yes_token:
            continue
        if slug in seen_slugs or yes_token in seen_yes_assets:
            continue

        fixture = find_matching_fixture(
            home_team=str(row.get("home_team", "") or ""),
            match_date=str(row.get("match_date", "") or ""),
        )
        if fixture is None:
            continue

        matched_count += 1
        league = str(row.get("league", "") or "")
        written.append(
            {
                "name": slug,
                "fixture_id": int(fixture["fixture_id"]),
                "sport": "soccer",
                "market_type": "home_win",
                "yes_asset_id": yes_token,
                "no_asset_id": str(row.get("no_token", "") or ""),
                "home_team": str(fixture.get("home_team", row.get("home_team", "")) or ""),
                "away_team": str(fixture.get("away_team", "") or ""),
                "prematch_home_win_prob": 0.45,
                "league": league,
            }
        )
        seen_slugs.add(slug)
        seen_yes_assets.add(yes_token)
        counts_by_league[league] += 1

    write_market_map(written)

    print("total discovered markets considered: %d" % len(discovered))
    print("total eligible after league + home_win filtering: %d" % len(eligible))
    print("total successfully matched to fixtures: %d" % matched_count)
    print("total written to market_map.json: %d" % len(written))
    print("counts by league written: %s" % json.dumps(dict(counts_by_league), sort_keys=True))


if __name__ == "__main__":
    main()
