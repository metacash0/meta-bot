from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Set

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


class SoccerSportsFeed:
    def __init__(self, poll_interval_sec: float | None = None):
        self.base_url = os.getenv("APIFOOTBALL_BASE_URL", "https://v3.football.api-sports.io").rstrip("/")
        self.api_key = os.getenv("APIFOOTBALL_API_KEY", "")
        self.poll_interval_sec = float(
            poll_interval_sec if poll_interval_sec is not None else _env_float("SPORTS_POLL_INTERVAL_SEC", 5.0)
        )
        self.debug = _env_bool("SPORTS_DEBUG", False)

    def poll_live_fixtures(self) -> List[Dict[str, Any]]:
        if not self.api_key or requests is None:
            return []

        url = f"{self.base_url}/fixtures"
        headers = {"x-apisports-key": self.api_key}
        params = {"live": "all"}

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=10.0)
            resp.raise_for_status()
            payload = resp.json()
        except Exception:
            return []

        rows = payload.get("response", []) if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            return []

        out: List[Dict[str, Any]] = []
        now_ts = float(time.time())

        for row in rows:
            if not isinstance(row, dict):
                continue

            fixture = row.get("fixture", {}) if isinstance(row.get("fixture"), dict) else {}
            league = row.get("league", {}) if isinstance(row.get("league"), dict) else {}
            teams = row.get("teams", {}) if isinstance(row.get("teams"), dict) else {}
            goals = row.get("goals", {}) if isinstance(row.get("goals"), dict) else {}
            score = row.get("score", {}) if isinstance(row.get("score"), dict) else {}
            status = fixture.get("status", {}) if isinstance(fixture.get("status"), dict) else {}

            fixture_id = fixture.get("id")
            try:
                fixture_id_int = int(fixture_id)
            except (TypeError, ValueError):
                continue

            minute = status.get("elapsed", 0)
            try:
                minute = int(minute) if minute is not None else 0
            except (TypeError, ValueError):
                minute = 0

            score_home = goals.get("home", 0)
            score_away = goals.get("away", 0)
            try:
                score_home = int(score_home) if score_home is not None else 0
            except (TypeError, ValueError):
                score_home = 0
            try:
                score_away = int(score_away) if score_away is not None else 0
            except (TypeError, ValueError):
                score_away = 0

            # API-Football can expose cards differently across endpoints; keep safe defaults.
            red_home = 0
            red_away = 0
            cards = row.get("cards")
            if isinstance(cards, dict):
                home_cards = cards.get("home", {}) if isinstance(cards.get("home"), dict) else {}
                away_cards = cards.get("away", {}) if isinstance(cards.get("away"), dict) else {}
                try:
                    red_home = int(home_cards.get("red", 0) or 0)
                except (TypeError, ValueError):
                    red_home = 0
                try:
                    red_away = int(away_cards.get("red", 0) or 0)
                except (TypeError, ValueError):
                    red_away = 0

            out.append(
                {
                    "fixture_id": fixture_id_int,
                    "ts": now_ts,
                    "sport": "soccer",
                    "league_name": str(league.get("name", "") or ""),
                    "home_team": str((teams.get("home", {}) or {}).get("name", "") or ""),
                    "away_team": str((teams.get("away", {}) or {}).get("name", "") or ""),
                    "minute": minute,
                    "status_short": str(status.get("short", "") or ""),
                    "status_long": str(status.get("long", "") or ""),
                    "score_home": score_home,
                    "score_away": score_away,
                    "red_home": red_home,
                    "red_away": red_away,
                }
            )

        if self.debug:
            print("[sportsfeed] polled fixtures=%d" % len(out))

        return out


def load_market_map(path: str = "market_map.json") -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {"markets": []}

    if not isinstance(payload, dict):
        return {"markets": []}

    markets = payload.get("markets")
    if not isinstance(markets, list):
        return {"markets": []}

    return payload


def get_tracked_fixture_ids_from_market_map(path: str = "market_map.json") -> Set[int]:
    payload = load_market_map(path=path)
    markets = payload.get("markets", [])
    if not isinstance(markets, list):
        return set()

    fixture_ids: Set[int] = set()
    for item in markets:
        if not isinstance(item, dict):
            continue
        value = item.get("fixture_id")
        try:
            fixture_ids.add(int(value))
        except (TypeError, ValueError):
            continue

    return fixture_ids
