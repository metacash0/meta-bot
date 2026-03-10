from __future__ import annotations

from typing import Any, Dict, List, Optional


class SoccerStateStore:
    def __init__(self):
        self._by_fixture: Dict[int, Dict[str, Any]] = {}

    def update_from_fixture_rows(self, rows: List[Dict[str, Any]]) -> None:
        for row in rows:
            if not isinstance(row, dict):
                continue

            fixture_id_raw = row.get("fixture_id")
            try:
                fixture_id = int(fixture_id_raw)
            except (TypeError, ValueError):
                continue

            score_home = row.get("score_home", 0)
            score_away = row.get("score_away", 0)
            minute = row.get("minute", 0)
            red_home = row.get("red_home", 0)
            red_away = row.get("red_away", 0)

            try:
                score_home = int(score_home)
            except (TypeError, ValueError):
                score_home = 0
            try:
                score_away = int(score_away)
            except (TypeError, ValueError):
                score_away = 0
            try:
                minute = int(minute)
            except (TypeError, ValueError):
                minute = 0
            try:
                red_home = int(red_home)
            except (TypeError, ValueError):
                red_home = 0
            try:
                red_away = int(red_away)
            except (TypeError, ValueError):
                red_away = 0

            status_short = str(row.get("status_short", "") or "")
            state = {
                "fixture_id": fixture_id,
                "ts": float(row.get("ts", 0.0) or 0.0),
                "sport": str(row.get("sport", "soccer") or "soccer"),
                "league_name": str(row.get("league_name", "") or ""),
                "home_team": str(row.get("home_team", "") or ""),
                "away_team": str(row.get("away_team", "") or ""),
                "minute": minute,
                "status_short": status_short,
                "status_long": str(row.get("status_long", "") or ""),
                "score_home": score_home,
                "score_away": score_away,
                "red_home": red_home,
                "red_away": red_away,
                "score_diff": int(score_home - score_away),
                "is_live": status_short in {"1H", "2H", "HT", "ET", "LIVE", "BT", "P"},
            }
            self._by_fixture[fixture_id] = state

    def get_fixture_state(self, fixture_id: int) -> Optional[Dict[str, Any]]:
        return self._by_fixture.get(int(fixture_id))

    def get_all_states(self) -> List[Dict[str, Any]]:
        return list(self._by_fixture.values())


def summarize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "fixture_id": int(state.get("fixture_id", 0) or 0),
        "minute": int(state.get("minute", 0) or 0),
        "status_short": str(state.get("status_short", "") or ""),
        "home_team": str(state.get("home_team", "") or ""),
        "away_team": str(state.get("away_team", "") or ""),
        "score_home": int(state.get("score_home", 0) or 0),
        "score_away": int(state.get("score_away", 0) or 0),
        "red_home": int(state.get("red_home", 0) or 0),
        "red_away": int(state.get("red_away", 0) or 0),
        "score_diff": int(state.get("score_diff", 0) or 0),
    }
