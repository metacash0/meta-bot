from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List


MARKET_MAP_PATH = "data/market_map.json"
CONSENSUS_PATH = "data/sportsbook_consensus.json"
OUTPUT_PATH = "data/fixture_mapping_index.json"
LEAGUE_TO_SPORT_KEY = {
    "Premier League": "soccer_epl",
    "La Liga": "soccer_spain_la_liga",
    "Serie A": "soccer_italy_serie_a",
    "Bundesliga": "soccer_germany_bundesliga",
    "Ligue 1": "soccer_france_ligue_one",
    "Champions League": "soccer_uefa_champs_league",
}
TEAM_ALIASES = {
    "sunderland afc": "sunderland",
    "sunderland": "sunderland",
    "brighton and hove albion": "brighton",
    "brighton": "brighton",
    "newcastle united": "newcastle",
    "newcastle": "newcastle",
    "west ham united": "west ham",
    "west ham": "west ham",
    "leeds united": "leeds",
    "leeds": "leeds",
    "tottenham hotspur": "tottenham",
    "tottenham": "tottenham",
    "deportivo alaves": "alaves",
    "alaves": "alaves",
    "club atletico de madrid": "atletico madrid",
    "atletico madrid": "atletico madrid",
    "borussia monchengladbach": "monchengladbach",
    "monchengladbach": "monchengladbach",
    "fc st pauli 1910": "st pauli",
    "st pauli": "st pauli",
    "tsg 1899 hoffenheim": "hoffenheim",
    "tsg hoffenheim": "hoffenheim",
    "1899 hoffenheim": "hoffenheim",
    "hoffenheim": "hoffenheim",
    "vfl wolfsburg": "wolfsburg",
    "wolfsburg": "wolfsburg",
    "athletic bilbao": "athletic club",
    "athletic club": "athletic club",
    "girona fc": "girona",
    "girona": "girona",
    "villarreal cf": "villarreal",
    "villarreal": "villarreal",
    "getafe cf": "getafe",
    "getafe": "getafe",
    "crystal palace fc": "crystal palace",
    "crystal palace": "crystal palace",
    "liverpool fc": "liverpool",
    "liverpool": "liverpool",
    "manchester city fc": "manchester city",
    "manchester city": "manchester city",
    "bv borussia 09 dortmund": "borussia dortmund",
    "borussia dortmund": "borussia dortmund",
    "fc augsburg": "augsburg",
    "augsburg": "augsburg",
    "bayer 04 leverkusen": "bayer leverkusen",
    "bayer leverkusen": "bayer leverkusen",
    "fc bayern munchen": "bayern munich",
    "fc bayern munchen": "bayern munich",
    "bayern munchen": "bayern munich",
    "bayern munich": "bayern munich",
    "fc lorient": "lorient",
    "lorient": "lorient",
    "rc lens": "lens",
    "racing club de lens": "lens",
    "lens": "lens",
    "as monaco fc": "monaco",
    "as monaco": "monaco",
    "monaco": "monaco",
    "stade brestois 29": "brest",
    "stade brestois": "brest",
    "brest": "brest",
    "fc internazionale milano": "inter",
    "internazionale": "inter",
    "inter milan": "inter",
    "inter": "inter",
    "atalanta bc": "atalanta",
    "atalanta": "atalanta",
    "sporting lisbon": "sporting cp",
    "sporting clube de portugal": "sporting cp",
    "sporting cp": "sporting cp",
    "sporting": "sporting cp",
    "fk bodo glimt": "bodo glimt",
    "bodo glimt": "bodo glimt",
    "bodo/glimt": "bodo glimt",
    "bodø/glimt": "bodo glimt",
}


def read_market_map() -> list[dict]:
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


def read_sportsbook_consensus() -> list[dict]:
    if not os.path.exists(CONSENSUS_PATH):
        return []
    try:
        with open(CONSENSUS_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []
    rows = payload.get("fixtures", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def normalize_team_name(name: str) -> str:
    text = str(name or "")
    text = text.replace("ø", "o").replace("Ø", "O")
    text = text.replace("/", " ")
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    parts = [part for part in text.split() if part]
    while parts and parts[-1] in {"fc", "cf", "afc", "sc"}:
        parts.pop()
    return " ".join(parts)


def canonical_team_name(name: str) -> str:
    normalized = normalize_team_name(name)
    return TEAM_ALIASES.get(normalized, normalized)


def find_consensus_row(league: str, home_team: str, away_team: str, rows: list[dict]) -> dict | None:
    target_home = canonical_team_name(home_team)
    target_away = canonical_team_name(away_team)
    target_sport_key = LEAGUE_TO_SPORT_KEY.get(str(league or ""))

    contains_match = None
    fallback_match = None
    for row in rows:
        row_sport_key = str(row.get("sport_key", "") or "")
        if target_sport_key and row_sport_key != target_sport_key:
            continue

        row_home = canonical_team_name(str(row.get("home_team", "") or ""))
        row_away = canonical_team_name(str(row.get("away_team", "") or ""))
        if row_home == target_home and row_away == target_away:
            return row

        home_contains = row_home == target_home or row_home in target_home or target_home in row_home
        away_contains = row_away == target_away or row_away in target_away or target_away in row_away
        if home_contains and away_contains and contains_match is None:
            contains_match = row
        if fallback_match is None:
            fallback_match = row

    return contains_match or fallback_match


def write_output(rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    payload = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "fixtures": rows,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")


def main() -> None:
    market_rows = read_market_map()
    consensus_rows = read_sportsbook_consensus()
    if not market_rows or not consensus_rows:
        if not market_rows:
            print("market map missing or unreadable at data/market_map.json; wrote empty fixture_mapping_index.json")
        else:
            print("sportsbook consensus missing or unreadable at data/sportsbook_consensus.json; wrote empty fixture_mapping_index.json")
        write_output([])
        print("total fixtures read: 0")
        print("total consensus rows available: 0")
        print("total mappings written: 0")
        print("total unmatched fixtures: 0")
        print("counts by league: {}")
        return

    output_rows: List[Dict[str, Any]] = []
    unmatched: List[Dict[str, str]] = []
    counts_by_league: Counter = Counter()

    for market_row in market_rows:
        try:
            fixture_id = int(market_row.get("fixture_id"))
        except (TypeError, ValueError):
            continue

        league = str(market_row.get("league", "") or "")
        home_team = str(market_row.get("home_team", "") or "")
        away_team = str(market_row.get("away_team", "") or "")
        consensus_row = find_consensus_row(league, home_team, away_team, consensus_rows)
        if consensus_row is None:
            unmatched.append(
                {
                    "league": league,
                    "home_team": home_team,
                    "away_team": away_team,
                }
            )
            continue

        output_rows.append(
            {
                "fixture_id": fixture_id,
                "market_name": str(market_row.get("name", "") or ""),
                "league": league,
                "home_team": home_team,
                "away_team": away_team,
                "sport_key": str(consensus_row.get("sport_key", "") or ""),
                "source_fixture_id": str(consensus_row.get("source_fixture_id", "") or ""),
                "consensus_home_team": str(consensus_row.get("home_team", "") or ""),
                "consensus_away_team": str(consensus_row.get("away_team", "") or ""),
                "commence_time": str(consensus_row.get("commence_time", "") or ""),
            }
        )
        counts_by_league[league] += 1

    write_output(output_rows)
    if unmatched:
        print("unmatched fixtures (up to 10):")
        for row in unmatched[:10]:
            print("%s | %s vs %s" % (row["league"], row["home_team"], row["away_team"]))
    print("total fixtures read: %d" % len(market_rows))
    print("total consensus rows available: %d" % len(consensus_rows))
    print("total mappings written: %d" % len(output_rows))
    print("total unmatched fixtures: %d" % len(unmatched))
    print("counts by league: %s" % json.dumps(dict(counts_by_league), sort_keys=True))


if __name__ == "__main__":
    main()
