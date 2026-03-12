from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import Counter
from typing import Any, Dict, List

from shared.prematch_lambda_fit import fit_lambdas_from_1x2_and_total
from shared.poisson_totals import infer_lambda_total


MARKET_MAP_PATH = "data/market_map.json"
CONSENSUS_PATH = "data/sportsbook_consensus.json"
OUTPUT_PATH = "data/prematch_lambdas.json"
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
    "borussia m nchengladbach": "monchengladbach",
    "monchengladbach": "monchengladbach",
    "fc st pauli 1910": "st pauli",
    "st pauli": "st pauli",
    "tsg 1899 hoffenheim": "hoffenheim",
    "1899 hoffenheim": "hoffenheim",
    "hoffenheim": "hoffenheim",
    "vfl wolfsburg": "wolfsburg",
    "wolfsburg": "wolfsburg",
    "athletic club": "athletic club",
    "athletic bilbao": "athletic club",
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
    "bayern munchen": "bayern munich",
    "bayern munich": "bayern munich",
    "fc lorient": "lorient",
    "lorient": "lorient",
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
    "sporting clube de portugal": "sporting cp",
    "sporting cp": "sporting cp",
    "sporting": "sporting cp",
    "bodo glimt": "bodo glimt",
    "bodo glimt": "bodo glimt",
}


def write_output(rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"fixtures": rows}, f, indent=2, sort_keys=False)
        f.write("\n")


def read_market_map() -> List[Dict[str, Any]]:
    if not os.path.exists(MARKET_MAP_PATH):
        return []

    try:
        with open(MARKET_MAP_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []

    markets = payload.get("markets", []) if isinstance(payload, dict) else []
    if not isinstance(markets, list):
        return []
    return [row for row in markets if isinstance(row, dict)]


def read_sportsbook_consensus() -> List[Dict[str, Any]]:
    if not os.path.exists(CONSENSUS_PATH):
        return []

    try:
        with open(CONSENSUS_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []

    fixtures = payload.get("fixtures", []) if isinstance(payload, dict) else []
    if not isinstance(fixtures, list):
        return []
    return [row for row in fixtures if isinstance(row, dict)]


def normalize_team_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
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

    fallback_match = None
    for row in rows:
        row_home = canonical_team_name(str(row.get("home_team", "") or ""))
        row_away = canonical_team_name(str(row.get("away_team", "") or ""))
        if row_home != target_home or row_away != target_away:
            continue

        row_sport_key = str(row.get("sport_key", "") or "")
        if target_sport_key and row_sport_key == target_sport_key:
            return row
        if fallback_match is None:
            fallback_match = row

    return fallback_match


def main() -> None:
    markets = read_market_map()
    consensus_rows = read_sportsbook_consensus()
    if not markets or not consensus_rows:
        if not markets:
            print("market map missing or unreadable at data/market_map.json; wrote empty prematch_lambdas.json")
        else:
            print("sportsbook consensus missing or unreadable at data/sportsbook_consensus.json; wrote empty prematch_lambdas.json")
        write_output([])
        print("total fixtures read: 0")
        print("total consensus rows available: 0")
        print("total lambda rows written: 0")
        print("total unmatched fixtures: 0")
        print("counts by league: {}")
        return

    output_rows: List[Dict[str, Any]] = []
    counts_by_league: Counter = Counter()
    unmatched: List[Dict[str, str]] = []

    for market in markets:
        fixture_id = market.get("fixture_id")
        try:
            fixture_id = int(fixture_id)
        except (TypeError, ValueError):
            continue

        name = str(market.get("name", "") or "")
        league = str(market.get("league", "") or "")
        home_team = str(market.get("home_team", "") or "")
        away_team = str(market.get("away_team", "") or "")

        consensus = find_consensus_row(
            league=league,
            home_team=home_team,
            away_team=away_team,
            rows=consensus_rows,
        )
        if consensus is None:
            unmatched.append(
                {
                    "league": league,
                    "home_team": home_team,
                    "away_team": away_team,
                }
            )
            continue

        try:
            lambda_total_result = infer_lambda_total(
                total_point=float(consensus["total_point"]),
                p_over=float(consensus["p_over"]),
            )
            fit = fit_lambdas_from_1x2_and_total(
                p_home=float(consensus["p_home"]),
                p_draw=float(consensus["p_draw"]),
                p_away=float(consensus["p_away"]),
                lambda_total=float(lambda_total_result["lambda_total"]),
            )
        except Exception:
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
                "name": name,
                "league": league,
                "home_team": home_team,
                "away_team": away_team,
                "lambda_home": float(fit["lambda_home"]),
                "lambda_away": float(fit["lambda_away"]),
                "lambda_total": float(fit["lambda_total"]),
                "target_home_prob": float(fit["target_home_prob"]),
                "target_draw_prob": float(fit["target_draw_prob"]),
                "target_away_prob": float(fit["target_away_prob"]),
                "fit_home_prob": float(fit["fit_home_prob"]),
                "fit_draw_prob": float(fit["fit_draw_prob"]),
                "fit_away_prob": float(fit["fit_away_prob"]),
                "fit_error": float(fit["fit_error"]),
            }
        )
        counts_by_league[league] += 1

    write_output(output_rows)
    if unmatched:
        print("unmatched fixtures (up to 10):")
        for row in unmatched[:10]:
            print("%s | %s vs %s" % (row["league"], row["home_team"], row["away_team"]))
    print("total fixtures read: %d" % len(markets))
    print("total consensus rows available: %d" % len(consensus_rows))
    print("total lambda rows written: %d" % len(output_rows))
    print("total unmatched fixtures: %d" % len(unmatched))
    print("counts by league: %s" % json.dumps(dict(counts_by_league), sort_keys=True))


if __name__ == "__main__":
    main()
