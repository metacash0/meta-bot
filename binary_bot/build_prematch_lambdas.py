from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any, Dict, List

from shared.prematch_lambda_fit import fit_lambdas_from_1x2_and_total


MARKET_MAP_PATH = "data/market_map.json"
OUTPUT_PATH = "data/prematch_lambdas.json"
BIG_HOME_CLUBS = (
    "manchester city",
    "liverpool",
    "arsenal",
    "bayern",
    "inter",
    "napoli",
    "barcelona",
    "real madrid",
)


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


def get_stub_prematch_inputs(league: str, home_team: str, away_team: str) -> dict:
    _ = str(league or "")
    _ = str(away_team or "")
    home_text = str(home_team or "").lower()

    if any(club in home_text for club in BIG_HOME_CLUBS):
        return {
            "p_home": 0.58,
            "p_draw": 0.23,
            "p_away": 0.19,
            "lambda_total": 2.85,
        }

    return {
        "p_home": 0.44,
        "p_draw": 0.27,
        "p_away": 0.29,
        "lambda_total": 2.55,
    }


def main() -> None:
    markets = read_market_map()
    if not markets:
        print("market map missing or unreadable at data/market_map.json; wrote empty prematch_lambdas.json")
        write_output([])
        print("total fixtures read: 0")
        print("total lambda rows written: 0")
        print("counts by league: {}")
        return

    output_rows: List[Dict[str, Any]] = []
    counts_by_league: Counter = Counter()

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

        stub_inputs = get_stub_prematch_inputs(
            league=league,
            home_team=home_team,
            away_team=away_team,
        )

        try:
            fit = fit_lambdas_from_1x2_and_total(
                p_home=stub_inputs["p_home"],
                p_draw=stub_inputs["p_draw"],
                p_away=stub_inputs["p_away"],
                lambda_total=stub_inputs["lambda_total"],
            )
        except Exception:
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
    print("total fixtures read: %d" % len(markets))
    print("total lambda rows written: %d" % len(output_rows))
    print("counts by league: %s" % json.dumps(dict(counts_by_league), sort_keys=True))


if __name__ == "__main__":
    main()
