from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List


RAW_PATH = "data/sportsbook_odds_raw.json"
OUTPUT_PATH = "data/sportsbook_consensus.json"


def decimal_odds_to_prob(price: float) -> float:
    price = float(price)
    if price <= 0.0:
        raise ValueError("decimal odds must be positive")
    return 1.0 / price


def normalize_probs(values: list[float]) -> list[float]:
    cleaned = [max(0.0, float(value)) for value in values]
    total = sum(cleaned)
    if total <= 0.0:
        raise ValueError("probabilities sum to zero")
    return [value / total for value in cleaned]


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_raw_fixtures() -> List[Dict[str, Any]]:
    if not os.path.exists(RAW_PATH):
        return []

    try:
        with open(RAW_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []

    fixtures = payload.get("fixtures", []) if isinstance(payload, dict) else []
    if not isinstance(fixtures, list):
        return []
    return [fixture for fixture in fixtures if isinstance(fixture, dict)]


def extract_h2h_probs(fixture: dict) -> list[dict]:
    home_team = str(fixture.get("home_team", "") or "")
    away_team = str(fixture.get("away_team", "") or "")
    bookmakers = fixture.get("bookmakers", [])
    if not isinstance(bookmakers, list):
        return []

    rows: List[dict] = []
    for bookmaker in bookmakers:
        if not isinstance(bookmaker, dict):
            continue
        markets = bookmaker.get("markets", [])
        if not isinstance(markets, list):
            continue

        for market in markets:
            if not isinstance(market, dict) or market.get("key") != "h2h":
                continue
            outcomes = market.get("outcomes", [])
            if not isinstance(outcomes, list):
                continue

            prices: Dict[str, float] = {}
            for outcome in outcomes:
                if not isinstance(outcome, dict):
                    continue
                name = str(outcome.get("name", "") or "")
                price = _safe_float(outcome.get("price"))
                if price is None or price <= 0.0:
                    continue
                if name == home_team:
                    prices["home"] = decimal_odds_to_prob(price)
                elif name == away_team:
                    prices["away"] = decimal_odds_to_prob(price)
                elif name == "Draw":
                    prices["draw"] = decimal_odds_to_prob(price)

            if {"home", "draw", "away"} - set(prices):
                continue

            p_home, p_draw, p_away = normalize_probs(
                [prices["home"], prices["draw"], prices["away"]]
            )
            rows.append(
                {
                    "bookmaker": str(bookmaker.get("key", bookmaker.get("title", "")) or ""),
                    "p_home": p_home,
                    "p_draw": p_draw,
                    "p_away": p_away,
                }
            )
            break

    return rows


def extract_totals_probs(fixture: dict) -> list[dict]:
    bookmakers = fixture.get("bookmakers", [])
    if not isinstance(bookmakers, list):
        return []

    rows: List[dict] = []
    for bookmaker in bookmakers:
        if not isinstance(bookmaker, dict):
            continue
        markets = bookmaker.get("markets", [])
        if not isinstance(markets, list):
            continue

        for market in markets:
            if not isinstance(market, dict) or market.get("key") != "totals":
                continue
            outcomes = market.get("outcomes", [])
            if not isinstance(outcomes, list):
                continue

            grouped: Dict[float, Dict[str, float]] = defaultdict(dict)
            for outcome in outcomes:
                if not isinstance(outcome, dict):
                    continue
                point = _safe_float(outcome.get("point"))
                price = _safe_float(outcome.get("price"))
                name = str(outcome.get("name", "") or "")
                if point is None or price is None or price <= 0.0:
                    continue
                if name == "Over":
                    grouped[point]["over"] = decimal_odds_to_prob(price)
                elif name == "Under":
                    grouped[point]["under"] = decimal_odds_to_prob(price)

            bookmaker_name = str(bookmaker.get("key", bookmaker.get("title", "")) or "")
            for point, probs in grouped.items():
                if "over" not in probs or "under" not in probs:
                    continue
                p_over, p_under = normalize_probs([probs["over"], probs["under"]])
                rows.append(
                    {
                        "bookmaker": bookmaker_name,
                        "point": float(point),
                        "p_over": p_over,
                        "p_under": p_under,
                    }
                )
            break

    return rows


def choose_canonical_total_point(rows: list[dict]) -> float | None:
    if not rows:
        return None
    counts: Counter = Counter()
    for row in rows:
        point = _safe_float(row.get("point"))
        if point is not None:
            counts[float(point)] += 1
    if not counts:
        return None
    best_count = max(counts.values())
    candidates = [point for point, count in counts.items() if count == best_count]
    return min(candidates)


def build_fixture_consensus(fixture: dict) -> dict | None:
    h2h_rows = extract_h2h_probs(fixture)
    totals_rows = extract_totals_probs(fixture)
    canonical_point = choose_canonical_total_point(totals_rows)
    if not h2h_rows or canonical_point is None:
        return None

    canonical_totals = [
        row for row in totals_rows if _safe_float(row.get("point")) == float(canonical_point)
    ]
    if not canonical_totals:
        return None

    p_home = sum(float(row["p_home"]) for row in h2h_rows) / float(len(h2h_rows))
    p_draw = sum(float(row["p_draw"]) for row in h2h_rows) / float(len(h2h_rows))
    p_away = sum(float(row["p_away"]) for row in h2h_rows) / float(len(h2h_rows))
    p_home, p_draw, p_away = normalize_probs([p_home, p_draw, p_away])

    p_over = sum(float(row["p_over"]) for row in canonical_totals) / float(len(canonical_totals))
    p_under = sum(float(row["p_under"]) for row in canonical_totals) / float(len(canonical_totals))
    p_over, p_under = normalize_probs([p_over, p_under])

    return {
        "source_fixture_id": str(fixture.get("id", "") or ""),
        "sport_key": str(
            fixture.get("_source_sport_key", fixture.get("sport_key", fixture.get("sport", ""))) or ""
        ),
        "home_team": str(fixture.get("home_team", "") or ""),
        "away_team": str(fixture.get("away_team", "") or ""),
        "commence_time": str(fixture.get("commence_time", "") or ""),
        "book_count_h2h": int(len(h2h_rows)),
        "book_count_totals": int(len(canonical_totals)),
        "p_home": float(p_home),
        "p_draw": float(p_draw),
        "p_away": float(p_away),
        "total_point": float(canonical_point),
        "p_over": float(p_over),
        "p_under": float(p_under),
    }


def _write_output(rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    payload = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "fixtures": rows,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")


def main() -> None:
    raw_fixtures = _read_raw_fixtures()
    consensus_rows: List[Dict[str, Any]] = []
    counts_by_sport_key: Counter = Counter()

    for fixture in raw_fixtures:
        consensus = build_fixture_consensus(fixture)
        if consensus is None:
            continue
        consensus_rows.append(consensus)
        counts_by_sport_key[str(consensus.get("sport_key", "") or "")] += 1

    _write_output(consensus_rows)
    print("total raw fixtures: %d" % len(raw_fixtures))
    print("total consensus fixtures written: %d" % len(consensus_rows))
    print("counts by sport_key: %s" % json.dumps(dict(counts_by_sport_key), sort_keys=True))


if __name__ == "__main__":
    main()
