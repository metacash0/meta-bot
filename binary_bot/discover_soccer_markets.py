from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import requests


GAMMA_URL = "https://gamma-api.polymarket.com/markets"

LEAGUE_TAGS = {
    82: "Premier League",
    306: "EPL",
    780: "La Liga",
    1494: "Bundesliga",
    102070: "Ligue 1",
    101962: "Serie A",
    100977: "Champions League",
    101787: "Europa League",
    102561: "Argentina",
    102448: "Liga MX",
    101735: "Eredivisie",
    102564: "Turkey",
    102594: "FA Cup",
    102595: "EFL",
    102008: "Coppa Italia",
    102562: "Libertadores",
    102563: "Sudamericana",
}


def parse_home_win_question(q: str) -> Optional[Dict[str, str]]:
    match = re.match(r"^Will\s+(.+?)\s+win on\s+(\d{4}-\d{2}-\d{2})\?$", q.strip(), re.IGNORECASE)
    if not match:
        return None
    return {
        "market_kind": "home_win",
        "home_team": match.group(1).strip(),
        "away_team": "",
        "match_date": match.group(2).strip(),
    }


def parse_draw_question(q: str) -> Optional[Dict[str, str]]:
    match = re.match(
        r"^Will\s+(.+?)\s+vs\.\s+(.+?)\s+end in a draw\?$",
        q.strip(),
        re.IGNORECASE,
    )
    if not match:
        return None
    return {
        "market_kind": "draw",
        "home_team": match.group(1).strip(),
        "away_team": match.group(2).strip(),
        "match_date": "",
    }


def parse_clob_token_ids(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            return []
        if isinstance(decoded, list):
            return [str(item) for item in decoded if item is not None]
    return []


def extract_yes_no_tokens(row: Dict[str, Any]) -> Tuple[str, str]:
    token_ids = parse_clob_token_ids(row.get("clobTokenIds"))
    if len(token_ids) >= 2:
        return token_ids[0], token_ids[1]

    tokens = row.get("tokens")
    if isinstance(tokens, list):
        extracted: List[str] = []
        for token in tokens:
            if not isinstance(token, dict):
                continue
            for key in ("token_id", "tokenId", "asset_id", "assetId", "id"):
                value = token.get(key)
                if value is not None:
                    extracted.append(str(value))
                    break
        if len(extracted) >= 2:
            return extracted[0], extracted[1]

    return "", ""


def extract_tag_slugs(row: Dict[str, Any]) -> List[str]:
    tags = row.get("tags")
    if not isinstance(tags, list):
        return []

    slugs: List[str] = []
    for tag in tags:
        if not isinstance(tag, dict):
            continue
        slug = tag.get("slug")
        if slug is not None:
            slugs.append(str(slug).strip().lower())
    return slugs


def is_true_soccer_match_market(row: Dict[str, Any], question: str) -> bool:
    slugs = set(extract_tag_slugs(row))
    if not {"sports", "soccer", "games"}.issubset(slugs):
        return False

    game_start_time = row.get("gameStartTime")
    if game_start_time in (None, ""):
        return False

    q = question.strip().lower()
    if "qualify" in q:
        return False
    if "win the" in q:
        return False
    if "world cup" in q:
        return False
    if "champions league" in q and game_start_time in (None, ""):
        return False

    return True


def discover_for_league(tag_id: int, league_name: str) -> Tuple[List[Dict[str, str]], int]:
    params = {
        "tag_id": int(tag_id),
        "closed": "false",
        "limit": 100,
        "include_tag": "true",
    }

    try:
        response = requests.get(GAMMA_URL, params=params, timeout=15.0)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return [], 0

    if not isinstance(payload, list):
        return [], 0

    scanned = 0
    discovered: List[Dict[str, str]] = []

    for row in payload:
        if not isinstance(row, dict):
            continue
        scanned += 1

        question = str(row.get("question", "") or "").strip()
        if not question:
            continue
        if not is_true_soccer_match_market(row, question):
            continue

        parsed = parse_home_win_question(question)
        if parsed is None:
            parsed = parse_draw_question(question)
        if parsed is None:
            continue

        yes_token, no_token = extract_yes_no_tokens(row)
        discovered.append(
            {
                "league": league_name,
                "question": question,
                "slug": str(row.get("slug", "") or ""),
                "market_kind": parsed["market_kind"],
                "home_team": parsed["home_team"],
                "away_team": parsed["away_team"],
                "match_date": parsed["match_date"],
                "gameStartTime": str(row.get("gameStartTime", "") or ""),
                "yes_token": yes_token,
                "no_token": no_token,
            }
        )

    return discovered, scanned


def main() -> None:
    all_markets: List[Dict[str, str]] = []
    total_scanned = 0
    counts_by_league: Counter = Counter()
    counts_by_market_kind: Counter = Counter()

    for tag_id, league_name in LEAGUE_TAGS.items():
        rows, scanned = discover_for_league(tag_id, league_name)
        total_scanned += scanned
        all_markets.extend(rows)
        counts_by_league[league_name] += len(rows)
        for row in rows:
            counts_by_market_kind[row["market_kind"]] += 1

    print(json.dumps(all_markets, indent=2, sort_keys=True))
    print("total rows scanned: %d" % total_scanned)
    print("total soccer match markets kept: %d" % len(all_markets))
    print("counts by league: %s" % json.dumps(dict(counts_by_league), sort_keys=True))
    print("counts by market_kind: %s" % json.dumps(dict(counts_by_market_kind), sort_keys=True))


if __name__ == "__main__":
    main()
