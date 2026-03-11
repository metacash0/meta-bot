from __future__ import annotations

import json
from typing import Any, Dict, Optional


def clamp01(x: float) -> float:
    return float(min(0.99, max(0.01, x)))


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return int(default)


def soccer_home_win_probability(
    minute: int,
    score_home: int,
    score_away: int,
    red_home: int = 0,
    red_away: int = 0,
    prematch_home_win_prob: float = 0.45,
) -> float:
    p = float(prematch_home_win_prob)
    score_diff = int(score_home - score_away)
    minute_clamped = max(0, min(safe_int(minute, 0), 120))
    time_progress = float(minute_clamped) / 90.0
    time_amplifier = 0.55 + (0.95 * min(1.35, time_progress))

    p += float(score_diff) * 0.145 * time_amplifier
    p += float(safe_int(red_away, 0) - safe_int(red_home, 0)) * 0.095 * time_amplifier

    if score_diff > 0:
        p += 0.035 * min(1.0, time_progress)
    elif score_diff < 0:
        p -= 0.035 * min(1.0, time_progress)

    if minute_clamped >= 45:
        p += 0.02 * float(score_diff)
    if minute_clamped >= 75:
        p += 0.015 * float(score_diff)

    return clamp01(p)


def market_fair_probability_from_state(
    market_type: str,
    state: Dict[str, Any],
    prematch_home_win_prob: float = 0.45,
) -> Optional[float]:
    if str(market_type) != "home_win":
        return None

    return soccer_home_win_probability(
        minute=safe_int(state.get("minute", 0), 0),
        score_home=safe_int(state.get("score_home", 0), 0),
        score_away=safe_int(state.get("score_away", 0), 0),
        red_home=safe_int(state.get("red_home", 0), 0),
        red_away=safe_int(state.get("red_away", 0), 0),
        prematch_home_win_prob=float(prematch_home_win_prob),
    )


def load_market_map(path: str = "market_map.json") -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {"markets": []}

    if not isinstance(payload, dict):
        return {"markets": []}
    if not isinstance(payload.get("markets"), list):
        return {"markets": []}
    return payload


def find_market_config_for_asset(asset_id: str, path: str = "market_map.json") -> Optional[Dict[str, Any]]:
    payload = load_market_map(path=path)
    for market in payload.get("markets", []):
        if not isinstance(market, dict):
            continue
        yes_asset_id = str(market.get("yes_asset_id", "") or "")
        no_asset_id = str(market.get("no_asset_id", "") or "")
        if asset_id == yes_asset_id or asset_id == no_asset_id:
            return market
    return None
