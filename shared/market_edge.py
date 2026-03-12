from __future__ import annotations


def clamp_prob(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _valid_prob(value: float | None) -> bool:
    if value is None:
        return False
    try:
        val = float(value)
    except (TypeError, ValueError):
        return False
    return 0.0 <= val <= 1.0


def compute_binary_mid(bid: float | None, ask: float | None) -> float | None:
    bid_val = float(bid) if _valid_prob(bid) else None
    ask_val = float(ask) if _valid_prob(ask) else None

    if bid_val is not None and ask_val is not None:
        return (bid_val + ask_val) / 2.0
    if bid_val is not None:
        return bid_val
    if ask_val is not None:
        return ask_val
    return None


def evaluate_home_win_market(
    fair_snapshot: dict,
    yes_bid: float | None = None,
    yes_ask: float | None = None,
    no_bid: float | None = None,
    no_ask: float | None = None,
    yes_bid_size: float | None = None,
    yes_ask_size: float | None = None,
    no_bid_size: float | None = None,
    no_ask_size: float | None = None,
    min_edge: float = 0.03,
    max_spread: float = 0.04,
    min_top_size: float = 25.0,
    cost_buffer: float = 0.01,
) -> dict:
    home_yes_fair = clamp_prob(float(fair_snapshot.get("home_yes_fair", 0.0) or 0.0))
    home_no_fair = clamp_prob(float(fair_snapshot.get("home_no_fair", 0.0) or 0.0))
    min_edge = float(min_edge)
    max_spread = float(max_spread)
    min_top_size = float(min_top_size)
    cost_buffer = float(cost_buffer)
    effective_min_edge = min_edge + cost_buffer

    yes_mid = compute_binary_mid(yes_bid, yes_ask)
    no_mid = compute_binary_mid(no_bid, no_ask)
    yes_spread = (
        float(yes_ask) - float(yes_bid)
        if _valid_prob(yes_bid) and _valid_prob(yes_ask)
        else None
    )
    no_spread = (
        float(no_ask) - float(no_bid)
        if _valid_prob(no_bid) and _valid_prob(no_ask)
        else None
    )

    yes_edge = home_yes_fair - yes_mid if yes_mid is not None else None
    no_edge = home_no_fair - no_mid if no_mid is not None else None
    yes_tradable = (
        yes_spread is not None
        and yes_spread <= max_spread
        and yes_ask_size is not None
        and float(yes_ask_size) >= min_top_size
    )
    no_tradable = (
        no_spread is not None
        and no_spread <= max_spread
        and no_ask_size is not None
        and float(no_ask_size) >= min_top_size
    )

    action = "HOLD"
    side = None
    yes_qualifies = yes_tradable and yes_edge is not None and yes_edge >= effective_min_edge
    no_qualifies = no_tradable and no_edge is not None and no_edge >= effective_min_edge
    if yes_qualifies and (not no_qualifies or float(yes_edge) >= float(no_edge)):
        action = "BUY_YES"
        side = "YES"
    elif no_qualifies:
        action = "BUY_NO"
        side = "NO"

    return {
        "fixture_id": fair_snapshot.get("fixture_id"),
        "home_team": fair_snapshot.get("home_team"),
        "away_team": fair_snapshot.get("away_team"),
        "home_yes_fair": home_yes_fair,
        "home_no_fair": home_no_fair,
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "yes_bid_size": yes_bid_size,
        "yes_ask_size": yes_ask_size,
        "yes_mid": yes_mid,
        "yes_spread": yes_spread,
        "no_bid": no_bid,
        "no_ask": no_ask,
        "no_bid_size": no_bid_size,
        "no_ask_size": no_ask_size,
        "no_mid": no_mid,
        "no_spread": no_spread,
        "yes_edge": yes_edge,
        "no_edge": no_edge,
        "min_edge": min_edge,
        "max_spread": max_spread,
        "min_top_size": min_top_size,
        "cost_buffer": cost_buffer,
        "effective_min_edge": effective_min_edge,
        "yes_tradable": yes_tradable,
        "no_tradable": no_tradable,
        "action": action,
        "side": side,
    }


if __name__ == "__main__":
    scenarios = [
        {
            "label": "A",
            "args": {
                "fair_snapshot": {
                    "fixture_id": 1,
                    "home_team": "Home A",
                    "away_team": "Away A",
                    "home_yes_fair": 0.56,
                    "home_no_fair": 0.44,
                },
                "yes_bid": 0.45,
                "yes_ask": 0.52,
                "yes_ask_size": 100.0,
                "min_edge": 0.03,
            },
        },
        {
            "label": "B",
            "args": {
                "fair_snapshot": {
                    "fixture_id": 2,
                    "home_team": "Home B",
                    "away_team": "Away B",
                    "home_yes_fair": 0.57,
                    "home_no_fair": 0.43,
                },
                "yes_bid": 0.47,
                "yes_ask": 0.49,
                "yes_ask_size": 10.0,
                "min_edge": 0.03,
            },
        },
        {
            "label": "C",
            "args": {
                "fair_snapshot": {
                    "fixture_id": 3,
                    "home_team": "Home C",
                    "away_team": "Away C",
                    "home_yes_fair": 0.56,
                    "home_no_fair": 0.44,
                },
                "yes_bid": 0.47,
                "yes_ask": 0.49,
                "yes_ask_size": 50.0,
                "min_edge": 0.03,
            },
        },
    ]

    for scenario in scenarios:
        print(scenario["label"])
        print(evaluate_home_win_market(**scenario["args"]))
