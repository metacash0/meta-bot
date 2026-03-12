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
    min_edge: float = 0.03,
) -> dict:
    home_yes_fair = clamp_prob(float(fair_snapshot.get("home_yes_fair", 0.0) or 0.0))
    home_no_fair = clamp_prob(float(fair_snapshot.get("home_no_fair", 0.0) or 0.0))
    min_edge = float(min_edge)

    yes_mid = compute_binary_mid(yes_bid, yes_ask)
    no_mid = compute_binary_mid(no_bid, no_ask)

    yes_edge = home_yes_fair - yes_mid if yes_mid is not None else None
    no_edge = home_no_fair - no_mid if no_mid is not None else None

    action = "HOLD"
    side = None
    if yes_edge is not None and yes_edge >= min_edge and (no_edge is None or yes_edge >= no_edge):
        action = "BUY_YES"
        side = "YES"
    elif no_edge is not None and no_edge >= min_edge:
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
        "yes_mid": yes_mid,
        "no_bid": no_bid,
        "no_ask": no_ask,
        "no_mid": no_mid,
        "yes_edge": yes_edge,
        "no_edge": no_edge,
        "min_edge": min_edge,
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
                "yes_bid": 0.47,
                "yes_ask": 0.49,
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
                    "home_yes_fair": 0.42,
                    "home_no_fair": 0.58,
                },
                "no_bid": 0.49,
                "no_ask": 0.51,
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
                    "home_yes_fair": 0.51,
                    "home_no_fair": 0.49,
                },
                "yes_bid": 0.49,
                "yes_ask": 0.50,
                "no_bid": 0.49,
                "no_ask": 0.50,
                "min_edge": 0.03,
            },
        },
    ]

    for scenario in scenarios:
        print(scenario["label"])
        print(evaluate_home_win_market(**scenario["args"]))
