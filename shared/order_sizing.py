from __future__ import annotations

import math


def clamp_positive(x: float | None) -> float:
    try:
        value = float(x)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(value) or value < 0.0:
        return 0.0
    return value


def recommend_order_size(
    action: str,
    side: str | None,
    ask_price: float | None,
    ask_size: float | None,
    edge: float | None,
    bankroll: float,
    max_risk_fraction: float = 0.02,
    max_book_share: float = 0.25,
    min_order_notional: float = 25.0,
    min_edge_for_full_size: float = 0.08,
) -> dict:
    ask_price_val = clamp_positive(ask_price)
    ask_size_val = clamp_positive(ask_size)
    edge_val = clamp_positive(edge)
    bankroll_val = clamp_positive(bankroll)
    max_risk_fraction_val = clamp_positive(max_risk_fraction)
    max_book_share_val = clamp_positive(max_book_share)
    min_order_notional_val = clamp_positive(min_order_notional)
    min_edge_for_full_size_val = clamp_positive(min_edge_for_full_size)

    result = {
        "action": str(action or ""),
        "side": side,
        "edge": edge,
        "ask_price": ask_price,
        "ask_size": ask_size,
        "bankroll": bankroll_val,
        "risk_cap_notional": 0.0,
        "book_cap_shares": 0.0,
        "book_cap_notional": 0.0,
        "edge_scale": 0.0,
        "target_notional": 0.0,
        "recommended_notional": 0.0,
        "recommended_shares": 0.0,
        "reason": "",
    }

    if str(action or "") == "HOLD" or side is None:
        result["reason"] = "no_action"
        return result
    if ask_price_val <= 0.0:
        result["reason"] = "missing_price"
        return result
    if ask_size_val <= 0.0:
        result["reason"] = "missing_size"
        return result
    if edge_val <= 0.0:
        result["reason"] = "no_edge"
        return result
    if bankroll_val <= 0.0:
        result["reason"] = "invalid_bankroll"
        return result

    risk_cap_notional = bankroll_val * max_risk_fraction_val
    book_cap_shares = ask_size_val * max_book_share_val
    book_cap_notional = book_cap_shares * ask_price_val
    edge_scale = 1.0
    if min_edge_for_full_size_val > 0.0:
        edge_scale = min(1.0, edge_val / min_edge_for_full_size_val)

    target_notional = risk_cap_notional * edge_scale
    executable_notional = min(target_notional, book_cap_notional)
    recommended_shares = executable_notional / ask_price_val if ask_price_val > 0.0 else 0.0

    result.update(
        {
            "risk_cap_notional": float(risk_cap_notional),
            "book_cap_shares": float(book_cap_shares),
            "book_cap_notional": float(book_cap_notional),
            "edge_scale": float(edge_scale),
            "target_notional": float(target_notional),
            "recommended_notional": float(executable_notional),
            "recommended_shares": float(recommended_shares),
        }
    )

    if executable_notional < min_order_notional_val:
        result["recommended_notional"] = 0.0
        result["recommended_shares"] = 0.0
        result["reason"] = "below_min_notional"
        return result

    result["reason"] = "ok"
    return result


def extract_relevant_side_inputs(signal_snapshot: dict) -> dict:
    action = str(signal_snapshot.get("action", "") or "")
    if action == "BUY_YES":
        return {
            "ask_price": signal_snapshot.get("yes_ask"),
            "ask_size": signal_snapshot.get("yes_ask_size"),
            "edge": signal_snapshot.get("yes_edge"),
        }
    if action == "BUY_NO":
        return {
            "ask_price": signal_snapshot.get("no_ask"),
            "ask_size": signal_snapshot.get("no_ask_size"),
            "edge": signal_snapshot.get("no_edge"),
        }
    return {
        "ask_price": None,
        "ask_size": None,
        "edge": None,
    }


def size_from_signal_snapshot(
    signal_snapshot: dict,
    bankroll: float,
    max_risk_fraction: float = 0.02,
    max_book_share: float = 0.25,
    min_order_notional: float = 25.0,
    min_edge_for_full_size: float = 0.08,
) -> dict:
    side_inputs = extract_relevant_side_inputs(signal_snapshot)
    return recommend_order_size(
        action=str(signal_snapshot.get("action", "") or ""),
        side=signal_snapshot.get("side"),
        ask_price=side_inputs.get("ask_price"),
        ask_size=side_inputs.get("ask_size"),
        edge=side_inputs.get("edge"),
        bankroll=bankroll,
        max_risk_fraction=max_risk_fraction,
        max_book_share=max_book_share,
        min_order_notional=min_order_notional,
        min_edge_for_full_size=min_edge_for_full_size,
    )


if __name__ == "__main__":
    scenarios = [
        {
            "label": "A",
            "signal_snapshot": {"action": "HOLD", "side": None},
            "bankroll": 1000.0,
        },
        {
            "label": "B",
            "signal_snapshot": {
                "action": "BUY_YES",
                "side": "YES",
                "yes_ask": 0.48,
                "yes_ask_size": 20.0,
                "yes_edge": 0.06,
            },
            "bankroll": 1000.0,
        },
        {
            "label": "C",
            "signal_snapshot": {
                "action": "BUY_NO",
                "side": "NO",
                "no_ask": 0.42,
                "no_ask_size": 200.0,
                "no_edge": 0.10,
            },
            "bankroll": 1000.0,
        },
    ]

    for scenario in scenarios:
        print(scenario["label"])
        print(
            size_from_signal_snapshot(
                signal_snapshot=scenario["signal_snapshot"],
                bankroll=scenario["bankroll"],
            )
        )
