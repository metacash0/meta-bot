from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional


OPEN_POSITIONS_PATH = "data/paper_open_positions.json"
PAPER_TRADES_PATH = "data/logs/paper_trades.jsonl"

os.makedirs("data", exist_ok=True)
os.makedirs("data/logs", exist_ok=True)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_open_positions() -> dict:
    if not os.path.exists(OPEN_POSITIONS_PATH):
        return {"positions": []}

    try:
        with open(OPEN_POSITIONS_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {"positions": []}

    positions = payload.get("positions", []) if isinstance(payload, dict) else []
    if not isinstance(positions, list):
        return {"positions": []}
    return {"positions": [row for row in positions if isinstance(row, dict)]}


def write_open_positions(payload: dict) -> None:
    try:
        with open(OPEN_POSITIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=False)
            f.write("\n")
    except Exception:
        pass


def append_jsonl(path: str, payload: dict) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
    except Exception:
        pass


def position_key(fixture_id: int, side: str) -> str:
    return f"{int(fixture_id)}:{str(side or '')}"


def find_open_position(payload: dict, fixture_id: int, side: str) -> dict | None:
    positions = payload.get("positions", []) if isinstance(payload, dict) else []
    target_key = position_key(fixture_id, side)
    for row in positions:
        if not isinstance(row, dict):
            continue
        if str(row.get("position_key", "") or "") == target_key and str(row.get("status", "")) == "open":
            return row
    return None


def _trade_log_row(
    execution_reason: str,
    signal_snapshot: dict,
    entry_price: float,
    entry_edge: float,
    notional: float,
    shares: float,
    position: dict,
) -> dict:
    minute = signal_snapshot.get("minute")
    try:
        minute_val = int(minute) if minute is not None else None
    except (TypeError, ValueError):
        minute_val = None

    status = signal_snapshot.get("status")
    return {
        "timestamp": _utc_now_iso(),
        "event_type": "paper_entry",
        "execution_reason": execution_reason,
        "fixture_id": int(signal_snapshot.get("fixture_id")),
        "market_name": str(signal_snapshot.get("market_name", "") or ""),
        "league": str(signal_snapshot.get("league", "") or ""),
        "side": str(signal_snapshot.get("side", "") or ""),
        "entry_price": float(entry_price),
        "entry_edge": float(entry_edge),
        "notional": float(notional),
        "shares": float(shares),
        "position_entries": int(position.get("entries", 0)),
        "position_total_notional": float(position.get("total_notional", 0.0)),
        "position_total_shares": float(position.get("total_shares", 0.0)),
        "position_avg_price": float(position.get("avg_price", 0.0)),
        "match_status": str(status) if status is not None else None,
        "match_minute": minute_val,
        "rank_at_decision": signal_snapshot.get("rank_at_decision"),
        "priority_score": signal_snapshot.get("priority_score"),
        "edge_bucket": signal_snapshot.get("edge_bucket"),
        "minute_bucket": signal_snapshot.get("minute_bucket"),
        "is_live": signal_snapshot.get("is_live"),
        "is_prematch": signal_snapshot.get("is_prematch"),
        "spread": signal_snapshot.get("spread"),
        "ask_price": signal_snapshot.get("ask_price"),
        "ask_size": signal_snapshot.get("ask_size"),
        "candidate_count": signal_snapshot.get("candidate_count"),
        "was_top_ranked": signal_snapshot.get("was_top_ranked"),
        "entry_mode": execution_reason,
        "projected_open_total_notional_after": signal_snapshot.get("projected_open_total_notional_after"),
        "projected_open_positions_after": signal_snapshot.get("projected_open_positions_after"),
    }


def maybe_execute_paper_trade(
    signal_snapshot: dict,
    sizing_snapshot: dict,
    max_entries_per_market_side: int = 2,
    max_total_notional_per_market_side: float = 150.0,
    scale_in_edge_step: float = 0.01,
) -> dict:
    action = str(signal_snapshot.get("action", "") or "")
    if action == "HOLD":
        return {"executed": False, "reason": "no_action"}

    sizing_reason = str(sizing_snapshot.get("reason", "") or "")
    if sizing_reason != "ok":
        return {"executed": False, "reason": sizing_reason}

    recommended_notional = float(sizing_snapshot.get("recommended_notional", 0.0) or 0.0)
    recommended_shares = float(sizing_snapshot.get("recommended_shares", 0.0) or 0.0)
    if recommended_notional <= 0.0:
        return {"executed": False, "reason": "zero_size"}

    side = signal_snapshot.get("side")
    fixture_id = int(signal_snapshot.get("fixture_id"))
    market_name = str(signal_snapshot.get("market_name", "") or "")
    league = str(signal_snapshot.get("league", "") or "")

    if side == "YES":
        entry_price = float(signal_snapshot.get("yes_ask", 0.0) or 0.0)
        entry_edge = float(signal_snapshot.get("yes_edge", 0.0) or 0.0)
    elif side == "NO":
        entry_price = float(signal_snapshot.get("no_ask", 0.0) or 0.0)
        entry_edge = float(signal_snapshot.get("no_edge", 0.0) or 0.0)
    else:
        return {"executed": False, "reason": "no_action"}

    payload = read_open_positions()
    positions = payload.get("positions", [])
    open_position = find_open_position(payload, fixture_id, str(side))
    now_iso = _utc_now_iso()

    if open_position is None:
        new_position = {
            "position_key": position_key(fixture_id, str(side)),
            "fixture_id": fixture_id,
            "market_name": market_name,
            "league": league,
            "side": str(side),
            "entries": 1,
            "total_notional": float(recommended_notional),
            "total_shares": float(recommended_shares),
            "avg_price": float(entry_price),
            "last_entry_edge": float(entry_edge),
            "opened_at": now_iso,
            "updated_at": now_iso,
            "status": "open",
            "entry_rank_at_decision": signal_snapshot.get("rank_at_decision"),
            "entry_priority_score": signal_snapshot.get("priority_score"),
            "entry_edge": float(entry_edge),
            "entry_edge_bucket": signal_snapshot.get("edge_bucket"),
            "entry_mode": "new_entry",
        }
        positions.append(new_position)
        write_open_positions({"positions": positions})
        append_jsonl(
            PAPER_TRADES_PATH,
            _trade_log_row(
                execution_reason="new_entry",
                signal_snapshot=signal_snapshot,
                entry_price=entry_price,
                entry_edge=entry_edge,
                notional=recommended_notional,
                shares=recommended_shares,
                position=new_position,
            ),
        )
        return {
            "executed": True,
            "reason": "new_entry",
            "fixture_id": fixture_id,
            "side": side,
            "notional": float(recommended_notional),
            "shares": float(recommended_shares),
        }

    entries = int(open_position.get("entries", 0) or 0)
    if entries >= int(max_entries_per_market_side):
        return {"executed": False, "reason": "max_entries_reached"}

    current_total_notional = float(open_position.get("total_notional", 0.0) or 0.0)
    if current_total_notional >= float(max_total_notional_per_market_side):
        return {"executed": False, "reason": "max_notional_reached"}

    last_entry_edge = float(open_position.get("last_entry_edge", 0.0) or 0.0)
    if entry_edge < last_entry_edge + float(scale_in_edge_step):
        return {"executed": False, "reason": "scale_in_edge_not_improved"}

    capped_notional = min(
        recommended_notional,
        float(max_total_notional_per_market_side) - current_total_notional,
    )
    if capped_notional <= 0.0:
        return {"executed": False, "reason": "max_notional_reached"}

    capped_shares = capped_notional / entry_price if entry_price > 0.0 else 0.0
    new_total_notional = current_total_notional + capped_notional
    new_total_shares = float(open_position.get("total_shares", 0.0) or 0.0) + capped_shares

    open_position["entries"] = entries + 1
    open_position["total_notional"] = float(new_total_notional)
    open_position["total_shares"] = float(new_total_shares)
    open_position["avg_price"] = float(new_total_notional / new_total_shares) if new_total_shares > 0.0 else 0.0
    open_position["last_entry_edge"] = float(entry_edge)
    open_position["updated_at"] = now_iso
    open_position["entry_rank_at_decision"] = signal_snapshot.get("rank_at_decision")
    open_position["entry_priority_score"] = signal_snapshot.get("priority_score")
    open_position["entry_edge"] = float(entry_edge)
    open_position["entry_edge_bucket"] = signal_snapshot.get("edge_bucket")
    open_position["entry_mode"] = "scale_in"

    write_open_positions({"positions": positions})
    append_jsonl(
        PAPER_TRADES_PATH,
        _trade_log_row(
            execution_reason="scale_in",
            signal_snapshot=signal_snapshot,
            entry_price=entry_price,
            entry_edge=entry_edge,
            notional=capped_notional,
            shares=capped_shares,
            position=open_position,
        ),
    )
    return {
        "executed": True,
        "reason": "scale_in",
        "fixture_id": fixture_id,
        "side": side,
        "notional": float(capped_notional),
        "shares": float(capped_shares),
    }


if __name__ == "__main__":
    hold_signal = {
        "fixture_id": 1001,
        "market_name": "demo-hold",
        "league": "Premier League",
        "side": None,
        "action": "HOLD",
    }
    valid_signal = {
        "fixture_id": 1002,
        "market_name": "demo-buy-yes",
        "league": "Premier League",
        "side": "YES",
        "action": "BUY_YES",
        "yes_ask": 0.46,
        "yes_edge": 0.04,
        "status": "NS",
        "minute": 0,
    }
    weak_reentry_signal = {
        "fixture_id": 1002,
        "market_name": "demo-buy-yes",
        "league": "Premier League",
        "side": "YES",
        "action": "BUY_YES",
        "yes_ask": 0.47,
        "yes_edge": 0.045,
        "status": "NS",
        "minute": 0,
    }
    stronger_reentry_signal = {
        "fixture_id": 1002,
        "market_name": "demo-buy-yes",
        "league": "Premier League",
        "side": "YES",
        "action": "BUY_YES",
        "yes_ask": 0.45,
        "yes_edge": 0.06,
        "status": "NS",
        "minute": 0,
    }

    zero_size = {
        "reason": "below_min_notional",
        "recommended_notional": 0.0,
        "recommended_shares": 0.0,
    }
    sized_entry = {
        "reason": "ok",
        "recommended_notional": 50.64,
        "recommended_shares": 110.10,
    }
    sized_scale_in = {
        "reason": "ok",
        "recommended_notional": 40.00,
        "recommended_shares": 88.89,
    }

    print("A")
    print(maybe_execute_paper_trade(hold_signal, zero_size))
    print("B")
    print(maybe_execute_paper_trade(valid_signal, sized_entry))
    print("C")
    print(maybe_execute_paper_trade(weak_reentry_signal, sized_scale_in))
    print("D")
    print(maybe_execute_paper_trade(stronger_reentry_signal, sized_scale_in))
