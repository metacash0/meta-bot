from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from shared.live_match_state import get_live_match_state


OPEN_POSITIONS_PATH = "data/paper_open_positions.json"
SETTLED_TRADES_PATH = "data/logs/paper_settlements.jsonl"
FINISHED_STATUSES = {"FT", "AET", "PEN"}

os.makedirs("data", exist_ok=True)
os.makedirs("data/logs", exist_ok=True)


def read_open_positions() -> dict:
    if not os.path.exists(OPEN_POSITIONS_PATH):
        return {"positions": []}
    try:
        with open(OPEN_POSITIONS_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {"positions": []}
    if not isinstance(payload, dict):
        return {"positions": []}
    positions = payload.get("positions")
    if not isinstance(positions, list):
        return {"positions": []}
    return {"positions": [row for row in positions if isinstance(row, dict)]}


def write_open_positions(payload: dict) -> None:
    safe_payload = payload if isinstance(payload, dict) else {"positions": []}
    try:
        with open(OPEN_POSITIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(safe_payload, f, indent=2, sort_keys=True)
            f.write("\n")
    except Exception:
        pass


def append_jsonl(path: str, payload: dict) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
    except Exception:
        pass


def is_position_open(row: dict) -> bool:
    return str(row.get("status", "") or "") == "open"


def settlement_price_for_side(side: str, home_won: bool) -> float:
    side_value = str(side or "").upper()
    if side_value == "YES":
        return 1.0 if home_won else 0.0
    return 0.0 if home_won else 1.0


def compute_position_pnl(avg_price: float, shares: float, settlement_price: float) -> dict:
    avg_price_val = float(avg_price or 0.0)
    shares_val = float(shares or 0.0)
    settlement_price_val = float(settlement_price or 0.0)
    cost_basis = avg_price_val * shares_val
    settlement_value = settlement_price_val * shares_val
    gross_pnl = settlement_value - cost_basis
    roi = (gross_pnl / cost_basis) if cost_basis > 0.0 else 0.0
    return {
        "settlement_price": settlement_price_val,
        "gross_pnl": gross_pnl,
        "roi": roi,
    }


def get_final_match_result(fixture_id: int) -> Optional[dict]:
    state = get_live_match_state(int(fixture_id))
    status = str(state.get("status", "") or "")
    if status not in FINISHED_STATUSES:
        return None

    try:
        score_home = int(state.get("score_home", 0) or 0)
    except (TypeError, ValueError):
        score_home = 0
    try:
        score_away = int(state.get("score_away", 0) or 0)
    except (TypeError, ValueError):
        score_away = 0

    return {
        "fixture_id": int(fixture_id),
        "status": status,
        "status_long": str(state.get("status_long", "") or ""),
        "score_home": score_home,
        "score_away": score_away,
        "home_won": score_home > score_away,
    }


def settle_open_positions() -> dict:
    payload = read_open_positions()
    positions = payload.get("positions", [])
    positions_checked = 0
    positions_settled = 0
    updated_positions = []

    for row in positions:
        if not isinstance(row, dict):
            continue
        if not is_position_open(row):
            updated_positions.append(row)
            continue

        positions_checked += 1
        try:
            fixture_id = int(row.get("fixture_id"))
        except (TypeError, ValueError):
            updated_positions.append(row)
            continue

        try:
            result = get_final_match_result(fixture_id)
        except RuntimeError:
            updated_positions.append(row)
            continue

        if result is None:
            updated_positions.append(row)
            continue

        side = str(row.get("side", "") or "")
        try:
            avg_price = float(row.get("avg_price", 0.0) or 0.0)
        except (TypeError, ValueError):
            avg_price = 0.0
        try:
            total_shares = float(row.get("total_shares", 0.0) or 0.0)
        except (TypeError, ValueError):
            total_shares = 0.0
        try:
            total_notional = float(row.get("total_notional", 0.0) or 0.0)
        except (TypeError, ValueError):
            total_notional = 0.0
        try:
            entries = int(row.get("entries", 0) or 0)
        except (TypeError, ValueError):
            entries = 0

        settlement_price = settlement_price_for_side(side, bool(result["home_won"]))
        pnl = compute_position_pnl(avg_price, total_shares, settlement_price)
        closed_at = datetime.now(timezone.utc).isoformat()

        updated_row = dict(row)
        updated_row["status"] = "closed"
        updated_row["closed_at"] = closed_at
        updated_row["settlement_price"] = pnl["settlement_price"]
        updated_row["gross_pnl"] = pnl["gross_pnl"]
        updated_row["roi"] = pnl["roi"]
        updated_row["final_status"] = result["status"]
        updated_row["final_score_home"] = result["score_home"]
        updated_row["final_score_away"] = result["score_away"]
        updated_positions.append(updated_row)

        append_jsonl(
            SETTLED_TRADES_PATH,
            {
                "timestamp": closed_at,
                "event_type": "paper_settlement",
                "fixture_id": fixture_id,
                "market_name": str(row.get("market_name", "") or ""),
                "league": str(row.get("league", "") or ""),
                "side": side,
                "entries": entries,
                "total_notional": total_notional,
                "total_shares": total_shares,
                "avg_price": avg_price,
                "settlement_price": pnl["settlement_price"],
                "gross_pnl": pnl["gross_pnl"],
                "roi": pnl["roi"],
                "opened_at": row.get("opened_at"),
                "closed_at": closed_at,
                "final_status": result["status"],
                "final_score_home": result["score_home"],
                "final_score_away": result["score_away"],
            },
        )
        positions_settled += 1

    write_open_positions({"positions": updated_positions})
    return {
        "positions_checked": positions_checked,
        "positions_settled": positions_settled,
        "positions_still_open": max(0, positions_checked - positions_settled),
    }


def main() -> None:
    print(json.dumps(settle_open_positions(), sort_keys=True))


if __name__ == "__main__":
    main()
