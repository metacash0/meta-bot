from __future__ import annotations

import json
import os
from typing import Any, Dict


OPEN_POSITIONS_PATH = "data/paper_open_positions.json"


def read_open_positions() -> Dict[str, Any]:
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


def write_open_positions(payload: Dict[str, Any]) -> None:
    os.makedirs("data", exist_ok=True)
    with open(OPEN_POSITIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def reset_paper_state() -> Dict[str, Any]:
    previous_payload = read_open_positions()
    previous_positions = previous_payload.get("positions", [])
    previous_open_positions = 0
    previous_open_notional = 0.0

    for row in previous_positions:
        if not isinstance(row, dict):
            continue
        if str(row.get("status", "") or "") != "open":
            continue
        previous_open_positions += 1
        try:
            previous_open_notional += float(
                row.get("position_total_notional", row.get("total_notional", 0.0)) or 0.0
            )
        except (TypeError, ValueError):
            pass

    reset_payload = {"positions": []}
    write_open_positions(reset_payload)

    return {
        "reset": True,
        "open_positions_path": OPEN_POSITIONS_PATH,
        "previous_open_positions": previous_open_positions,
        "previous_open_notional": previous_open_notional,
        "current_open_positions": 0,
        "current_open_notional": 0.0,
        "historical_logs_preserved": True,
    }


def main() -> None:
    print(json.dumps(reset_paper_state(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
