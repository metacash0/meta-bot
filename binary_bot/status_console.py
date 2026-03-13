from __future__ import annotations

import json
import os
from typing import Any, Dict, List


OPEN_POSITIONS_PATH = "data/paper_open_positions.json"
SCAN_SUMMARY_PATH = "data/logs/scan_summary.jsonl"
SIGNAL_EVENTS_PATH = "data/logs/signal_events.jsonl"
PAPER_TRADES_PATH = "data/logs/paper_trades.jsonl"
PAPER_SETTLEMENTS_PATH = "data/logs/paper_settlements.jsonl"


def read_json_file(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def read_jsonl_tail(path: str, limit: int = 5) -> List[dict]:
    if not os.path.exists(path):
        return []
    rows: List[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except Exception:
        return []
    if limit <= 0:
        return []
    return rows[-limit:]


def sum_open_positions(positions_payload: dict) -> dict:
    positions = positions_payload.get("positions", []) if isinstance(positions_payload, dict) else []
    open_positions = 0
    open_yes_positions = 0
    open_no_positions = 0
    open_total_notional = 0.0
    open_yes_notional = 0.0
    open_no_notional = 0.0

    for row in positions:
        if not isinstance(row, dict):
            continue
        if str(row.get("status", "") or "") != "open":
            continue

        open_positions += 1
        side = str(row.get("side", "") or "").upper()
        try:
            total_notional = float(row.get("total_notional", 0.0) or 0.0)
        except (TypeError, ValueError):
            total_notional = 0.0

        open_total_notional += total_notional
        if side == "YES":
            open_yes_positions += 1
            open_yes_notional += total_notional
        elif side == "NO":
            open_no_positions += 1
            open_no_notional += total_notional

    return {
        "open_positions": open_positions,
        "open_yes_positions": open_yes_positions,
        "open_no_positions": open_no_positions,
        "open_total_notional": open_total_notional,
        "open_yes_notional": open_yes_notional,
        "open_no_notional": open_no_notional,
    }


def sum_realized_pnl(settlement_rows: List[dict]) -> dict:
    settled_trades = 0
    realized_gross_pnl = 0.0
    roi_values: List[float] = []

    for row in settlement_rows:
        if not isinstance(row, dict):
            continue
        settled_trades += 1
        try:
            realized_gross_pnl += float(row.get("gross_pnl", 0.0) or 0.0)
        except (TypeError, ValueError):
            pass
        try:
            roi_values.append(float(row.get("roi", 0.0) or 0.0))
        except (TypeError, ValueError):
            pass

    avg_roi = (sum(roi_values) / len(roi_values)) if roi_values else 0.0
    return {
        "settled_trades": settled_trades,
        "realized_gross_pnl": realized_gross_pnl,
        "avg_roi": avg_roi,
    }


def read_env_controls() -> dict:
    return {
        "TRADING_ENABLED": os.getenv("TRADING_ENABLED", "true").lower() == "true",
        "NEW_ENTRIES_ENABLED": os.getenv("NEW_ENTRIES_ENABLED", "true").lower() == "true",
        "SCALE_INS_ENABLED": os.getenv("SCALE_INS_ENABLED", "true").lower() == "true",
        "PREMATCH_WINDOW_HOURS": float(os.getenv("PREMATCH_WINDOW_HOURS", "6")),
    }


def _print_section(title: str) -> None:
    print("=== %s ===" % title)


def _print_json_block(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    controls = read_env_controls()
    open_positions_payload = read_json_file(OPEN_POSITIONS_PATH, {"positions": []})
    open_position_summary = sum_open_positions(open_positions_payload)
    latest_scan_rows = read_jsonl_tail(SCAN_SUMMARY_PATH, limit=1)
    latest_scan_summary = latest_scan_rows[-1] if latest_scan_rows else {}
    recent_signal_events = read_jsonl_tail(SIGNAL_EVENTS_PATH, limit=5)
    recent_paper_trades = read_jsonl_tail(PAPER_TRADES_PATH, limit=5)
    recent_paper_settlements = read_jsonl_tail(PAPER_SETTLEMENTS_PATH, limit=5)
    realized_pnl_summary = sum_realized_pnl(read_jsonl_tail(PAPER_SETTLEMENTS_PATH, limit=10_000))

    _print_section("Bot Controls")
    for key in (
        "TRADING_ENABLED",
        "NEW_ENTRIES_ENABLED",
        "SCALE_INS_ENABLED",
        "PREMATCH_WINDOW_HOURS",
    ):
        print("%s: %s" % (key, controls[key]))
    print()

    _print_section("Open Paper Positions")
    _print_json_block(open_position_summary)
    print()

    _print_section("Realized Paper PnL")
    _print_json_block(realized_pnl_summary)
    print()

    _print_section("Latest Scan Summary")
    _print_json_block(latest_scan_summary)
    print()

    _print_section("Recent Paper Trades")
    for row in recent_paper_trades:
        print("- %s" % json.dumps(row, sort_keys=True))
    if not recent_paper_trades:
        print("- none")
    print()

    _print_section("Recent Paper Settlements")
    for row in recent_paper_settlements:
        print("- %s" % json.dumps(row, sort_keys=True))
    if not recent_paper_settlements:
        print("- none")
    print()

    _print_section("Recent Signal Events")
    for row in recent_signal_events:
        print("- %s" % json.dumps(row, sort_keys=True))
    if not recent_signal_events:
        print("- none")


if __name__ == "__main__":
    main()
