from __future__ import annotations

import json
import os
import time
from collections import Counter
from typing import Any, Dict, List, Optional


LOG_DIR = "binary_bot/logs"


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []

    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def safe_last(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    return rows[-1]


def summarize_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    counter = Counter()
    signal_reasons = Counter()
    signal_actions = Counter()
    risk_reasons = Counter()

    bot_start = None
    bot_stop = None

    for row in events:
        event_type = row.get("event_type", "unknown")
        counter[event_type] += 1

        payload = row.get("payload", {})

        if event_type == "signal":
            signal_actions[payload.get("action", "unknown")] += 1
            signal_reasons[payload.get("reason", "unknown")] += 1

        if event_type == "risk_check":
            risk_reasons[payload.get("reason", "unknown")] += 1

        if event_type == "bot_start":
            bot_start = row

        if event_type == "bot_stop":
            bot_stop = row

    return {
        "event_counts": counter,
        "signal_actions": signal_actions,
        "signal_reasons": signal_reasons,
        "risk_reasons": risk_reasons,
        "heartbeat_count": int(counter.get("heartbeat", 0)),
        "stale_warning_count": int(counter.get("stale_feed_warning", 0)),
        "bot_start": bot_start,
        "bot_stop": bot_stop,
    }


def summarize_orders(orders: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not orders:
        return {
            "submitted_count": 0,
            "side_counts": Counter(),
            "avg_size": 0.0,
        }

    side_counts = Counter()
    total_size = 0.0

    for row in orders:
        side_counts[row.get("side", "unknown")] += 1
        total_size += float(row.get("size", 0.0))

    return {
        "submitted_count": len(orders),
        "side_counts": side_counts,
        "avg_size": total_size / max(len(orders), 1),
    }


def summarize_fills(fills: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not fills:
        return {
            "count": 0,
            "side_counts": Counter(),
            "avg_fill_size": 0.0,
        }

    side_counts = Counter()
    total_fill_size = 0.0

    for row in fills:
        side_counts[row.get("side", "unknown")] += 1
        total_fill_size += float(row.get("filled_size", 0.0))

    return {
        "count": len(fills),
        "side_counts": side_counts,
        "avg_fill_size": total_fill_size / max(len(fills), 1),
    }


def count_canceled_orders(events: List[Dict[str, Any]]) -> int:
    return sum(1 for row in events if row.get("event_type") == "order_canceled")


def summarize_last_fill(fills: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not fills:
        return {
            "has_fill": False,
            "side": "none",
            "price": 0.0,
            "size": 0.0,
        }

    last = fills[-1]
    return {
        "has_fill": True,
        "side": str(last.get("side", "unknown")),
        "price": float(last.get("price", 0.0)),
        "size": float(last.get("filled_size", 0.0)),
    }


def summarize_snapshots(snaps: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not snaps:
        return {
            "count": 0,
            "markets": [],
            "avg_spread": 0.0,
            "avg_mid": 0.0,
        }

    markets = sorted({row.get("market_id", "unknown") for row in snaps})
    avg_spread = sum(float(row.get("ask", 0.0)) - float(row.get("bid", 0.0)) for row in snaps) / len(snaps)
    avg_mid = sum((float(row.get("ask", 0.0)) + float(row.get("bid", 0.0))) / 2.0 for row in snaps) / len(snaps)

    return {
        "count": len(snaps),
        "markets": markets,
        "avg_spread": avg_spread,
        "avg_mid": avg_mid,
    }


def summarize_snapshot_monitoring(snaps: List[Dict[str, Any]]) -> Dict[str, float]:
    if not snaps:
        return {
            "latest_snapshot_age_sec": 0.0,
            "avg_snapshot_gap_sec": 0.0,
            "max_snapshot_gap_sec": 0.0,
            "p50_snapshot_gap_sec": 0.0,
            "p95_snapshot_gap_sec": 0.0,
            "p99_snapshot_gap_sec": 0.0,
        }

    sorted_snaps = sorted(snaps, key=lambda row: float(row.get("ts", 0.0)))
    ts_values = [float(row.get("ts", 0.0)) for row in sorted_snaps]
    latest_ts = ts_values[-1]
    latest_age = max(0.0, float(time.time()) - latest_ts)

    gaps: List[float] = []
    if len(ts_values) >= 2:
        prev = ts_values[0]
        for cur in ts_values[1:]:
            gaps.append(max(0.0, float(cur - prev)))
            prev = cur

    if not gaps:
        return {
            "latest_snapshot_age_sec": float(latest_age),
            "avg_snapshot_gap_sec": 0.0,
            "max_snapshot_gap_sec": 0.0,
            "p50_snapshot_gap_sec": 0.0,
            "p95_snapshot_gap_sec": 0.0,
            "p99_snapshot_gap_sec": 0.0,
        }

    sorted_gaps = sorted(gaps)

    def _percentile_nearest_rank(values: List[float], pct: float) -> float:
        if not values:
            return 0.0
        rank = int((pct * len(values)) + 0.999999999)
        rank = max(1, min(rank, len(values)))
        return float(values[rank - 1])

    return {
        "latest_snapshot_age_sec": float(latest_age),
        "avg_snapshot_gap_sec": float(sum(gaps) / len(gaps)),
        "max_snapshot_gap_sec": float(max(gaps)),
        "p50_snapshot_gap_sec": _percentile_nearest_rank(sorted_gaps, 0.50),
        "p95_snapshot_gap_sec": _percentile_nearest_rank(sorted_gaps, 0.95),
        "p99_snapshot_gap_sec": _percentile_nearest_rank(sorted_gaps, 0.99),
    }


def summarize_heartbeat_monitoring(events: List[Dict[str, Any]]) -> Dict[str, float]:
    heartbeats = [row for row in events if row.get("event_type") == "heartbeat"]
    if not heartbeats:
        return {
            "max_reconnect_count": 0.0,
            "latest_heartbeat_snapshot_age_sec": 0.0,
            "latest_message_age_sec": 0.0,
            "latest_snapshot_wallclock_age_sec": 0.0,
            "latest_observer_uptime_sec": 0.0,
            "latest_message_count": 0.0,
            "latest_snapshot_count": 0.0,
            "latest_message_rate_per_min": 0.0,
            "latest_snapshot_rate_per_min": 0.0,
        }

    max_reconnect = 0
    for row in heartbeats:
        payload = row.get("payload", {})
        try:
            max_reconnect = max(max_reconnect, int(payload.get("reconnect_count", 0) or 0))
        except (TypeError, ValueError):
            continue

    latest_payload = heartbeats[-1].get("payload", {})
    return {
        "max_reconnect_count": float(max_reconnect),
        "latest_heartbeat_snapshot_age_sec": float(latest_payload.get("snapshot_age_sec", 0.0) or 0.0),
        "latest_message_age_sec": float(latest_payload.get("last_message_age_sec", 0.0) or 0.0),
        "latest_snapshot_wallclock_age_sec": float(
            latest_payload.get("last_snapshot_wallclock_age_sec", 0.0) or 0.0
        ),
        "latest_observer_uptime_sec": float(latest_payload.get("observer_uptime_sec", 0.0) or 0.0),
        "latest_message_count": float(latest_payload.get("message_count", 0.0) or 0.0),
        "latest_snapshot_count": float(latest_payload.get("snapshot_count", 0.0) or 0.0),
        "latest_message_rate_per_min": float(latest_payload.get("message_rate_per_min", 0.0) or 0.0),
        "latest_snapshot_rate_per_min": float(latest_payload.get("snapshot_rate_per_min", 0.0) or 0.0),
    }


def summarize_candidate_trades(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    candidate_rows = [row for row in events if row.get("event_type") == "candidate_trade"]
    if not candidate_rows:
        return {
            "count": 0,
            "buy_count": 0,
            "sell_count": 0,
            "reason_counts": Counter(),
            "action_reason_counts": Counter(),
            "avg_edge": 0.0,
            "avg_edge_abs": 0.0,
            "avg_ev_est": 0.0,
            "avg_ev_req": 0.0,
            "avg_stake_base": 0.0,
            "avg_stake_scaled": 0.0,
        }

    reason_counts = Counter()
    action_reason_counts = Counter()
    buy_count = 0
    sell_count = 0

    edge_vals: List[float] = []
    edge_abs_vals: List[float] = []
    ev_est_vals: List[float] = []
    ev_req_vals: List[float] = []
    stake_base_vals: List[float] = []
    stake_scaled_vals: List[float] = []

    def _append_numeric(values: List[float], payload: Dict[str, Any], key: str) -> None:
        try:
            values.append(float(payload.get(key, 0.0)))
        except (TypeError, ValueError):
            return

    for row in candidate_rows:
        payload = row.get("payload", {})
        action = str(payload.get("action", ""))
        reason = str(payload.get("reason", "unknown"))
        reason_counts[reason] += 1
        action_reason_counts["%s/%s" % (action, reason)] += 1

        if action == "BUY":
            buy_count += 1
        elif action == "SELL":
            sell_count += 1

        _append_numeric(edge_vals, payload, "edge")
        _append_numeric(edge_abs_vals, payload, "edge_abs")
        _append_numeric(ev_est_vals, payload, "ev_est")
        _append_numeric(ev_req_vals, payload, "ev_req")
        _append_numeric(stake_base_vals, payload, "stake_base")
        _append_numeric(stake_scaled_vals, payload, "stake_scaled")

    def _avg(values: List[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    return {
        "count": len(candidate_rows),
        "buy_count": int(buy_count),
        "sell_count": int(sell_count),
        "reason_counts": reason_counts,
        "action_reason_counts": action_reason_counts,
        "avg_edge": _avg(edge_vals),
        "avg_edge_abs": _avg(edge_abs_vals),
        "avg_ev_est": _avg(ev_est_vals),
        "avg_ev_req": _avg(ev_req_vals),
        "avg_stake_base": _avg(stake_base_vals),
        "avg_stake_scaled": _avg(stake_scaled_vals),
    }


def main() -> None:
    events = read_jsonl(os.path.join(LOG_DIR, "events.jsonl"))
    orders = read_jsonl(os.path.join(LOG_DIR, "orders.jsonl"))
    fills = read_jsonl(os.path.join(LOG_DIR, "fills.jsonl"))
    snaps = read_jsonl(os.path.join(LOG_DIR, "snapshots.jsonl"))

    event_summary = summarize_events(events)
    order_summary = summarize_orders(orders)
    fill_summary = summarize_fills(fills)
    snap_summary = summarize_snapshots(snaps)
    snap_monitoring = summarize_snapshot_monitoring(snaps)
    heartbeat_monitoring = summarize_heartbeat_monitoring(events)
    candidate_summary = summarize_candidate_trades(events)
    canceled_orders = count_canceled_orders(events)
    last_fill = summarize_last_fill(fills)

    fill_rate = 0.0
    if order_summary["submitted_count"] > 0:
        fill_rate = float(fill_summary["count"]) / float(order_summary["submitted_count"])

    print("Paper Bot Replay Summary")
    print("-" * 32)

    print(f"Snapshots processed: {snap_summary['count']}")
    print(f"Markets seen: {', '.join(snap_summary['markets']) if snap_summary['markets'] else 'none'}")
    print(f"Average mid: {snap_summary['avg_mid']:.4f}")
    print(f"Average spread: {snap_summary['avg_spread']:.4f}")
    print()

    print("Events")
    for k, v in sorted(event_summary["event_counts"].items()):
        print(f"  {k}: {v}")
    print()

    print("Signals by action")
    for k, v in sorted(event_summary["signal_actions"].items()):
        print(f"  {k}: {v}")
    print()

    print("Signal reasons")
    for k, v in sorted(event_summary["signal_reasons"].items()):
        print(f"  {k}: {v}")
    print()

    print("Risk check reasons")
    for k, v in sorted(event_summary["risk_reasons"].items()):
        print(f"  {k}: {v}")
    print()

    print("Monitoring")
    print(f"  Heartbeats: {event_summary['heartbeat_count']}")
    print(f"  Stale feed warnings: {event_summary['stale_warning_count']}")
    print(f"  Max reconnect count: {int(heartbeat_monitoring['max_reconnect_count'])}")
    print(f"  Latest snapshot age sec: {snap_monitoring['latest_snapshot_age_sec']:.2f}")
    print(f"  Latest heartbeat snapshot age sec: {heartbeat_monitoring['latest_heartbeat_snapshot_age_sec']:.2f}")
    print(f"  Latest message age sec: {heartbeat_monitoring['latest_message_age_sec']:.2f}")
    print(
        f"  Latest snapshot wallclock age sec: {heartbeat_monitoring['latest_snapshot_wallclock_age_sec']:.2f}"
    )
    print(f"  Latest observer uptime sec: {heartbeat_monitoring['latest_observer_uptime_sec']:.2f}")
    print(f"  Latest message count: {int(heartbeat_monitoring['latest_message_count'])}")
    print(f"  Latest snapshot count: {int(heartbeat_monitoring['latest_snapshot_count'])}")
    print(f"  Latest message rate per min: {heartbeat_monitoring['latest_message_rate_per_min']:.2f}")
    print(f"  Latest snapshot rate per min: {heartbeat_monitoring['latest_snapshot_rate_per_min']:.2f}")
    print(f"  Avg snapshot gap sec: {snap_monitoring['avg_snapshot_gap_sec']:.2f}")
    print(f"  Max snapshot gap sec: {snap_monitoring['max_snapshot_gap_sec']:.2f}")
    print(f"  P50 snapshot gap sec: {snap_monitoring['p50_snapshot_gap_sec']:.2f}")
    print(f"  P95 snapshot gap sec: {snap_monitoring['p95_snapshot_gap_sec']:.2f}")
    print(f"  P99 snapshot gap sec: {snap_monitoring['p99_snapshot_gap_sec']:.2f}")
    print()

    print("Candidate Trades")
    print(f"  Total: {candidate_summary['count']}")
    print(f"  BUY candidates: {candidate_summary['buy_count']}")
    print(f"  SELL candidates: {candidate_summary['sell_count']}")
    print("  By reason:")
    for k, v in sorted(candidate_summary["reason_counts"].items()):
        print(f"    {k}: {v}")
    print(f"  Avg edge: {candidate_summary['avg_edge']:.4f}")
    print(f"  Avg edge_abs: {candidate_summary['avg_edge_abs']:.4f}")
    print(f"  Avg ev_est: {candidate_summary['avg_ev_est']:.4f}")
    print(f"  Avg ev_req: {candidate_summary['avg_ev_req']:.4f}")
    print(f"  Avg stake_base: {candidate_summary['avg_stake_base']:.4f}")
    print(f"  Avg stake_scaled: {candidate_summary['avg_stake_scaled']:.4f}")
    if candidate_summary["action_reason_counts"]:
        print("  Action/reason:")
        for k, v in sorted(candidate_summary["action_reason_counts"].items()):
            print(f"    {k}: {v}")
    print()

    print("Orders")
    print(f"  Submitted: {order_summary['submitted_count']}")
    print(f"  Filled: {fill_summary['count']}")
    print(f"  Canceled: {canceled_orders}")
    print(f"  Avg submitted size: {order_summary['avg_size']:.2f}")
    print(f"  Fill rate: {fill_rate:.2%}")
    for k, v in sorted(order_summary["side_counts"].items()):
        print(f"  Side {k}: {v}")
    print()

    if fill_summary["side_counts"]:
        print("Fills by side")
        for k, v in sorted(fill_summary["side_counts"].items()):
            print(f"  Side {k}: {v}")
        print()

    if not last_fill["has_fill"]:
        print("Last fill: none")
    else:
        print("Last fill")
        print(f"  Side: {last_fill['side']}")
        print(f"  Price: {last_fill['price']:.4f}")
        print(f"  Size: {last_fill['size']:.2f}")
    print()

    bot_stop = event_summary["bot_stop"]
    if bot_stop:
        payload = bot_stop.get("payload", {})
        print("Final bot state")
        print(f"  Bankroll: {float(payload.get('bankroll', 0.0)):.2f}")
        print(f"  Realized PnL: {float(payload.get('realized_pnl', 0.0)):.2f}")
        print(f"  Open orders: {int(payload.get('open_orders', 0))}")
        print(f"  Positions: {int(payload.get('positions', 0))}")
        print(f"  Halted: {bool(payload.get('halted', False))}")


if __name__ == "__main__":
    main()
