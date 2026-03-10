from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

from binary_bot.datafeed import get_default_feed
from binary_bot.journal import Journal
from binary_bot.oms import PaperOMS
from binary_bot.risk import RiskManager
from binary_bot.sports_state import SoccerStateStore, summarize_state
from binary_bot.sportsfeed import SoccerSportsFeed, get_tracked_fixture_ids_from_market_map
from binary_bot.state import BotState
from binary_bot.strategy import Strategy


class TradingBot:
    """
    Phase-1 paper bot:
    - consumes snapshots
    - logs snapshots
    - simulates fills/cancels
    - marks to market
    - generates signals
    - applies risk checks
    - places paper orders
    """

    def __init__(
        self,
        bankroll: float = 1000.0,
        order_ttl_sec: float = 15.0,
    ):
        self.state = BotState(bankroll=float(bankroll), peak_equity=float(bankroll))
        self.journal = Journal()
        self.strategy = Strategy(
            min_edge=0.002,
            min_ev_per_dollar=0.0005,
        )
        self.risk = RiskManager()
        self.oms = PaperOMS(journal=self.journal, order_ttl_sec=float(order_ttl_sec))
        self.sports_feed = SoccerSportsFeed()
        self.soccer_state = SoccerStateStore()
        self._tracked_fixture_ids = get_tracked_fixture_ids_from_market_map()
        self._last_sports_poll_wallclock = 0.0

    def on_snapshot(
        self,
        snap,
        p_base: Optional[float] = None,
        observations: Optional[Dict[str, Dict]] = None,
    ) -> None:
        self.journal.snapshot(snap.__dict__)

        self.oms.simulate_fills(self.state, snap)
        self.oms.cancel_stale_orders(self.state)
        equity = self.oms.mark_to_market(self.state, snap)

        signal, meta = self.strategy.decide(
            snap=snap,
            bankroll=float(self.state.bankroll),
            p_base=p_base,
            observations=observations,
        )

        self.journal.event(
            "signal",
            {
                **signal.__dict__,
                "meta": meta,
                "equity": float(equity),
            },
        )

        candidate_reason = str(meta.get("reason", signal.reason))
        should_emit_candidate = signal.action in {"BUY", "SELL"} or candidate_reason in {
            "stake_zero",
            "ev_gate_block",
            "edge_too_small",
        }
        if should_emit_candidate:
            candidate_market_id = str(meta.get("market_id", signal.market_id))
            self.journal.event(
                "candidate_trade",
                {
                    "ts": float(snap.ts),
                    "market_id": candidate_market_id,
                    "action": str(signal.action),
                    "reason": str(candidate_reason),
                    "equity": float(equity),
                    "bid": float(meta.get("bid", snap.bid)),
                    "ask": float(meta.get("ask", snap.ask)),
                    "mid": float(meta.get("mid", snap.mid)),
                    "spread": float(meta.get("spread", snap.spread)),
                    "p_fair": float(meta.get("p_fair", signal.p_fair)),
                    "p_exec": float(meta.get("p_exec", signal.p_exec)),
                    "ci_lo": float(meta.get("ci_lo", 0.0)),
                    "ci_hi": float(meta.get("ci_hi", 0.0)),
                    "edge": float(meta.get("edge", signal.edge)),
                    "edge_abs": float(meta.get("edge_abs", abs(signal.edge))),
                    "regime": str(signal.regime),
                    "toxicity": float(meta.get("toxicity", signal.toxicity)),
                    "tox_reg": str(meta.get("tox_reg", "")),
                    "trend_strength": float(meta.get("trend_strength", 0.0)),
                    "pf_latency_mult": float(meta.get("pf_latency_mult", 0.0)),
                    "book_mid_gap": float(meta.get("book_mid_gap", 0.0)),
                    "ev_est": float(meta.get("ev_est", 0.0)),
                    "ev_req": float(meta.get("ev_req", 0.0)),
                    "stake_base": float(meta.get("stake_base", 0.0)),
                    "stake_scaled": float(meta.get("stake_scaled", 0.0)),
                    "min_edge": float(meta.get("min_edge", self.strategy.min_edge)),
                    "min_ev_per_dollar": float(meta.get("min_ev_per_dollar", self.strategy.min_ev_per_dollar)),
                },
            )

        if signal.action == "HOLD":
            return

        ok, reason = self.risk.allow_trade(self.state, float(signal.size))
        self.journal.event(
            "risk_check",
            {
                "market_id": signal.market_id,
                "action": signal.action,
                "size": float(signal.size),
                "ok": bool(ok),
                "reason": str(reason),
            },
        )

        if not ok:
            return

        # Paper maker-first logic:
        # BUY posts at bid, SELL posts at ask.
        order_price = float(snap.bid if signal.action == "BUY" else snap.ask)

        self.oms.place_post_order(
            state=self.state,
            market_id=signal.market_id,
            side=signal.action,
            price=order_price,
            size=float(signal.size),
            meta=meta,
        )

    def run(self, total_ticks: int = 120, tick_interval_sec: float = 0.25) -> None:
        feed = get_default_feed(
            market_id="mock_market_1",
            tick_interval_sec=float(tick_interval_sec),
            total_ticks=int(total_ticks),
        )
        heartbeat_sec = float(os.getenv("POLYMARKET_HEARTBEAT_SEC", "10"))
        stale_feed_sec = float(os.getenv("POLYMARKET_STALE_FEED_SEC", "15"))
        last_heartbeat_wallclock = float(time.time())
        last_snapshot_wallclock: Optional[float] = None
        stale_warning_emitted = False
        last_seen_reconnect_count: Optional[int] = None

        self.journal.event(
            "bot_start",
            {
                "bankroll": float(self.state.bankroll),
                "total_ticks": int(total_ticks),
                "tick_interval_sec": float(tick_interval_sec),
            },
        )

        try:
            for snap in feed.snapshots():
                now = float(time.time())

                if (now - self._last_sports_poll_wallclock) >= float(self.sports_feed.poll_interval_sec):
                    self._last_sports_poll_wallclock = now
                    sports_rows = self.sports_feed.poll_live_fixtures()
                    self.soccer_state.update_from_fixture_rows(sports_rows)

                    if self._tracked_fixture_ids:
                        for state in self.soccer_state.get_all_states():
                            fixture_id = int(state.get("fixture_id", 0) or 0)
                            if fixture_id not in self._tracked_fixture_ids:
                                continue
                            self.journal.event("sports_state", summarize_state(state))

                snapshot_age_sec = max(0.0, float(now - float(snap.ts)))
                monitor: Dict[str, Any] = {}
                feed_monitor = getattr(feed, "monitoring_state", None)
                if callable(feed_monitor):
                    try:
                        raw_monitor = feed_monitor()
                        if isinstance(raw_monitor, dict):
                            monitor = raw_monitor
                    except Exception:
                        monitor = {}

                reconnect_count = int(monitor.get("reconnect_count", 0) or 0)
                last_message_wallclock = monitor.get("last_message_wallclock")
                last_snapshot_feed_wallclock = monitor.get("last_snapshot_wallclock")
                message_count = int(monitor.get("message_count", 0) or 0)
                snapshot_count = int(monitor.get("snapshot_count", 0) or 0)
                started_wallclock = monitor.get("started_wallclock")
                last_message_age_sec = 0.0
                last_snapshot_wallclock_age_sec = 0.0
                observer_uptime_sec = 0.0
                message_rate_per_min = 0.0
                snapshot_rate_per_min = 0.0
                if last_message_wallclock is not None:
                    try:
                        last_message_age_sec = max(0.0, float(now - float(last_message_wallclock)))
                    except (TypeError, ValueError):
                        last_message_age_sec = 0.0
                if last_snapshot_feed_wallclock is not None:
                    try:
                        last_snapshot_wallclock_age_sec = max(
                            0.0, float(now - float(last_snapshot_feed_wallclock))
                        )
                    except (TypeError, ValueError):
                        last_snapshot_wallclock_age_sec = 0.0
                if started_wallclock is not None:
                    try:
                        observer_uptime_sec = max(0.0, float(now - float(started_wallclock)))
                    except (TypeError, ValueError):
                        observer_uptime_sec = 0.0

                per_min_den = max(observer_uptime_sec / 60.0, 1e-9)
                message_rate_per_min = float(message_count) / per_min_den
                snapshot_rate_per_min = float(snapshot_count) / per_min_den
                tracked_soccer_fixtures = 0
                if self._tracked_fixture_ids:
                    tracked_soccer_fixtures = sum(
                        1
                        for row in self.soccer_state.get_all_states()
                        if int(row.get("fixture_id", 0) or 0) in self._tracked_fixture_ids
                    )

                if (now - last_heartbeat_wallclock) >= heartbeat_sec:
                    last_snapshot_age = 0.0
                    if last_snapshot_wallclock is not None:
                        last_snapshot_age = float(now - last_snapshot_wallclock)
                    self.journal.event(
                        "heartbeat",
                        {
                            "bankroll": float(self.state.bankroll),
                            "realized_pnl": float(self.state.realized_pnl),
                            "open_orders": len(self.state.open_orders),
                            "positions": len(self.state.positions),
                            "halted": bool(self.state.halted),
                            "last_snapshot_age_sec": float(last_snapshot_age),
                            "snapshot_age_sec": float(snapshot_age_sec),
                            "reconnect_count": int(reconnect_count),
                            "last_message_age_sec": float(last_message_age_sec),
                            "last_snapshot_wallclock_age_sec": float(last_snapshot_wallclock_age_sec),
                            "message_count": int(message_count),
                            "snapshot_count": int(snapshot_count),
                            "observer_uptime_sec": float(observer_uptime_sec),
                            "message_rate_per_min": float(message_rate_per_min),
                            "snapshot_rate_per_min": float(snapshot_rate_per_min),
                            "tracked_soccer_fixtures": int(tracked_soccer_fixtures),
                        },
                    )
                    last_heartbeat_wallclock = now

                stale_seconds: Optional[float] = None
                if last_message_wallclock is not None:
                    stale_seconds = max(0.0, float(now - float(last_message_wallclock)))
                elif last_snapshot_wallclock is not None:
                    stale_seconds = max(0.0, float(now - last_snapshot_wallclock))

                if stale_seconds is not None and stale_seconds > stale_feed_sec:
                    if not stale_warning_emitted:
                        self.journal.event(
                            "stale_feed_warning",
                            {
                                "seconds_since_snapshot": float(stale_seconds),
                            },
                        )
                    stale_warning_emitted = True
                else:
                    stale_warning_emitted = False

                last_snapshot_wallclock = now

                if last_seen_reconnect_count is None:
                    last_seen_reconnect_count = reconnect_count
                elif reconnect_count > last_seen_reconnect_count:
                    self.journal.event(
                        "feed_reconnect_seen",
                        {
                            "reconnect_count": int(reconnect_count),
                        },
                    )
                    last_seen_reconnect_count = reconnect_count

                self.on_snapshot(snap=snap)
        except KeyboardInterrupt:
            self.journal.event("bot_interrupt", {"reason": "keyboard_interrupt"})
        finally:
            self.journal.event(
                "bot_stop",
                {
                    "bankroll": float(self.state.bankroll),
                    "realized_pnl": float(self.state.realized_pnl),
                    "open_orders": len(self.state.open_orders),
                    "positions": len(self.state.positions),
                    "halted": bool(self.state.halted),
                },
            )


if __name__ == "__main__":
    bot = TradingBot(bankroll=1000.0, order_ttl_sec=10.0)
    bot.run(total_ticks=120, tick_interval_sec=0.10)
