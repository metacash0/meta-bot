from __future__ import annotations

from typing import Dict, Optional

from binary_bot.datafeed import get_default_feed
from binary_bot.journal import Journal
from binary_bot.oms import PaperOMS
from binary_bot.risk import RiskManager
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
