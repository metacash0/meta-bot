import time
import uuid
from typing import List, Optional

from binary_bot.journal import Journal
from binary_bot.models import MarketSnapshot, Order, Position
from binary_bot.state import BotState


class PaperOMS:
    """
    Very simple paper OMS for phase-1 forward testing.

    Behavior:
    - stores open orders
    - simulates fills against live snapshots
    - supports canceling stale orders
    - updates positions
    - logs orders/fills/events
    """

    def __init__(self, journal: Journal, order_ttl_sec: float = 15.0):
        self.journal = journal
        self.order_ttl_sec = float(order_ttl_sec)

    def place_post_order(
        self,
        state: BotState,
        market_id: str,
        side: str,
        price: float,
        size: float,
        meta: Optional[dict] = None,
    ) -> Order:
        order = Order(
            order_id=str(uuid.uuid4()),
            ts=time.time(),
            market_id=market_id,
            side=side,
            price=float(price),
            size=float(size),
            kind="POST",
        )
        state.open_orders[order.order_id] = order
        self.journal.order(
            {
                **order.__dict__,
                "meta": meta or {},
            }
        )
        return order

    def cancel_stale_orders(self, state: BotState) -> None:
        now = time.time()
        to_remove: List[str] = []

        for order_id, order in state.open_orders.items():
            age = now - float(order.ts)
            if order.status == "OPEN" and age >= self.order_ttl_sec:
                order.status = "CANCELED"
                self.journal.event(
                    "order_canceled",
                    {
                        "order_id": order.order_id,
                        "market_id": order.market_id,
                        "age_sec": age,
                    },
                )
                to_remove.append(order_id)

        for order_id in to_remove:
            del state.open_orders[order_id]

    def simulate_fills(self, state: BotState, snap: MarketSnapshot) -> None:
        to_remove: List[str] = []

        for order_id, order in state.open_orders.items():
            if order.market_id != snap.market_id:
                continue

            fill = False

            if order.side == "BUY" and order.price >= snap.ask:
                fill = True
            elif order.side == "SELL" and order.price <= snap.bid:
                fill = True

            if not fill:
                continue

            order.status = "FILLED"
            order.filled_size = float(order.size)

            self.journal.fill(order.__dict__)

            existing = state.positions.get(order.market_id)
            if existing is None:
                state.positions[order.market_id] = Position(
                    market_id=order.market_id,
                    side=order.side,
                    avg_price=float(order.price),
                    size=float(order.size),
                    open_ts=time.time(),
                    meta={},
                )
            else:
                # Phase-1 simplification:
                # same-side fills increase size and average price;
                # opposite-side fills reduce or flip position.
                if existing.side == order.side:
                    new_size = float(existing.size + order.size)
                    new_avg = (
                        (existing.avg_price * existing.size) + (order.price * order.size)
                    ) / max(new_size, 1e-9)
                    existing.avg_price = float(new_avg)
                    existing.size = float(new_size)
                else:
                    closed_size = float(min(order.size, existing.size))
                    realized = 0.0
                    if existing.side == "BUY" and order.side == "SELL":
                        realized = (float(order.price) - float(existing.avg_price)) * closed_size * 100.0
                    elif existing.side == "SELL" and order.side == "BUY":
                        realized = (float(existing.avg_price) - float(order.price)) * closed_size * 100.0

                    if closed_size > 0.0:
                        state.realized_pnl = float(state.realized_pnl + realized)
                        self.journal.event(
                            "position_realized",
                            {
                                "market_id": order.market_id,
                                "existing_side": existing.side,
                                "close_side": order.side,
                                "close_price": float(order.price),
                                "avg_price": float(existing.avg_price),
                                "closed_size": float(closed_size),
                                "realized_pnl": float(realized),
                                "realized_pnl_total": float(state.realized_pnl),
                            },
                        )

                    if order.size < existing.size:
                        existing.size = float(existing.size - order.size)
                    elif order.size == existing.size:
                        del state.positions[order.market_id]
                    else:
                        residual = float(order.size - existing.size)
                        state.positions[order.market_id] = Position(
                            market_id=order.market_id,
                            side=order.side,
                            avg_price=float(order.price),
                            size=float(residual),
                            open_ts=time.time(),
                            meta={},
                        )

            to_remove.append(order_id)

        for order_id in to_remove:
            del state.open_orders[order_id]

    def mark_to_market(self, state: BotState, snap: MarketSnapshot) -> float:
        """
        Very simple paper MTM:
        - BUY positions are marked to bid
        - SELL positions are marked to ask
        """
        pos = state.positions.get(snap.market_id)
        if pos is None:
            equity = float(state.bankroll + state.realized_pnl)
            state.peak_equity = max(float(state.peak_equity), equity)
            return equity

        mark = float(snap.bid if pos.side == "BUY" else snap.ask)
        if pos.side == "BUY":
            unrealized = (mark - pos.avg_price) * pos.size * 100.0
        else:
            unrealized = (pos.avg_price - mark) * pos.size * 100.0

        equity = float(state.bankroll + state.realized_pnl + unrealized)
        state.peak_equity = max(float(state.peak_equity), equity)
        return equity
