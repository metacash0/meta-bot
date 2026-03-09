import math
import time
from typing import Iterator, Optional

from binary_bot.models import MarketSnapshot


class MockDataFeed:
    """
    Deterministic mock data feed for phase-1 testing.

    Produces a smooth oscillating market with small spread changes and
    simple depth changes so the rest of the paper bot can be exercised
    end-to-end before plugging in a real websocket feed.
    """

    def __init__(
        self,
        market_id: str = "mock_market_1",
        start_mid: float = 0.50,
        base_spread: float = 0.02,
        tick_interval_sec: float = 1.0,
        total_ticks: int = 300,
    ):
        self.market_id = market_id
        self.start_mid = float(start_mid)
        self.base_spread = float(base_spread)
        self.tick_interval_sec = float(tick_interval_sec)
        self.total_ticks = int(total_ticks)

    def snapshots(self) -> Iterator[MarketSnapshot]:
        for t in range(self.total_ticks):
            ts = time.time()

            wave = 0.10 * math.sin(t / 8.0)
            drift = 0.0005 * t / max(self.total_ticks, 1)
            mid = min(0.95, max(0.05, self.start_mid + wave + drift))

            spread_bump = 0.01 * (0.5 + 0.5 * math.sin(t / 17.0))
            spread = max(0.01, self.base_spread + spread_bump)

            bid = max(0.01, mid - spread / 2.0)
            ask = min(0.99, mid + spread / 2.0)

            bid_size = 100.0 + 20.0 * (1.0 + math.sin(t / 7.0))
            ask_size = 100.0 + 20.0 * (1.0 + math.cos(t / 9.0))
            last = mid

            yield MarketSnapshot(
                ts=float(ts),
                market_id=self.market_id,
                bid=float(bid),
                ask=float(ask),
                last=float(last),
                bid_size=float(bid_size),
                ask_size=float(ask_size),
            )

            time.sleep(self.tick_interval_sec)


def get_default_feed(
    market_id: str = "mock_market_1",
    tick_interval_sec: float = 0.25,
    total_ticks: int = 120,
) -> MockDataFeed:
    return MockDataFeed(
        market_id=market_id,
        start_mid=0.50,
        base_spread=0.02,
        tick_interval_sec=tick_interval_sec,
        total_ticks=total_ticks,
    )
