from dataclasses import dataclass
from typing import Optional


@dataclass
class MarketSnapshot:
    ts: float
    market_id: str
    bid: float
    ask: float
    last: float
    bid_size: float
    ask_size: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return max(0.0, self.ask - self.bid)


@dataclass
class Signal:
    ts: float
    market_id: str
    p_fair: float
    p_exec: float
    edge: float
    regime: str
    toxicity: float
    action: str
    size: float
    reason: str


@dataclass
class Order:
    order_id: str
    ts: float
    market_id: str
    side: str
    price: float
    size: float
    kind: str
    status: str = "OPEN"
    filled_size: float = 0.0


@dataclass
class Position:
    market_id: str
    side: str
    avg_price: float
    size: float
    open_ts: float
    meta: Optional[dict] = None
