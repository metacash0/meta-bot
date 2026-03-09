from dataclasses import dataclass, field
from typing import Dict

from binary_bot.models import Order, Position


@dataclass
class BotState:
    bankroll: float = 1000.0
    open_orders: Dict[str, Order] = field(default_factory=dict)
    positions: Dict[str, Position] = field(default_factory=dict)
    realized_pnl: float = 0.0
    peak_equity: float = 1000.0
    halted: bool = False
