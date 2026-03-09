from typing import Tuple

from binary_bot.state import BotState


class RiskManager:
    """
    Simple phase-1 paper risk controls.
    """

    def __init__(
        self,
        max_order_size: float = 65.0,
        max_open_orders: int = 2,
        max_total_exposure: float = 150.0,
        max_drawdown_pct: float = 0.20,
    ):
        self.max_order_size = float(max_order_size)
        self.max_open_orders = int(max_open_orders)
        self.max_total_exposure = float(max_total_exposure)
        self.max_drawdown_pct = float(max_drawdown_pct)

    def current_exposure(self, state: BotState) -> float:
        order_exposure = sum(o.size for o in state.open_orders.values() if o.status == "OPEN")
        position_exposure = sum(p.size for p in state.positions.values())
        return float(order_exposure + position_exposure)

    def allow_trade(self, state: BotState, size: float) -> Tuple[bool, str]:
        if state.halted:
            return False, "halted"

        if size <= 0.0:
            return False, "zero_size"

        if size > self.max_order_size:
            return False, "order_size_cap"

        if len(state.open_orders) >= self.max_open_orders:
            return False, "max_open_orders"

        exposure_after = self.current_exposure(state) + float(size)
        if exposure_after > self.max_total_exposure:
            return False, "exposure_cap"

        peak = max(float(state.peak_equity), 1e-9)
        drawdown = 1.0 - (float(state.bankroll) / peak)
        if drawdown > self.max_drawdown_pct:
            state.halted = True
            return False, "kill_switch_drawdown"

        return True, "ok"
