from typing import Dict, Union

from binary_bot.models import MarketSnapshot
from shared.signals import (
    toxicity_score,
    toxicity_regime,
    market_regime,
    trend_strength_score,
    pf_latency_multiplier,
)

FeatureValue = Union[float, str]


class FeatureEngine:
    def __init__(self):
        self.prev_mid: Dict[str, float] = {}

    def compute(self, snap: MarketSnapshot) -> Dict[str, FeatureValue]:
        prev_mid = self.prev_mid.get(snap.market_id, snap.mid)
        mid_delta = abs(snap.mid - prev_mid)
        self.prev_mid[snap.market_id] = snap.mid

        depth_proxy = float(snap.bid_size + snap.ask_size)
        vol_proxy = float(max(1e-6, depth_proxy))

        book_mid_gap = float(abs(snap.last - snap.mid))

        tox_score = toxicity_score(
            mid_delta=float(mid_delta),
            spread=float(snap.spread),
            vol=float(vol_proxy),
            book_mid_gap=float(book_mid_gap),
        )

        tox_reg = toxicity_regime(float(tox_score))

        regime_mode = market_regime(
            mid_delta=float(mid_delta),
            tox_reg=str(tox_reg),
            run_signal=0.0,
            pace_signal=0.0,
        )

        trend_strength = trend_strength_score(
            mid_delta=float(mid_delta),
            tox_reg=str(tox_reg),
            run_signal=0.0,
            pace_signal=0.0,
            spread=float(snap.spread),
        )

        pf_lat_mult = pf_latency_multiplier(
            spread=float(snap.spread),
            mid_delta=float(mid_delta),
            tox_reg=str(tox_reg),
        )

        return {
            "mid": float(snap.mid),
            "spread": float(snap.spread),
            "mid_delta": float(mid_delta),
            "depth_proxy": float(depth_proxy),
            "vol_proxy": float(vol_proxy),
            "toxicity": float(tox_score),
            "tox_reg": str(tox_reg),
            "regime_mode": str(regime_mode),
            "trend_strength": float(trend_strength),
            "pf_latency_mult": float(pf_lat_mult),
            "book_mid_gap": float(book_mid_gap),
        }
