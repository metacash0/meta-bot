from __future__ import annotations

from typing import Dict, Optional, Tuple

from binary_bot.features import FeatureEngine
from binary_bot.models import MarketSnapshot, Signal
from shared.pf import LogitBiasParticleFilter
from shared.signals import exec_alpha_dynamic
from shared.sizing import stake_ladder, directional_edge_scale
from shared.risk_models import trade_ev_reversion_exit_from_fill


class ProbabilityEngine:
    """
    Thin production wrapper around the shared particle filter.
    Keeps one PF state per market_id.
    """

    def __init__(self, pf_particles: int = 4000, pf_process_vol: float = 0.03):
        self.pf_particles = int(pf_particles)
        self.pf_process_vol = float(pf_process_vol)
        self._pf_by_market: Dict[str, LogitBiasParticleFilter] = {}

    def _get_pf(self, market_id: str) -> LogitBiasParticleFilter:
        pf = self._pf_by_market.get(market_id)
        if pf is None:
            pf = LogitBiasParticleFilter(
                N=self.pf_particles,
                process_vol=self.pf_process_vol,
            )
            self._pf_by_market[market_id] = pf
        return pf

    def fair_value(
        self,
        snap: MarketSnapshot,
        feats: Dict[str, float | str],
        p_base: Optional[float] = None,
        observations: Optional[Dict[str, Dict]] = None,
    ) -> Tuple[float, float, Tuple[float, float]]:
        """
        Returns:
            p_fair: PF-estimated fair value
            p_exec: execution-adjusted fair value
            ci: 95% credible interval from PF
        """
        if p_base is None:
            p_base = snap.mid

        pf = self._get_pf(snap.market_id)

        if observations:
            pf.update(
                p_base=float(p_base),
                observations=observations,
                process_vol=self.pf_process_vol * float(feats["pf_latency_mult"]),
            )

        p_fair = float(pf.estimate(float(p_base)))
        ci = pf.credible_interval(float(p_base))

        alpha = exec_alpha_dynamic(
            float(snap.spread),
            float(feats["mid_delta"]),
        )
        p_exec = float(snap.mid + alpha * (p_fair - snap.mid))

        return p_fair, p_exec, ci


class Strategy:
    """
    Paper-trading strategy wrapper that reuses shared research logic.
    """

    def __init__(
        self,
        min_edge: float = 0.04,
        min_ev_per_dollar: float = 0.015,
        bankroll_cap_pct: float = 0.15,
    ):
        self.min_edge = float(min_edge)
        self.min_ev_per_dollar = float(min_ev_per_dollar)
        self.bankroll_cap_pct = float(bankroll_cap_pct)

        self.features = FeatureEngine()
        self.prob = ProbabilityEngine()

    def decide(
        self,
        snap: MarketSnapshot,
        bankroll: float,
        p_base: Optional[float] = None,
        observations: Optional[Dict[str, Dict]] = None,
        sports_fair: Optional[float] = None,
    ) -> tuple[Signal, Dict[str, float | str]]:
        fair_source = "market_mid"
        fair_input = p_base
        if sports_fair is not None:
            fair_input = float(sports_fair)
            fair_source = "sports_model"

        feats = self.features.compute(snap)
        p_fair, p_exec, ci = self.prob.fair_value(
            snap=snap,
            feats=feats,
            p_base=fair_input,
            observations=observations,
        )

        edge = float(p_exec - snap.mid)
        edge_abs = abs(edge)
        base_meta: Dict[str, float | str] = {
            **feats,
            "market_id": str(snap.market_id),
            "bid": float(snap.bid),
            "ask": float(snap.ask),
            "mid": float(snap.mid),
            "spread": float(snap.spread),
            "p_fair": float(p_fair),
            "p_exec": float(p_exec),
            "ci_lo": float(ci[0]),
            "ci_hi": float(ci[1]),
            "edge": float(edge),
            "edge_abs": float(edge_abs),
            "regime_mode": str(feats.get("regime_mode", "")),
            "toxicity": float(feats.get("toxicity", 0.0)),
            "tox_reg": str(feats.get("tox_reg", "")),
            "trend_strength": float(feats.get("trend_strength", 0.0)),
            "pf_latency_mult": float(feats.get("pf_latency_mult", 0.0)),
            "book_mid_gap": float(snap.mid - ((snap.bid + snap.ask) / 2.0)),
            "ev_est": 0.0,
            "ev_req": 0.0,
            "stake_base": 0.0,
            "stake_scaled": 0.0,
            "min_edge": float(self.min_edge),
            "min_ev_per_dollar": float(self.min_ev_per_dollar),
            "sports_fair": float(sports_fair) if sports_fair is not None else 0.0,
            "fair_source": fair_source,
            "action": "",
            "reason": "",
        }

        if edge_abs < self.min_edge:
            sig = Signal(
                ts=snap.ts,
                market_id=snap.market_id,
                p_fair=float(p_fair),
                p_exec=float(p_exec),
                edge=float(edge),
                regime=str(feats["regime_mode"]),
                toxicity=float(feats["toxicity"]),
                action="HOLD",
                size=0.0,
                reason="edge_too_small",
            )
            meta = {
                **base_meta,
                "action": "HOLD",
                "reason": "edge_too_small",
            }
            return sig, meta

        side = "BUY" if edge > 0 else "SELL"
        regime_mode = str(feats["regime_mode"])
        trend_strength = float(feats["trend_strength"])
        regime_thin = bool(float(feats["spread"]) >= 0.05)

        stake_base = float(
            stake_ladder(
                edge_net=float(edge),
                regime_thin=regime_thin,
                min_stake=20.0,
            )
        )


        dir_scale = float(
            directional_edge_scale(
                edge_net=float(edge),
                regime_mode=regime_mode,
                trend_strength=trend_strength,
            )
        )

        trend_scale = 1.0
        if regime_mode == "TREND":
            trend_scale = 0.90 + 0.35 * trend_strength

        stake = float(stake_base * dir_scale * trend_scale)
        stake = float(max(0.0, min(stake, bankroll * self.bankroll_cap_pct, 65.0)))

        if stake <= 0.0:
            sig = Signal(
                ts=snap.ts,
                market_id=snap.market_id,
                p_fair=float(p_fair),
                p_exec=float(p_exec),
                edge=float(edge),
                regime=regime_mode,
                toxicity=float(feats["toxicity"]),
                action="HOLD",
                size=0.0,
                reason="stake_zero",
            )
            meta = {
                **base_meta,
                "stake_base": float(stake_base),
                "stake_scaled": 0.0,
                "action": "HOLD",
                "reason": "stake_zero",
            }
            return sig, meta

        ev_est, _, _ = trade_ev_reversion_exit_from_fill(
            side=side,
            stake=float(stake),
            p_fill_yes=float(snap.ask if side == "BUY" else snap.bid),
            mid=float(snap.mid),
            p_exec=float(p_exec),
            spread=float(snap.spread),
            slippage_bps=0.0,
            regime_thin=regime_thin,
            tox_reg=str(feats["tox_reg"]),
            mid_delta=float(feats["mid_delta"]),
            vol=float(feats["vol_proxy"]),
            regime_mode=regime_mode,
            exit_extra_mult=1.0,
        )

        ev_req = float(stake) * self.min_ev_per_dollar

        if ev_est < ev_req:
            sig = Signal(
                ts=snap.ts,
                market_id=snap.market_id,
                p_fair=float(p_fair),
                p_exec=float(p_exec),
                edge=float(edge),
                regime=regime_mode,
                toxicity=float(feats["toxicity"]),
                action="HOLD",
                size=0.0,
                reason="ev_gate_block",
            )
            meta = {
                **base_meta,
                "ev_est": float(ev_est),
                "ev_req": float(ev_req),
                "stake_base": float(stake_base),
                "stake_scaled": float(stake),
                "action": "HOLD",
                "reason": "ev_gate_block",
            }
            return sig, meta

        sig = Signal(
            ts=snap.ts,
            market_id=snap.market_id,
            p_fair=float(p_fair),
            p_exec=float(p_exec),
            edge=float(edge),
            regime=regime_mode,
            toxicity=float(feats["toxicity"]),
            action=side,
            size=float(stake),
            reason="edge_and_ev_ok",
        )
        meta = {
            **base_meta,
            "ev_est": float(ev_est),
            "ev_req": float(ev_req),
            "stake_base": float(stake_base),
            "stake_scaled": float(stake),
            "action": str(side),
            "reason": "edge_and_ev_ok",
        }
        return sig, meta
