import numpy as np
from typing import Tuple

from shared.math_utils import clip01
from shared.execution import _edge_extra_slip, _size_extra_slip, exit_fill_yes
from shared.signals import trend_strength_score

def trade_ev(side: str, stake: float, p0: float, p_fair: float, spr: float, slippage_bps: float) -> float:
    stake = float(stake)
    p0 = clip01(p0)
    p_fair = clip01(p_fair)
    spr = float(max(spr, 0.0))
    slip = float(slippage_bps) / 10000.0

    if side == "BUY":
        shares = stake / p0
        exp_pnl = shares * p_fair - stake
    else:
        q0 = clip01(1.0 - p0)
        q_fair = clip01(1.0 - p_fair)
        shares = stake / q0
        exp_pnl = shares * q_fair - stake

    per_side = (spr / 2.0) + slip
    costs = stake * (per_side + per_side)
    return float(exp_pnl - costs)


# ----------------------------
# Consistent EV calculation at adverse fill
# ----------------------------
def trade_ev_from_fill(
    side: str,
    stake: float,
    p_fill_yes: float,
    p_fair: float,
    spread: float,
    slippage_bps: float,
    edge_net: float,
    regime_thin: bool,
    exit_extra_mult: float = 1.0,
) -> float:
    """EV computed consistently with adverse entry fills.

    We assume the entry already occurred at `p_fill_yes` (which embeds bid/ask + slip + adverse extra).
    For gating, we subtract a conservative *expected exit* friction.
    """
    stake = float(stake)
    p_fill_yes = clip01(p_fill_yes)
    p_fair = clip01(p_fair)

    spr = float(max(spread, 0.0))
    slip = float(slippage_bps) / 10000.0

    # Conservative exit friction estimate (as fraction of stake)
    # Include edge-based adverse selection AND size/liquidity impact.
    extra_edge = _edge_extra_slip(edge_net=edge_net, regime_thin=regime_thin)

    # approximate size effect on exit using an effective price
    if side == "BUY":
        eff_price = p_fill_yes
    else:
        eff_price = clip01(1.0 - p_fill_yes)

    extra_size = _size_extra_slip(
        stake=float(stake),
        vol=50.0,              # gate-time liquidity proxy; conservative constant
        price=float(eff_price),
        regime_thin=bool(regime_thin),
    )

    exit_cost_frac = (spr / 2.0) + slip + exit_extra_mult * extra_edge + extra_size
    exit_cost_frac = float(np.clip(exit_cost_frac, 0.0, 0.08))

    if side == "BUY":
        shares = stake / p_fill_yes
        exp_pnl = shares * p_fair - stake
    else:
        q_fill = clip01(1.0 - p_fill_yes)
        q_fair = clip01(1.0 - p_fair)
        shares = stake / q_fill
        exp_pnl = shares * q_fair - stake

    return float(exp_pnl - stake * exit_cost_frac)


def roundtrip_cost_gate(
    side: str,
    mid: float,
    spread: float,
    slippage_bps: float,
    p_fill_yes: float,
    stake: float,
    edge_net: float,
    regime_thin: bool,
    mid_delta: float,
    buffer: float = 0.002,
) -> float:
    """Return required_edge_prob.

    required_edge_prob is the minimum abs(p_exec-mid) needed to cover:
      - spread (roundtrip)
      - slippage (roundtrip)
      - observed adverse entry extra (in prob units)
      - conservative expected adverse exit extra
      - small safety buffer

    This prevents 'paper EV' that can't survive realistic fills.
    """
    mid = clip01(mid)
    spr = float(max(spread, 0.0))
    slip = float(slippage_bps) / 10000.0

    # Observed entry distance from mid includes half-spread + slip + adverse extra
    entry_dist = float(abs(clip01(p_fill_yes) - mid))
    adverse_entry = float(max(0.0, entry_dist - (spr / 2.0) - slip))

    # Conservative expected exit adverse extra: larger in THIN / fast mid
    exit_mult = 1.3
    if regime_thin:
        exit_mult = 1.8
    if mid_delta >= 0.010:
        exit_mult = max(exit_mult, 2.0)

    # Edge-based + size-based exit adverse extra estimate
    edge_like = float(abs(edge_net))
    extra_edge = _edge_extra_slip(edge_net=edge_like, regime_thin=regime_thin)

    if side == "BUY":
        eff_price = clip01(p_fill_yes)
    else:
        eff_price = clip01(1.0 - p_fill_yes)

    extra_size = _size_extra_slip(
        stake=float(stake),
        vol=50.0,  # gate-time conservative liquidity proxy
        price=float(eff_price),
        regime_thin=bool(regime_thin),
    )

    adverse_exit = float(np.clip(exit_mult * (extra_edge + extra_size), 0.0, 0.05))

    required = float(spr + 2.0 * slip + adverse_entry + adverse_exit + float(buffer))
    return required


def reversion_lambda(regime_thin: bool, tox_reg: str, mid_delta: float, regime_mode: str) -> float:
    """Expected fraction of dislocation captured on exit.

    MEANREV regime:
      higher reversion capture
    TREND regime:
      lower reversion capture
    """
    if regime_mode == "TREND":
        if tox_reg == "SOFT":
            lam = 0.40
        elif tox_reg == "TOXIC":
            lam = 0.20
        else:
            lam = 0.30
    else:
        if tox_reg == "SOFT":
            lam = 0.75
        elif tox_reg == "TOXIC":
            lam = 0.45
        else:
            lam = 0.65

    if regime_thin:
        lam -= 0.05

    if mid_delta >= 0.010:
        lam -= 0.05
    elif mid_delta >= 0.006:
        lam -= 0.02

    return float(np.clip(lam, 0.15, 0.80))


def trend_continuation_kappa(mid_delta: float, tox_reg: str, regime_thin: bool) -> float:
    """Expected continuation strength in TREND regime.

    Higher kappa => expected exit moves beyond p_exec in the same direction.
    """
    k = 0.70

    if mid_delta >= 0.012:
        k += 0.35
    elif mid_delta >= 0.008:
        k += 0.20
    elif mid_delta >= 0.005:
        k += 0.10

    if tox_reg == "TOXIC":
        k += 0.15
    elif tox_reg == "SOFT":
        k -= 0.10

    if regime_thin:
        k -= 0.10

    return float(np.clip(k, 0.30, 1.20))


def trade_ev_reversion_exit_from_fill(
    side: str,
    stake: float,
    p_fill_yes: float,
    mid: float,
    p_exec: float,
    spread: float,
    slippage_bps: float,
    regime_thin: bool,
    tox_reg: str,
    mid_delta: float,
    vol: float,
    regime_mode: str,
    exit_extra_mult: float = 1.0,
) -> Tuple[float, float, float]:
    """Expected EV using partial reversion then execution-aware expected exit fill.

    Returns: (ev, expected_exit_mid, lambda_reversion)
    """
    stake = float(stake)
    p_fill_yes = clip01(p_fill_yes)
    mid = clip01(mid)
    p_exec = clip01(p_exec)
    if regime_mode == "TREND":
        lam = 0.0
        kappa = trend_continuation_kappa(
            mid_delta=mid_delta,
            tox_reg=tox_reg,
            regime_thin=regime_thin,
        )

        # Refine TREND continuation by tape strength.
        ts = trend_strength_score(
            mid_delta=float(mid_delta),
            tox_reg=str(tox_reg),
            run_signal=0.0,
            pace_signal=0.0,
            spread=float(spread),
        )
        kappa *= float(0.90 + 0.25 * ts)   # modest scaler only
        kappa = float(np.clip(kappa, 0.25, 1.35))

        p_exit_mid = clip01(p_exec + kappa * (p_exec - mid))
    else:
        lam = reversion_lambda(
            regime_thin=regime_thin,
            tox_reg=tox_reg,
            mid_delta=mid_delta,
            regime_mode=regime_mode,
        )
        p_exit_mid = clip01(mid + lam * (p_exec - mid))

    edge_proxy = float(abs(p_exit_mid - p_fill_yes))
    p_exit_fill_yes = exit_fill_yes(
        side=side,
        mid=float(p_exit_mid),
        spread=float(spread),
        edge_proxy=edge_proxy,
        regime_thin=bool(regime_thin),
        slippage_bps=float(slippage_bps),
        stake=float(stake),
        vol=float(vol),
        exit_extra_mult=float(exit_extra_mult),
    )

    if side == "BUY":
        shares = stake / p_fill_yes
        ev = shares * (p_exit_fill_yes - p_fill_yes)
    else:
        q_fill = clip01(1.0 - p_fill_yes)
        q_exit = clip01(1.0 - p_exit_fill_yes)
        shares = stake / q_fill
        ev = shares * (q_exit - q_fill)

    return float(ev), float(p_exit_mid), float(lam)