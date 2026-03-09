import numpy as np
from typing import Tuple

from shared.math_utils import clip01

# ----------------------------
# Adverse selection fill helpers
# ----------------------------
def _edge_extra_slip(edge_net: float, regime_thin: bool, k: float = 0.25) -> float:
    """Extra adverse-selection slippage as a fraction of price (edge-based component)."""
    extra = k * float(abs(edge_net))
    if regime_thin:
        extra *= 1.5
    return float(np.clip(extra, 0.0, 0.03))


def _size_extra_slip(
    stake: float,
    vol: float,
    price: float,
    regime_thin: bool,
    k_size: float = 0.20,
    vol_scale: float = 50.0,
) -> float:
    """Size/liquidity slippage component as a fraction of price.

    Use shares (stake/price) so the penalty scales with how much size you push through the book.
    `vol` is the market state's pseudo-liquidity metric.
    """
    stake = float(max(stake, 0.0))
    vol = float(max(vol, 1.0))
    price = float(max(price, 1e-6))

    shares = stake / price
    extra = k_size * (shares / (vol * vol_scale))
    if regime_thin:
        extra *= 1.5
    return float(np.clip(extra, 0.0, 0.03))


def entry_fill_yes(
    side: str,
    mid: float,
    spread: float,
    edge_net: float,
    regime_thin: bool,
    slippage_bps: float,
    stake: float,
    vol: float,
) -> float:
    """Return entry fill as YES price.

    BUY: pay ask (mid + spr/2) + slip + extra
    SELL: buy NO at ask in NO space, then convert to YES price
    """
    mid = clip01(mid)
    spr = float(np.clip(spread, 0.0, 0.25))
    slip = float(slippage_bps) / 10000.0
    # Use an effective price for share-based size slippage
    if side == "BUY":
        eff_price = mid
    else:
        eff_price = clip01(1.0 - mid)

    extra = _edge_extra_slip(edge_net=edge_net, regime_thin=regime_thin) + _size_extra_slip(
        stake=stake,
        vol=vol,
        price=eff_price,
        regime_thin=regime_thin,
    )
    # Cap combined adverse selection so we don't annihilate all trade frequency.
    extra = float(np.clip(extra, 0.0, 0.015))

    if side == "BUY":
        return clip01(mid + spr / 2.0 + slip + extra)

    # SELL => long NO
    q_mid = clip01(1.0 - mid)
    q_fill = clip01(q_mid + spr / 2.0 + slip + extra)
    return clip01(1.0 - q_fill)


def passive_fill_probability(
    regime_thin: bool,
    tox_reg: str,
    mid_delta: float,
    spread: float,
    vol: float,
) -> float:
    """Probability our entry gets a passive/maker-style fill.

    Higher when:
      - not thin
      - lower toxicity
      - spread is decent but not too wide
      - mid is stable
      - liquidity is reasonable
    """
    p = 0.0

    if tox_reg == "SOFT":
        p = 0.75
    elif tox_reg == "NEUTRAL":
        p = 0.35
    else:
        p = 0.05

    if regime_thin:
        p *= 0.35

    if mid_delta >= 0.010:
        p *= 0.25
    elif mid_delta >= 0.006:
        p *= 0.60

    if spread < 0.02:
        p *= 0.80
    elif spread > 0.06:
        p *= 0.50

    if vol < 15:
        p *= 0.60
    elif vol > 80:
        p *= 1.10

    return float(np.clip(p, 0.0, 0.90))


def entry_fill_yes_hybrid(
    rng,
    side: str,
    mid: float,
    spread: float,
    edge_net: float,
    regime_thin: bool,
    slippage_bps: float,
    stake: float,
    vol: float,
    tox_reg: str,
    mid_delta: float,
) -> Tuple[float, bool]:
    """Hybrid maker/taker entry fill model.

    Returns:
      (fill_yes_price, was_passive)

    Passive logic:
      - BUY: fill somewhere between mid and ask, biased toward mid
      - SELL (long NO): in NO-space, fill between mid and ask, then convert back

    Taker logic:
      - use existing entry_fill_yes(...)
    """
    mid = clip01(mid)
    spr = float(np.clip(spread, 0.0, 0.25))

    p_passive = passive_fill_probability(
        regime_thin=bool(regime_thin),
        tox_reg=str(tox_reg),
        mid_delta=float(mid_delta),
        spread=float(spread),
        vol=float(vol),
    )

    if rng.random() < p_passive:
        # Passive/maker-style fill: improve relative to full taker ask.
        # Use a point between mid and ask with mild adverse tilt.
        # BUY YES: price between mid and mid+spr/2
        if side == "BUY":
            # closer to mid in softer conditions; slightly worse in neutral
            frac = 0.20 if tox_reg == "SOFT" else 0.35
            fill = mid + frac * (spr / 2.0)
            return clip01(fill), True

        # SELL => long NO, do passive fill in NO space, then convert back to YES
        q_mid = clip01(1.0 - mid)
        frac = 0.20 if tox_reg == "SOFT" else 0.35
        q_fill = q_mid + frac * (spr / 2.0)
        return clip01(1.0 - q_fill), True

    # Fallback: taker/adverse fill
    fill = entry_fill_yes(
        side=side,
        mid=float(mid),
        spread=float(spread),
        edge_net=float(edge_net),
        regime_thin=bool(regime_thin),
        slippage_bps=float(slippage_bps),
        stake=float(stake),
        vol=float(vol),
    )
    return float(fill), False


def post_limit_yes_price(
    side: str,
    mid: float,
    spread: float,
    tox_reg: str,
) -> float:
    """Return the posted passive YES-equivalent limit price.

    BUY:
      post between mid and bid/ask center, closer to mid.
    SELL (long NO):
      post in NO-space, then convert back to YES.
    """
    mid = clip01(mid)
    spr = float(np.clip(spread, 0.0, 0.25))

    # In soft markets we can post more aggressively; in neutral slightly less so.
    frac = 0.10 if tox_reg == "SOFT" else 0.20

    if side == "BUY":
        # post a passive bid just above mid minus half-spread center
        # but keep it inside the spread
        px = mid + frac * (spr / 2.0)
        return clip01(px)

    # SELL => long NO; post passive order in NO-space and convert back
    q_mid = clip01(1.0 - mid)
    q_px = q_mid + frac * (spr / 2.0)
    return clip01(1.0 - q_px)


def post_quote_yes_price(
    side: str,
    mid: float,
    spread: float,
    tox_reg: str,
) -> float:
    """Conservative passive quote placement for inventory quoting.

    These quotes should be posted closer to mid than directional maker orders,
    because the inventory engine monetizes small edges and is more vulnerable
    to being picked off.
    """
    mid = clip01(mid)
    spr = float(np.clip(spread, 0.0, 0.25))

    # More conservative than the directional posting helper.
    if tox_reg == "SOFT":
        frac = 0.03
    else:
        frac = 0.08

    if side == "BUY":
        px = mid + frac * (spr / 2.0)
        return clip01(px)

    q_mid = clip01(1.0 - mid)
    q_px = q_mid + frac * (spr / 2.0)
    return clip01(1.0 - q_px)


def passive_fill_hit(
    side: str,
    limit_yes: float,
    mid: float,
    prev_mid: float,
    spread: float,
    tox_reg: str,
    mid_delta: float,
    vol: float,
    queue_pos: float,
    touches: int,
    rng,
) -> Tuple[bool, float, int]:
    """Queue-aware passive fill model.

    Returns:
      (filled, new_queue_pos, new_touches)

    Intuition:
      - resting longer improves queue priority
      - when market moves toward our quote, fill odds improve
      - when market moves away, queue priority decays slightly
      - toxicity and low vol reduce fill odds
    """
    mid = clip01(mid)
    prev_mid = clip01(prev_mid)
    spr = float(np.clip(spread, 0.0, 0.25))
    q = float(np.clip(queue_pos, 0.0, 1.0))
    tchs = int(touches)

    # Base fill chance by toxicity
    if tox_reg == "SOFT":
        p = 0.35
    elif tox_reg == "NEUTRAL":
        p = 0.18
    else:
        p = 0.03

    # Detect whether market moved toward our resting quote
    moved_toward = False
    moved_away = False

    if side == "BUY":
        if mid <= prev_mid:
            moved_toward = True
        elif mid > prev_mid + 0.002:
            moved_away = True
    else:
        if mid >= prev_mid:
            moved_toward = True
        elif mid < prev_mid - 0.002:
            moved_away = True

    # Touch logic: if market is near our posted quote, count a touch
    if abs(mid - limit_yes) <= 0.006:
        tchs += 1
        p *= 1.55

    if moved_toward:
        p *= 1.45
        q -= 0.12   # queue improves when market drifts toward us
    if moved_away:
        p *= 0.80
        q += 0.04   # queue degrades when market drifts away

    # Resting time / touches improve effective fill chance
    p *= (1.0 + 0.30 * min(tchs, 3))

    # Queue priority effect: lower q => better chance
    p *= (1.30 - 0.60 * np.clip(q, 0.0, 1.0))

    # Faster tape hurts passive fills
    if mid_delta >= 0.010:
        p *= 0.35
    elif mid_delta >= 0.006:
        p *= 0.65

    # Wider spread can help passive posting, but too wide usually means toxic
    if 0.02 <= spr <= 0.05:
        p *= 1.10
    elif spr > 0.06:
        p *= 0.80

    if vol < 15:
        p *= 0.65
    elif vol > 80:
        p *= 1.10

    q = float(np.clip(q, 0.0, 1.0))
    p = float(np.clip(p, 0.0, 0.95))
    filled = bool(rng.random() < p)

    return filled, q, tchs


def initial_queue_position(
    tox_reg: str,
    spread: float,
    vol: float,
    rng,
) -> float:
    """Initial queue priority in [0, 1].

    Lower is better. We assume:
      - softer conditions and higher vol => better queue access
      - wider spread => more uncertainty / worse queue
    """
    if tox_reg == "SOFT":
        base = 0.18
    elif tox_reg == "NEUTRAL":
        base = 0.38
    else:
        base = 0.65

    spr = float(np.clip(spread, 0.0, 0.25))
    if spr > 0.06:
        base += 0.10
    elif spr < 0.03:
        base -= 0.05

    if vol > 80:
        base -= 0.05
    elif vol < 15:
        base += 0.10

    base += float(rng.normal(0.0, 0.06))
    return float(np.clip(base, 0.0, 1.0))


def exit_fill_yes(
    side: str,
    mid: float,
    spread: float,
    edge_proxy: float,
    regime_thin: bool,
    slippage_bps: float,
    stake: float = 0.0,
    vol: float = 50.0,
    exit_extra_mult: float = 1.0,
) -> float:
    """Return exit fill as YES price.

    BUY exits by selling YES at bid (mid - spr/2) minus slip and adverse-selection extra.
    SELL exits by selling NO at bid in NO space, then convert back.

    We include:
      - base bid/ask + slippage
      - edge-based adverse selection
      - size/liquidity slippage (shares/vol)
    `exit_extra_mult` lets us make flip/TTL exits more conservative.
    """
    mid = clip01(mid)
    spr = float(np.clip(spread, 0.0, 0.25))
    slip = float(slippage_bps) / 10000.0

    # Edge proxy is in probability units; convert to an edge-like magnitude.
    edge_like = float(abs(edge_proxy))

    # Use an effective price for share-based size slippage
    if side == "BUY":
        eff_price = mid
    else:
        eff_price = clip01(1.0 - mid)

    extra = exit_extra_mult * _edge_extra_slip(edge_net=edge_like, regime_thin=regime_thin)
    extra += _size_extra_slip(
        stake=float(stake),
        vol=float(vol),
        price=float(eff_price),
        regime_thin=bool(regime_thin),
    )
    extra = float(np.clip(extra, 0.0, 0.015))

    if side == "BUY":
        return clip01(mid - spr / 2.0 - slip - extra)

    q_mid = clip01(1.0 - mid)
    q_fill = clip01(q_mid - spr / 2.0 - slip - extra)
    return clip01(1.0 - q_fill)