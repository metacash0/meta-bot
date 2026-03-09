import numpy as np

# ----------------------------
# Execution realism helpers (dynamic alpha + roundtrip cost gate)
# ----------------------------

def runner_ttl_bonus(trend_strength: float, base_ttl: int = 2, max_ttl: int = 6) -> int:
    """Extra TTL for TREND runners based on local trend strength.

    Weak runner -> small bonus
    Strong runner -> larger bonus
    """
    ts = float(np.clip(trend_strength, 0.0, 1.0))
    bonus = base_ttl + int(round((max_ttl - base_ttl) * ts))
    return int(np.clip(bonus, base_ttl, max_ttl))


def allow_trend_reentry(
    regime_mode: str,
    tox_reg: str,
    edge_abs: float,
    mid_delta: float,
    last_reentry_t: int,
    t: int,
    max_per_night_used: int,
    max_per_night: int,
    edge_min: float,
    middelta_max: float,
    toxic_block: bool,
    cooldown: int,
) -> bool:
    if regime_mode != "TREND":
        return False
    if toxic_block and tox_reg == "TOXIC":
        return False
    if edge_abs < edge_min:
        return False
    if mid_delta > middelta_max:
        return False
    if max_per_night_used >= max_per_night:
        return False
    if (t - last_reentry_t) <= cooldown:
        return False
    return True


def ttl_by_regime(regime_mode: str, tox_reg: str, regime_thin: bool, ttl_trend: int = 10, ttl_meanrev: int = 5) -> int:
    """Return holding TTL based on regime.

    TREND: longer leash
    MEANREV: shorter leash
    Toxic/thin conditions shorten TTL a bit.
    """
    ttl = ttl_trend if regime_mode == "TREND" else ttl_meanrev

    if tox_reg == "TOXIC":
        ttl -= 2
    elif tox_reg == "SOFT" and regime_mode == "TREND":
        ttl += 1

    if regime_thin:
        ttl -= 1

    return int(np.clip(ttl, 3, 12))