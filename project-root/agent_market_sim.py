import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

from shared.math_utils import expit, logit, clip01
from shared.pf import LogitBiasParticleFilter

from shared.signals import (
    exec_alpha_dynamic,
    toxicity_score,
    toxicity_regime,
    pf_latency_multiplier,
    market_regime,
    trend_strength_score,
)

from shared.sizing import (
    stake_ladder,
    directional_edge_scale,
    quote_stake,
)

from shared.execution import (
    _edge_extra_slip,
    _size_extra_slip,
    entry_fill_yes,
    passive_fill_probability,
    entry_fill_yes_hybrid,
    post_limit_yes_price,
    post_quote_yes_price,
    passive_fill_hit,
    initial_queue_position,
    exit_fill_yes,
)

from shared.risk_models import (
    trade_ev,
    trade_ev_from_fill,
    roundtrip_cost_gate,
    reversion_lambda,
    trend_continuation_kappa,
    trade_ev_reversion_exit_from_fill,
)

from shared.exits import (
    runner_ttl_bonus,
    allow_trend_reentry,
    ttl_by_regime,
)

# ----------------------------
# Market microstructure (very simple)
# ----------------------------
@dataclass
class MarketState:
    mid: float          # market mid price for YES (Over)
    spread: float       # quoted spread
    inventory: float    # market maker inventory (YES shares)
    vol: float          # pseudo volume metric

def market_step(
    state: MarketState,
    net_order_yes_shares: float,
    mm_kappa: float,
    impact: float,
    inv_penalty: float,
    spread_floor: float,
    spread_slope: float,
):
    """
    net_order_yes_shares > 0 => net BUY YES pressure; < 0 => net SELL YES pressure.
    We update mid with:
      - price impact from net order
      - mean reversion (mm_kappa) toward 0.5-ish
      - inventory penalty pushes mid against inventory
    """
    mid = state.mid
    inv = state.inventory

    # price impact
    mid = mid + impact * net_order_yes_shares

    # mean reversion (market maker pulls toward fair anchor 0.5 for demo)
    mid = mid + mm_kappa * (0.50 - mid)

    # inventory penalty: if MM is long YES inventory, he shades price down
    mid = mid - inv_penalty * inv

    mid = clip01(mid)

    # update inventory (MM takes the other side of net flow)
    inv = inv - net_order_yes_shares

    # update spread based on "stress" (|flow| + |inv|)
    stress = abs(net_order_yes_shares) + 0.1 * abs(inv)
    spr = max(spread_floor, spread_floor + spread_slope * stress)

    # update pseudo volume
    vol = 0.8 * state.vol + 0.2 * (abs(net_order_yes_shares) * 100.0 + 10.0)

    state.mid = mid
    state.inventory = inv
    state.spread = float(np.clip(spr, 0.01, 0.20))
    state.vol = float(np.clip(vol, 1.0, 500.0))

# ----------------------------
# Agents
# ----------------------------
def agent_recs(rng, p_obs: float, mid: float) -> float:
    """
    Recreational bettors: small noisy flow, mildly trend-following.
    Returns YES shares demand.
    """
    noise = rng.normal(0, 3.0)
    tilt = 10.0 * (p_obs - mid)  # chase perceived edge
    return float(0.2 * tilt + noise)

def agent_sharp(rng, p_true_noisy: float, mid: float) -> float:
    """
    Sharp: lower-noise signal, bigger size, trades when mispriced.
    """
    edge = p_true_noisy - mid
    if abs(edge) < 0.01:
        return 0.0
    size = 40.0 * np.tanh(edge / 0.03)  # saturating
    return float(size)

def agent_arb(rng, book_price: float, mid: float) -> float:
    """
    Arb: if book implies meaningfully different prob, pushes market toward book.
    """
    edge = book_price - mid
    if abs(edge) < 0.008:
        return 0.0
    size = 25.0 * np.tanh(edge / 0.02)
    return float(size)


def adverse_cluster_flow_fn(mid: float, p_exec: float, strength: float) -> float:
    """Short-horizon adverse flow triggered after our fills.

    If fair is above mid, crowd/sharps are more likely to buy YES;
    if fair is below mid, more likely to sell YES.
    `strength` controls the magnitude and decays over time.
    """
    edge = float(p_exec - mid)
    if abs(edge) < 1e-6 or strength <= 0.0:
        return 0.0
    return float(18.0 * strength * np.tanh(edge / 0.02))

# ----------------------------
# Our strategy (totals only) - EV gate + ladder + TTL
# ----------------------------
@dataclass
class Position:
    t_open: int
    side: str          # "BUY" or "SELL" (BUY = long YES; SELL = long NO)
    stake: float
    fill_yes: float    # entry fill expressed as YES price (even for SELL)
    spread: float
    ev: float
    edge_bucket: str   # attribution bucket based on |edge_net| at entry
    is_quote: bool = False
    regime_mode: str = "MEANREV"
    ttl_bonus: int = 0
    is_runner: bool = False
    is_reentry: bool = False


@dataclass
class PendingOrder:
    t_post: int
    side: str           # "BUY" or "SELL"
    stake: float
    limit_yes: float    # posted YES-equivalent price
    ttl: int            # max life in ticks
    edge_bucket: str
    tox_reg: str
    edge_net: float
    ev_at_post: float
    queue_pos: float    # 0..1, smaller means better queue priority
    touches: int        # number of favorable touch opportunities seen
    is_reentry: bool = False


@dataclass
class QuoteOrder:
    t_post: int
    side: str             # "BUY" or "SELL"
    stake: float
    limit_yes: float
    ttl: int
    edge_net: float
    tox_reg: str
    queue_pos: float
    touches: int


def pos_mtm_pnl(pos: Position, p_now: float) -> float:
    p0 = clip01(pos.fill_yes)
    p_now = clip01(p_now)
    stake = float(pos.stake)

    if pos.side == "BUY":
        shares = stake / p0
        return shares * (p_now - p0)

    # SELL means long NO => value uses q = 1-p
    q0 = clip01(1.0 - p0)
    q_now = clip01(1.0 - p_now)
    shares = stake / q0
    return shares * (q_now - q0)


def decide_trade(edge_net: float, ci_width: float, spread: float, regime_thin: bool,
                edge_thr_net=0.04, max_ci=0.10, max_spread=0.06) -> str:
    if spread > max_spread:
        return "HOLD(spread)"
    if ci_width > max_ci:
        return "HOLD(ci)"

    thr = edge_thr_net * (1.5 if regime_thin else 1.0)
    if edge_net >= thr:
        return "BUY"
    if edge_net <= -thr:
        return "SELL"
    return "HOLD(edge)"


# ----------------------------
# Game process: "true" Over probability evolves with 2 factors (pace + strength)
# ----------------------------
def simulate_true_prob_path(rng, T: int, p0=0.50, nu_pace=3.0, nu_str=3.0, load_pace=0.10, load_str=0.08):
    """
    Latent logit process:
      logit(p_t) = logit(p0) + pace_t + strength_t
    pace_t and strength_t have heavy tails (t-distributed shocks).
    """
    x = logit(np.array([p0]))[0]
    pace = 0.0
    strength = 0.0
    out = []
    for _ in range(T):
        # heavy-tailed innovations
        e1 = rng.standard_t(df=nu_pace)
        e2 = rng.standard_t(df=nu_str)
        pace = 0.85 * pace + load_pace * e1
        strength = 0.90 * strength + load_str * e2
        x_t = x + pace + strength
        out.append(float(expit(np.array([x_t]))[0]))
    return np.array(out)

# ----------------------------
# One "night" simulation
# ----------------------------
def run_one_night(
    seed: int,
    T: int = 60,
    bankroll_start: float = 1000.0,
    bad_night_pct: float = 0.15,
    max_open: int = 3,
    ttl_ticks: int = 6,
    slippage_bps: float = 10.0,
    # Back-compat: if `min_ev_per_dollar` is provided, gate on EV/stake; else use absolute $ min_ev.
    min_ev: float = 0.50,
    min_ev_per_dollar: Optional[float] = 0.015,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)

    # True probability path (unknown to agents)
    p_true = simulate_true_prob_path(rng, T=T, p0=0.50)

    # -------------------------------------------------
    # Synthetic live NBA signals (pace shock + score-run)
    # -------------------------------------------------
    # pace_z: persistent AR(1) heavy-tail shocks (tempo/possession-rate regime)
    pace_z = np.zeros(T, dtype=float)
    for t in range(1, T):
        pace_z[t] = 0.85 * pace_z[t - 1] + 0.55 * rng.standard_t(df=4.0)

    # run_z: transient bursts (short-term scoring streaks / overreaction driver)
    run_z = np.zeros(T, dtype=float)
    for t in range(1, T):
        run_z[t] = 0.55 * run_z[t - 1] + 0.90 * rng.standard_t(df=6.0)
        # occasional big run events
        if rng.random() < 0.06:
            run_z[t] += rng.normal(0.0, 3.0)

    # Strength of signals injected into OUR baseline model
    # (tune these; larger => more alpha but harder to execute)
    PACE_ALPHA = 0.07
    RUN_FADE_ALPHA = 0.05

    # Execution haircut: dynamic per tick (harder execution => shrink fair value toward market)
    # NOTE: adverse-selection fills already penalize execution; this adds a regime/volatility haircut.
    EXEC_ALPHA_BASE = 0.85
    EXEC_ALPHA_THIN = 0.60

    # Baseline model (ours): noisy version of truth + pace detection + run-fade
    # Using logit-space so adjustments stay in [0,1]
    base_logit = logit(np.clip(p_true, 0.01, 0.99))

    # pace pushes probability around persistently
    base_logit = base_logit + PACE_ALPHA * np.tanh(pace_z / 2.5)

    # run-fade: when a short-term run is large, we fade it slightly (mean reversion)
    base_logit = base_logit - RUN_FADE_ALPHA * np.tanh(run_z / 2.0)

    # add model noise (represents imperfections in possessions model)
    base_logit = base_logit + rng.normal(0, 0.10, size=T)

    p_base = np.clip(expit(base_logit), 0.01, 0.99)

    # Sportsbook "consensus" is noisy and lags a bit; also partially reacts to runs
    p_book = np.clip(
        0.60 * p_true
        + 0.30 * np.roll(p_true, 1)
        + 0.10 * np.clip(expit(logit(p_true) + 0.03 * np.tanh(run_z / 2.0)), 0.01, 0.99)
        + rng.normal(0, 0.02, size=T),
        0.01,
        0.99,
    )
    p_book[0] = np.clip(p_true[0] + rng.normal(0, 0.02), 0.01, 0.99)

    # Polymarket state starts near 0.5
    mkt = MarketState(mid=0.50, spread=0.03, inventory=0.0, vol=50.0)

    # Our PF
    pf = LogitBiasParticleFilter(N=3000, process_vol=0.02, seed=seed + 99)

    bankroll = float(bankroll_start)
    realized = 0.0
    expected = 0.0
    positions: List[Position] = []
    pending_orders: List[PendingOrder] = []
    quote_orders: List[QuoteOrder] = []
    peak = bankroll
    max_dd = 0.0
    n_trades = 0
    n_skips_ev = 0

    prev_sig_raw: str = "HOLD"
    prev_mid = float(mkt.mid)
    prev_spread = float(mkt.spread)

    exit_tp = 0
    exit_flip = 0
    exit_ttl = 0
    realized_tp = 0.0
    realized_flip = 0.0
    realized_ttl = 0.0
    realized_eod = 0.0
    entry_extras = []  # track adverse entry extras for diagnostics
    exit_extras = []   # track adverse exit extras for diagnostics
    edge_raw_abs_list = []
    edge_exec_abs_list = []
    entry_dist_abs_list = []
    edge_net_abs_list = []
    soft_gate_used_ticks = 0
    soft_gate_trades = 0
    soft_gate_skips = 0
    ev_old_assump = []
    ev_reversion_assump = []
    rev_lam_list = []
    exp_exit_move_list = []
    trend_kappas = []
    trend_strengths = []
    trend_strength_trade = []
    trend_scale_used = []
    dir_edge_scale_used = []
    dir_edge_scale_trend = []
    dir_edge_scale_meanrev = []
    exp_exit_modes = {"MEANREV": 0, "TREND": 0}
    regime_ev_haircuts = {"MEANREV": [], "TREND": []}
    passive_fills = 0
    taker_fills = 0
    passive_entry_dists = []
    taker_entry_dists = []
    posted_orders = 0
    posted_fills = 0
    posted_cancels = 0
    taker_fallback_fills = 0
    pending_live_sum = 0.0
    maker_entry_dists = []
    fallback_entry_dists = []
    queue_pos_posted = []
    queue_pos_filled = []
    queue_touches_filled = []
    quote_posts = 0
    quote_fills = 0
    quote_cancels = 0
    quote_fill_dists = []
    quote_live_sum = 0.0
    quote_realized = 0.0
    quote_trades_soft = 0
    quote_trades_neutral = 0
    shock_events = 0
    shock_flow_abs = []
    adverse_cluster_strength = 0.0
    adverse_cluster_ttl = 0
    adverse_cluster_activations = 0
    adverse_cluster_flow = []
    adverse_cluster_ticks = 0
    pf_latency_mults = []
    pf_process_vols = []
    exit_early = 0
    realized_early = 0.0
    ttl_used_trend = []
    ttl_used_meanrev = []
    tp_frac_used_meanrev = []
    tp_frac_used_trend = []
    trend_tp_strengths = []
    regime_counts = {"MEANREV": 0, "TREND": 0}
    regime_trade_counts = {"MEANREV": 0, "TREND": 0}
    regime_realized = {"MEANREV": 0.0, "TREND": 0.0}
    regime_modes_seen = []
    tox_scores = []
    tox_ticks: Dict[str, int] = {"SOFT": 0, "NEUTRAL": 0, "TOXIC": 0}
    tox_trades: Dict[str, int] = {"SOFT": 0, "NEUTRAL": 0, "TOXIC": 0}
    tox_realized: Dict[str, float] = {"SOFT": 0.0, "NEUTRAL": 0.0, "TOXIC": 0.0}
    last_tox_reg = "NEUTRAL"
    trend_fb_blocked = 0
    trend_fb_allowed = 0

    # ----------------------------
    # Edge bucket attribution (trade-level)
    # ----------------------------
    BUCKET_EDGES = (0.04, 0.05, 0.06, 0.08)
    BUCKET_LABELS = ("0.04–0.05", "0.05–0.06", "0.06–0.08", "0.08+")
    bucket_counts: Dict[str, int] = {k: 0 for k in BUCKET_LABELS}
    bucket_ev: Dict[str, float] = {k: 0.0 for k in BUCKET_LABELS}
    bucket_pnl: Dict[str, float] = {k: 0.0 for k in BUCKET_LABELS}
    bucket_entry_x: Dict[str, float] = {k: 0.0 for k in BUCKET_LABELS}
    bucket_exit_x: Dict[str, float] = {k: 0.0 for k in BUCKET_LABELS}

    def _edge_bucket(a: float) -> str:
        a = float(abs(a))
        if a < BUCKET_EDGES[0]:
            return "<0.04"
        if a < BUCKET_EDGES[1]:
            return BUCKET_LABELS[0]
        if a < BUCKET_EDGES[2]:
            return BUCKET_LABELS[1]
        if a < BUCKET_EDGES[3]:
            return BUCKET_LABELS[2]
        return BUCKET_LABELS[3]

    POST_TTL = 4
    SHOCK_PROB = 0.050
    SHOCK_SCALE = 145.0
    SHOCK_CLIP = 220.0
    shock_burst_remaining = 0
    shock_burst_strength = 0.0
    EV_REALISM_MULT = 0.35
    EV_REALISM_MULT_TREND = 0.42
    EV_REALISM_MULT_MEANREV = 0.18
    MEANREV_EV_MULT = 1.10
    TREND_FB_EDGE_MULT = 1.25
    TREND_FB_RT_MULT = 0.70
    TREND_FB_MIDDELTA_MAX = 0.008
    TREND_FB_TOXIC_BLOCK = True
    TTL_TREND = 10
    TTL_MEANREV = 5
    EARLY_EXIT_EV_FRAC = 0.25
    EARLY_EXIT_MIDDELTA_TOXIC = 0.010
    RUNNER_MIN_EV = 2.0
    RUNNER_MIN_MTM_FRAC = 0.10
    RUNNER_EXTRA_TTL_BASE = 2
    RUNNER_EXTRA_TTL_MAX = 6
    RUNNER_TOXIC_BLOCK = True
    RUNNER_MIDDELTA_MAX = 0.012
    TP_FRAC_MEANREV = 0.70
    TP_FRAC_TREND = 0.90
    TP_TREND_STRENGTH_BONUS = 0.35   # strongest TREND tapes can require up to ~1.10x EV
    TP_TREND_MAX = 1.25
    TREND_REENTRY_MAX_PER_NIGHT = 2
    TREND_REENTRY_EDGE_MIN = 0.055
    TREND_REENTRY_MIDDELTA_MAX = 0.012
    TREND_REENTRY_TOXIC_BLOCK = True
    TREND_REENTRY_STAKE_MULT = 0.75
    TREND_REENTRY_COOLDOWN = 1
    runner_extensions = 0
    runner_realized = 0.0
    runner_ttl_bonus_used = []
    trend_reentries = 0
    trend_reentry_realized = 0.0
    trend_reentry_attempts = 0
    trend_reentry_last_t = -999

    for t in range(T):
        # regime proxy
        regime_thin = (mkt.spread >= 0.06) or (mkt.vol <= 20)

        # Agents observe signals
        p_true_noisy_for_sharp = np.clip(p_true[t] + rng.normal(0, 0.01), 0.01, 0.99)
        p_obs_for_recs = np.clip(mkt.mid + rng.normal(0, 0.05), 0.01, 0.99)

        # Net flow from exogenous agents
        flow = 0.0
        flow += agent_recs(rng, p_obs_for_recs, mkt.mid)
        flow += agent_sharp(rng, p_true_noisy_for_sharp, mkt.mid)
        flow += agent_arb(rng, p_book[t], mkt.mid)

        # Occasional liquidity shock / sweep flow
        shock_flow = 0.0

        # Continue existing burst
        if shock_burst_remaining > 0:
            shock_flow = float(
                np.clip(
                    rng.normal(shock_burst_strength, SHOCK_SCALE * 0.35),
                    -SHOCK_CLIP,
                    SHOCK_CLIP,
                )
            )
            shock_burst_remaining -= 1

        # Possibly start new burst
        elif rng.random() < SHOCK_PROB:

            shock_burst_strength = float(
                np.clip(
                    rng.normal(0.0, SHOCK_SCALE),
                    -SHOCK_CLIP,
                    SHOCK_CLIP,
                )
            )

            # burst lasts 2–4 ticks
            shock_burst_remaining = int(rng.integers(2, 5))

            shock_flow = shock_burst_strength

            shock_events += 1

        flow += shock_flow
        shock_flow_abs.append(float(abs(shock_flow)))

        # Adverse flow clustering: after our fills, nearby sharp/copycat flow can appear
        cluster_flow_now = 0.0
        if adverse_cluster_ttl > 0 and adverse_cluster_strength > 1e-8:
            cluster_flow_now = adverse_cluster_flow_fn(
                mid=float(mkt.mid),
                p_exec=float(p_book[t]),
                strength=float(adverse_cluster_strength),
            )
            flow += cluster_flow_now
            adverse_cluster_flow.append(float(cluster_flow_now))
            adverse_cluster_ticks += 1

            adverse_cluster_ttl -= 1
            adverse_cluster_strength *= 0.70
        else:
            adverse_cluster_flow.append(0.0)

        # Update market from exogenous flow
        market_step(
            mkt,
            net_order_yes_shares=flow,
            mm_kappa=0.03,
            impact=0.0006,
            inv_penalty=0.00008,
            spread_floor=0.02,
            spread_slope=0.00025,
        )

        # Haircut the fair value toward the market to reflect execution reality (dynamic)
        if t == 0:
            mid_delta_now = 0.0
        else:
            mid_delta_now = float(abs(mkt.mid - prev_mid))
        book_mid_gap = float(abs(p_book[t] - mkt.mid))
        tox_score = toxicity_score(
            mid_delta=mid_delta_now,
            spread=float(mkt.spread),
            vol=float(mkt.vol),
            book_mid_gap=book_mid_gap,
        )
        tox_reg = toxicity_regime(tox_score)
        tox_scores.append(float(tox_score))
        if tox_reg in tox_ticks:
            tox_ticks[tox_reg] += 1
        last_tox_reg = tox_reg
        regime_mode = market_regime(
            mid_delta=float(mid_delta_now),
            tox_reg=str(tox_reg),
            run_signal=float(run_z[t]),
            pace_signal=float(pace_z[t]),
        )
        trend_strength = trend_strength_score(
            mid_delta=float(mid_delta_now),
            tox_reg=str(tox_reg),
            run_signal=float(run_z[t]),
            pace_signal=float(pace_z[t]),
            spread=float(mkt.spread),
        )
        trend_strengths.append(float(trend_strength))
        regime_counts[regime_mode] += 1
        regime_modes_seen.append(regime_mode)

        pf_lat_mult = pf_latency_multiplier(
            mid_delta=float(mid_delta_now),
            tox_reg=str(tox_reg),
            spread=float(mkt.spread),
        )
        if regime_mode == "TREND":
            pf_lat_mult *= 0.85
        else:
            pf_lat_mult *= 1.05
        pf_lat_mult = float(np.clip(pf_lat_mult, 0.30, 1.00))

        # Slightly higher process volatility in fast tapes so particles can spread,
        # while observation weights are simultaneously reduced.
        pf_proc_vol = 0.02
        if mid_delta_now >= 0.012:
            pf_proc_vol = 0.030
        elif mid_delta_now >= 0.008:
            pf_proc_vol = 0.026
        elif mid_delta_now >= 0.005:
            pf_proc_vol = 0.023

        if tox_reg == "TOXIC":
            pf_proc_vol *= 1.10
        if regime_mode == "TREND":
            pf_proc_vol *= 1.10
        else:
            pf_proc_vol *= 0.95

        pf_latency_mults.append(float(pf_lat_mult))
        pf_process_vols.append(float(pf_proc_vol))

        # Our observation bundle (poly + book)
        # When short-term runs are extreme, treat Poly as noisier (overreaction/microstructure)
        run_shock = float(abs(run_z[t]))
        poly_noise = 0.07 * (1.0 + 0.35 * np.tanh(run_shock / 2.0))

        obs = {
            "poly": {
                "p": float(mkt.mid),
                "obs_noise": float(poly_noise),
                "weight": float(
                    np.clip(
                        pf_lat_mult * (mkt.vol / 50.0) * (0.03 / max(mkt.spread, 1e-6)),
                        0.10,
                        3.00,
                    )
                ),
            },
            "book": {
                "p": float(p_book[t]),
                "obs_noise": 0.05,
                "weight": float(np.clip(1.0 * pf_lat_mult, 0.10, 1.50)),
            },
        }

        pf.update(p_base=float(p_base[t]), observations=obs, process_vol=float(pf_proc_vol))
        p_fair = pf.estimate(float(p_base[t]))
        lo, hi = pf.credible_interval(float(p_base[t]))
        ci_w = hi - lo

        exec_a = exec_alpha_dynamic(regime_thin=regime_thin, mid_delta=mid_delta_now, base=EXEC_ALPHA_BASE, thin=EXEC_ALPHA_THIN)
        if tox_reg == "TOXIC":
            exec_a *= 0.85
        elif tox_reg == "SOFT":
            exec_a *= 1.05
        exec_a = float(np.clip(exec_a, 0.25, 0.95))
        p_exec = clip01(float(mkt.mid) + exec_a * float(p_fair - mkt.mid))
        edge_raw_abs_list.append(float(abs(p_fair - mkt.mid)))
        edge_exec_abs_list.append(float(abs(p_exec - mkt.mid)))

        # Close positions: take-profit, signal flip, or TTL
        p_now_mid = float(mkt.mid)

        for pos in list(positions):
            # mark-to-mid PnL (rough unrealized estimate)
            mtm = pos_mtm_pnl(pos, p_now_mid)
            pos_regime = getattr(pos, "regime_mode", "MEANREV")
            if pos_regime == "TREND":
                local_trend_strength = trend_strength_score(
                    mid_delta=float(mid_delta_now),
                    tox_reg=str(tox_reg),
                    run_signal=float(run_z[t]),
                    pace_signal=float(pace_z[t]),
                    spread=float(mkt.spread),
                )
                trend_tp_strengths.append(float(local_trend_strength))
                tp_frac = TP_FRAC_TREND + TP_TREND_STRENGTH_BONUS * float(local_trend_strength)
                tp_frac = float(np.clip(tp_frac, TP_FRAC_TREND, TP_TREND_MAX))
                tp_frac_used_trend.append(float(tp_frac))
            else:
                tp_frac = TP_FRAC_MEANREV
                tp_frac_used_meanrev.append(float(tp_frac))
            base_ttl = ttl_by_regime(
                regime_mode=str(pos_regime),
                tox_reg=str(tox_reg),
                regime_thin=bool(regime_thin),
                ttl_trend=TTL_TREND,
                ttl_meanrev=TTL_MEANREV,
            )

            # Trend runner extension: if a strong TREND trade is already working,
            # let it run a bit longer unless tape conditions are hostile.
            if (
                (not pos.is_quote)
                and pos_regime == "TREND"
                and (not getattr(pos, "is_runner", False))
                and float(pos.ev) >= RUNNER_MIN_EV
                and mtm >= (RUNNER_MIN_MTM_FRAC * float(pos.ev))
                and (not (RUNNER_TOXIC_BLOCK and tox_reg == "TOXIC"))
                and float(mid_delta_now) <= RUNNER_MIDDELTA_MAX
            ):
                local_runner_strength = trend_strength_score(
                    mid_delta=float(mid_delta_now),
                    tox_reg=str(tox_reg),
                    run_signal=float(run_z[t]),
                    pace_signal=float(pace_z[t]),
                    spread=float(mkt.spread),
                )
                ttl_bonus_now = runner_ttl_bonus(
                    trend_strength=float(local_runner_strength),
                    base_ttl=RUNNER_EXTRA_TTL_BASE,
                    max_ttl=RUNNER_EXTRA_TTL_MAX,
                )
                pos.ttl_bonus = int(getattr(pos, "ttl_bonus", 0)) + int(ttl_bonus_now)
                pos.is_runner = True
                runner_extensions += 1
                runner_ttl_bonus_used.append(float(ttl_bonus_now))

            pos_ttl = int(base_ttl + int(getattr(pos, "ttl_bonus", 0)))

            if pos_regime == "TREND":
                ttl_used_trend.append(float(pos_ttl))
            else:
                ttl_used_meanrev.append(float(pos_ttl))

            # Early exit if trade quality has deteriorated sharply relative to its expected EV.
            early_exit = False
            if pos.ev > 0:
                if mtm < (-EARLY_EXIT_EV_FRAC * pos.ev):
                    early_exit = True
                if tox_reg == "TOXIC" and mid_delta_now >= EARLY_EXIT_MIDDELTA_TOXIC:
                    if mtm <= 0:
                        early_exit = True

            # 1) TAKE PROFIT: if we've captured a good chunk of expected EV
            if mtm > tp_frac * pos.ev:
                edge_proxy = float(abs(p_now_mid - pos.fill_yes))
                p_now_fill_yes = exit_fill_yes(
                    side=pos.side,
                    mid=p_now_mid,
                    spread=mkt.spread,
                    edge_proxy=edge_proxy,
                    regime_thin=regime_thin,
                    slippage_bps=slippage_bps,
                    stake=float(pos.stake),
                    vol=float(mkt.vol),
                    exit_extra_mult=1.0,
                )
                exit_extras.append(float(abs(p_now_mid - p_now_fill_yes)))
                if getattr(pos, "edge_bucket", None) in bucket_exit_x:
                    bucket_exit_x[pos.edge_bucket] += float(abs(p_now_mid - p_now_fill_yes))
                exit_tp += 1
                pnl = pos_mtm_pnl(pos, p_now_fill_yes)
                if getattr(pos, "is_runner", False):
                    runner_realized += float(pnl)
                if getattr(pos, "is_reentry", False):
                    trend_reentry_realized += float(pnl)
                realized_tp += float(pnl)
                if pos.is_quote:
                    quote_realized += float(pnl)
                if tox_reg in tox_realized:
                    tox_realized[tox_reg] += float(pnl)
                if getattr(pos, "edge_bucket", None) in bucket_pnl:
                    bucket_pnl[pos.edge_bucket] += float(pnl)
                bankroll += pnl
                realized += pnl
                regime_realized[regime_mode] += float(pnl)

                # TREND continuation re-entry:
                # if the same TREND signal still persists after a profitable TP,
                # allow a smaller follow-on position.
                same_dir_sig = None
                edge_now_abs = float(abs(p_exec - mkt.mid))

                if pos.side == "BUY" and p_exec > mkt.mid:
                    same_dir_sig = "BUY"
                elif pos.side == "SELL" and p_exec < mkt.mid:
                    same_dir_sig = "SELL"

                if (
                    same_dir_sig is not None
                    and pnl > 0.0
                ):
                    trend_reentry_attempts += 1

                    can_reenter = allow_trend_reentry(
                        regime_mode=str(regime_mode),
                        tox_reg=str(tox_reg),
                        edge_abs=float(edge_now_abs),
                        mid_delta=float(mid_delta_now),
                        last_reentry_t=int(trend_reentry_last_t),
                        t=int(t),
                        max_per_night_used=int(trend_reentries),
                        max_per_night=int(TREND_REENTRY_MAX_PER_NIGHT),
                        edge_min=float(TREND_REENTRY_EDGE_MIN),
                        middelta_max=float(TREND_REENTRY_MIDDELTA_MAX),
                        toxic_block=bool(TREND_REENTRY_TOXIC_BLOCK),
                        cooldown=int(TREND_REENTRY_COOLDOWN),
                    )

                    if can_reenter:
                        re_stake = stake_ladder(edge_net=edge_now_abs, regime_thin=regime_thin, min_stake=20.0)

                        re_dir_scale = directional_edge_scale(
                            edge_net=float(edge_now_abs),
                            regime_mode="TREND",
                            trend_strength=float(trend_strength),
                        )
                        re_stake *= float(re_dir_scale)
                        re_stake *= float(TREND_REENTRY_STAKE_MULT)
                        re_stake = float(np.clip(re_stake, 0.0, 45.0))

                        exposure_re = (
                            sum(p.stake for p in positions)
                            + sum(po.stake for po in pending_orders)
                            + sum(qo.stake for qo in quote_orders)
                        )
                        cap_re = bankroll * bad_night_pct
                        rem_re = max(0.0, cap_re - exposure_re)

                        if re_stake > 0.0 and re_stake <= rem_re + 1e-9 and len(positions) < max_open:
                            p_re_yes = post_limit_yes_price(
                                side=same_dir_sig,
                                mid=float(mkt.mid),
                                spread=float(mkt.spread),
                                tox_reg=str(tox_reg),
                            )

                            # Conservative EV check for the continuation re-entry
                            ev_re, _, _ = trade_ev_reversion_exit_from_fill(
                                side=same_dir_sig,
                                stake=float(re_stake),
                                p_fill_yes=float(p_re_yes),
                                mid=float(mkt.mid),
                                p_exec=float(p_exec),
                                spread=float(mkt.spread),
                                slippage_bps=float(slippage_bps),
                                regime_thin=bool(regime_thin),
                                tox_reg=str(tox_reg),
                                mid_delta=float(mid_delta_now),
                                vol=float(mkt.vol),
                                regime_mode=str(regime_mode),
                                exit_extra_mult=0.9,
                            )

                            ev_re *= EV_REALISM_MULT_TREND

                            if min_ev_per_dollar is not None:
                                ev_req_re = float(re_stake) * float(min_ev_per_dollar * 0.80)
                            else:
                                ev_req_re = float(min_ev)

                            if ev_re >= ev_req_re:
                                bkt_re = _edge_bucket(edge_now_abs)
                                pending_orders.append(
                                    PendingOrder(
                                        t_post=t,
                                        side=same_dir_sig,
                                        stake=float(re_stake),
                                        limit_yes=float(p_re_yes),
                                        ttl=POST_TTL,
                                        edge_bucket=bkt_re,
                                        tox_reg=str(tox_reg),
                                        edge_net=float(edge_now_abs),
                                        ev_at_post=float(ev_re),
                                        queue_pos=initial_queue_position(
                                            tox_reg=str(tox_reg),
                                            spread=float(mkt.spread),
                                            vol=float(mkt.vol),
                                            rng=rng,
                                        ),
                                        touches=0,
                                        is_reentry=True,
                                    )
                                )
                                queue_pos_posted.append(float(pending_orders[-1].queue_pos))
                                posted_orders += 1
                                trend_reentries += 1
                                trend_reentry_last_t = int(t)
                positions.remove(pos)
                continue

            # 2) SIGNAL FLIP: if execution-haircutted fair value crosses the market against us
            if (pos.side == "BUY" and p_exec < mkt.mid) or (pos.side == "SELL" and p_exec > mkt.mid):
                edge_proxy = float(abs(p_now_mid - pos.fill_yes))
                p_now_fill_yes = exit_fill_yes(
                    side=pos.side,
                    mid=p_now_mid,
                    spread=mkt.spread,
                    edge_proxy=edge_proxy,
                    regime_thin=regime_thin,
                    slippage_bps=slippage_bps,
                    stake=float(pos.stake),
                    vol=float(mkt.vol),
                    exit_extra_mult=1.5,
                )
                exit_extras.append(float(abs(p_now_mid - p_now_fill_yes)))
                if getattr(pos, "edge_bucket", None) in bucket_exit_x:
                    bucket_exit_x[pos.edge_bucket] += float(abs(p_now_mid - p_now_fill_yes))
                exit_flip += 1
                pnl = pos_mtm_pnl(pos, p_now_fill_yes)
                if getattr(pos, "is_runner", False):
                    runner_realized += float(pnl)
                if getattr(pos, "is_reentry", False):
                    trend_reentry_realized += float(pnl)
                realized_flip += float(pnl)
                if pos.is_quote:
                    quote_realized += float(pnl)
                if tox_reg in tox_realized:
                    tox_realized[tox_reg] += float(pnl)
                if getattr(pos, "edge_bucket", None) in bucket_pnl:
                    bucket_pnl[pos.edge_bucket] += float(pnl)
                bankroll += pnl
                realized += pnl
                regime_realized[regime_mode] += float(pnl)
                positions.remove(pos)
                continue

            # 3) EARLY EXIT: deteriorating trade quality / toxic tape
            if early_exit:
                edge_proxy = float(abs(p_now_mid - pos.fill_yes))
                p_now_fill_yes = exit_fill_yes(
                    side=pos.side,
                    mid=p_now_mid,
                    spread=mkt.spread,
                    edge_proxy=edge_proxy,
                    regime_thin=regime_thin,
                    slippage_bps=slippage_bps,
                    stake=float(pos.stake),
                    vol=float(mkt.vol),
                    exit_extra_mult=1.6,
                )
                exit_extras.append(float(abs(p_now_mid - p_now_fill_yes)))
                if getattr(pos, "edge_bucket", None) in bucket_exit_x:
                    bucket_exit_x[pos.edge_bucket] += float(abs(p_now_mid - p_now_fill_yes))
                exit_early += 1
                pnl = pos_mtm_pnl(pos, p_now_fill_yes)
                if getattr(pos, "is_runner", False):
                    runner_realized += float(pnl)
                if getattr(pos, "is_reentry", False):
                    trend_reentry_realized += float(pnl)
                realized_early += float(pnl)
                if pos.is_quote:
                    quote_realized += float(pnl)
                if tox_reg in tox_realized:
                    tox_realized[tox_reg] += float(pnl)
                if getattr(pos, "edge_bucket", None) in bucket_pnl:
                    bucket_pnl[pos.edge_bucket] += float(pnl)
                bankroll += pnl
                realized += pnl
                regime_realized[regime_mode] += float(pnl)
                positions.remove(pos)
                continue

            # 4) TTL exit (fallback)
            if (t - pos.t_open) >= pos_ttl:
                edge_proxy = float(abs(p_now_mid - pos.fill_yes))
                p_now_fill_yes = exit_fill_yes(
                    side=pos.side,
                    mid=p_now_mid,
                    spread=mkt.spread,
                    edge_proxy=edge_proxy,
                    regime_thin=regime_thin,
                    slippage_bps=slippage_bps,
                    stake=float(pos.stake),
                    vol=float(mkt.vol),
                    exit_extra_mult=2.0,
                )
                exit_extras.append(float(abs(p_now_mid - p_now_fill_yes)))
                if getattr(pos, "edge_bucket", None) in bucket_exit_x:
                    bucket_exit_x[pos.edge_bucket] += float(abs(p_now_mid - p_now_fill_yes))
                exit_ttl += 1
                pnl = pos_mtm_pnl(pos, p_now_fill_yes)
                if getattr(pos, "is_runner", False):
                    runner_realized += float(pnl)
                if getattr(pos, "is_reentry", False):
                    trend_reentry_realized += float(pnl)
                realized_ttl += float(pnl)
                if pos.is_quote:
                    quote_realized += float(pnl)
                if tox_reg in tox_realized:
                    tox_realized[tox_reg] += float(pnl)
                if getattr(pos, "edge_bucket", None) in bucket_pnl:
                    bucket_pnl[pos.edge_bucket] += float(pnl)
                bankroll += pnl
                realized += pnl
                regime_realized[regime_mode] += float(pnl)
                positions.remove(pos)

        # Process posted pending orders before considering fresh entries this tick
        for po in list(pending_orders):
            age = t - po.t_post
            filled, new_qpos, new_touches = passive_fill_hit(
                side=po.side,
                limit_yes=float(po.limit_yes),
                mid=float(mkt.mid),
                prev_mid=float(prev_mid),
                spread=float(mkt.spread),
                tox_reg=str(tox_reg),
                mid_delta=float(mid_delta_now),
                vol=float(mkt.vol),
                queue_pos=float(po.queue_pos),
                touches=int(po.touches),
                rng=rng,
            )
            po.queue_pos = float(new_qpos)
            po.touches = int(new_touches)
            if filled:
                if po.edge_bucket in bucket_counts:
                    bucket_counts[po.edge_bucket] += 1
                    bucket_ev[po.edge_bucket] += float(po.ev_at_post)
                    bucket_entry_x[po.edge_bucket] += float(abs(po.limit_yes - mkt.mid))
                positions.append(
                    Position(
                        t_open=t,
                        side=po.side,
                        stake=float(po.stake),
                        fill_yes=float(po.limit_yes),
                        spread=float(mkt.spread),
                        ev=float(po.ev_at_post),
                        edge_bucket=po.edge_bucket,
                        is_quote=False,
                        regime_mode=str(regime_mode),
                        ttl_bonus=0,
                        is_runner=False,
                        is_reentry=bool(getattr(po, "is_reentry", False)),
                    )
                )
                posted_fills += 1
                passive_fills += 1
                n_trades += 1
                if regime_mode == "TREND":
                    trend_strength_trade.append(float(trend_strength))
                regime_trade_counts[regime_mode] += 1
                expected += float(po.ev_at_post)
                if po.tox_reg in tox_trades:
                    tox_trades[po.tox_reg] += 1
                maker_entry_dists.append(float(abs(po.limit_yes - float(mkt.mid))))
                passive_entry_dists.append(float(abs(po.limit_yes - float(mkt.mid))))
                queue_pos_filled.append(float(po.queue_pos))
                queue_touches_filled.append(float(po.touches))
                adverse_cluster_strength = max(adverse_cluster_strength, 0.9)
                adverse_cluster_ttl = max(adverse_cluster_ttl, 3)
                adverse_cluster_activations += 1
                pending_orders.remove(po)
                continue

            if age >= po.ttl:
                edge_now = float(p_exec - mkt.mid)
                edge_net_now = float(edge_now - (mkt.spread / 2.0))
                sig_fb_raw = decide_trade(edge_net=edge_net_now, ci_width=ci_w, spread=mkt.spread, regime_thin=regime_thin)

                HARD_EDGE_FB = 0.06
                SOFT_EDGE_FB = 0.045
                TIGHT_SPREAD_MAX_FB = 0.04
                STABLE_MID_MAX_D_SOFT_FB = 0.006
                STABLE_MID_MAX_D_FB = 0.010
                if tox_reg == "TOXIC":
                    HARD_EDGE_FB += 0.02
                    SOFT_EDGE_FB += 0.02
                elif tox_reg == "SOFT":
                    HARD_EDGE_FB -= 0.01
                    SOFT_EDGE_FB -= 0.01
                allow_soft_fb = (not regime_thin) and (mkt.spread <= TIGHT_SPREAD_MAX_FB) and (mid_delta_now <= STABLE_MID_MAX_D_SOFT_FB)
                edge_gate_fb = SOFT_EDGE_FB if allow_soft_fb else HARD_EDGE_FB
                if sig_fb_raw in ("BUY", "SELL") and abs(edge_net_now) < edge_gate_fb:
                    sig_fb = "HOLD(edge_gate)"
                else:
                    sig_fb = sig_fb_raw
                still_favorable = (sig_fb == po.side)

                filled_fallback = False
                if still_favorable:
                    p_fb_fill_yes = entry_fill_yes(
                        side=po.side,
                        mid=float(mkt.mid),
                        spread=float(mkt.spread),
                        edge_net=float(edge_net_now),
                        regime_thin=bool(regime_thin),
                        slippage_bps=float(slippage_bps),
                        stake=float(po.stake),
                        vol=float(mkt.vol),
                    )

                    RT_GATE_MULT = 0.40
                    RT_GATE_BUF = 0.001
                    req_edge_fb = roundtrip_cost_gate(
                        side=po.side,
                        mid=float(mkt.mid),
                        spread=float(mkt.spread),
                        slippage_bps=slippage_bps,
                        p_fill_yes=float(p_fb_fill_yes),
                        stake=float(po.stake),
                        edge_net=float(edge_net_now),
                        regime_thin=bool(regime_thin),
                        mid_delta=float(mid_delta_now),
                        buffer=float(RT_GATE_BUF),
                    )
                    raw_edge_fb = float(abs(p_exec - mkt.mid))
                    fb_allowed = True
                    rt_gate_mult_fb = RT_GATE_MULT

                    if regime_mode == "TREND":
                        # Require stronger edge and more roundtrip coverage in TREND fallback.
                        if raw_edge_fb < (TREND_FB_EDGE_MULT * edge_gate_fb):
                            fb_allowed = False
                        if mid_delta_now > TREND_FB_MIDDELTA_MAX:
                            fb_allowed = False
                        if TREND_FB_TOXIC_BLOCK and tox_reg == "TOXIC":
                            fb_allowed = False
                        rt_gate_mult_fb = max(rt_gate_mult_fb, TREND_FB_RT_MULT)

                        if fb_allowed:
                            trend_fb_allowed += 1
                        else:
                            trend_fb_blocked += 1

                    if fb_allowed and raw_edge_fb >= (rt_gate_mult_fb * req_edge_fb):
                        exit_mult_gate_fb = 1.3
                        if regime_thin:
                            exit_mult_gate_fb = 1.8
                        if mid_delta_now >= STABLE_MID_MAX_D_FB:
                            exit_mult_gate_fb = max(exit_mult_gate_fb, 2.0)

                        ev_old_fb = trade_ev_from_fill(
                            side=po.side,
                            stake=float(po.stake),
                            p_fill_yes=float(p_fb_fill_yes),
                            p_fair=float(p_exec),
                            spread=float(mkt.spread),
                            slippage_bps=slippage_bps,
                            edge_net=float(edge_net_now),
                            regime_thin=bool(regime_thin),
                            exit_extra_mult=float(exit_mult_gate_fb),
                        )
                        ev_fb, p_exit_mid_fb, lam_fb = trade_ev_reversion_exit_from_fill(
                            side=po.side,
                            stake=float(po.stake),
                            p_fill_yes=float(p_fb_fill_yes),
                            mid=float(mkt.mid),
                            p_exec=float(p_exec),
                            spread=float(mkt.spread),
                            slippage_bps=slippage_bps,
                            regime_thin=bool(regime_thin),
                            tox_reg=str(tox_reg),
                            mid_delta=float(mid_delta_now),
                            vol=float(mkt.vol),
                            regime_mode=str(regime_mode),
                            exit_extra_mult=float(exit_mult_gate_fb),
                        )
                        exp_exit_modes[regime_mode] += 1
                        if regime_mode == "TREND":
                            trend_kappas.append(
                                trend_continuation_kappa(
                                    mid_delta=float(mid_delta_now),
                                    tox_reg=str(tox_reg),
                                    regime_thin=bool(regime_thin),
                                )
                            )
                        if regime_mode == "TREND":
                            ev_fb *= EV_REALISM_MULT_TREND
                        else:
                            ev_fb *= EV_REALISM_MULT_MEANREV
                        regime_ev_haircuts[regime_mode].append(
                            EV_REALISM_MULT_TREND if regime_mode == "TREND" else EV_REALISM_MULT_MEANREV
                        )
                        ev_old_assump.append(float(ev_old_fb))
                        ev_reversion_assump.append(float(ev_fb))
                        rev_lam_list.append(float(lam_fb))
                        exp_exit_move_list.append(float(abs(p_exit_mid_fb - mkt.mid)))

                        if min_ev_per_dollar is not None:
                            ev_req_fb = float(po.stake) * float(min_ev_per_dollar)
                        else:
                            ev_req_fb = float(min_ev)
                        ev_mult_fb = 1.00
                        if tox_reg == "TOXIC":
                            ev_mult_fb = 1.20
                        elif tox_reg == "SOFT":
                            ev_mult_fb = 0.90
                        ev_req_fb *= float(ev_mult_fb)
                        if regime_mode == "MEANREV":
                            ev_req_fb *= MEANREV_EV_MULT

                        exposure_fb = (
                            sum(p.stake for p in positions)
                            + sum(p2.stake for p2 in pending_orders if p2 is not po)
                            + sum(qo.stake for qo in quote_orders)
                        )
                        cap_fb = bankroll * bad_night_pct
                        rem_fb = max(0.0, cap_fb - exposure_fb)

                        if (ev_fb >= ev_req_fb) and (len(positions) < max_open) and (float(po.stake) <= rem_fb + 1e-9):
                            if po.edge_bucket in bucket_counts:
                                bucket_counts[po.edge_bucket] += 1
                                bucket_ev[po.edge_bucket] += float(ev_fb)
                                bucket_entry_x[po.edge_bucket] += float(abs(p_fb_fill_yes - mkt.mid))
                            positions.append(
                                Position(
                                    t_open=t,
                                    side=po.side,
                                    stake=float(po.stake),
                                    fill_yes=float(p_fb_fill_yes),
                                    spread=float(mkt.spread),
                                    ev=float(ev_fb),
                                    edge_bucket=po.edge_bucket,
                                    is_quote=False,
                                    regime_mode=str(regime_mode),
                                    ttl_bonus=0,
                                    is_runner=False,
                                    is_reentry=False,
                                )
                            )
                            taker_fallback_fills += 1
                            taker_fills += 1
                            n_trades += 1
                            if regime_mode == "TREND":
                                trend_strength_trade.append(float(trend_strength))
                            regime_trade_counts[regime_mode] += 1
                            expected += float(ev_fb)
                            if allow_soft_fb:
                                soft_gate_trades += 1
                            if tox_reg in tox_trades:
                                tox_trades[tox_reg] += 1
                            fallback_entry_dists.append(float(abs(p_fb_fill_yes - float(mkt.mid))))
                            taker_entry_dists.append(float(abs(p_fb_fill_yes - float(mkt.mid))))
                            adverse_cluster_strength = max(adverse_cluster_strength, 1.2)
                            adverse_cluster_ttl = max(adverse_cluster_ttl, 3)
                            adverse_cluster_activations += 1
                            filled_fallback = True

                if filled_fallback:
                    pending_orders.remove(po)
                else:
                    posted_cancels += 1
                    pending_orders.remove(po)

        pending_live_sum += float(len(pending_orders))

        # Process inventory quote orders (passive-only, no taker fallback)
        for qo in list(quote_orders):
            age = t - qo.t_post
            q_filled, q_new_pos, q_new_touches = passive_fill_hit(
                side=qo.side,
                limit_yes=float(qo.limit_yes),
                mid=float(mkt.mid),
                prev_mid=float(prev_mid),
                spread=float(mkt.spread),
                tox_reg=str(tox_reg),
                mid_delta=float(mid_delta_now),
                vol=float(mkt.vol),
                queue_pos=float(qo.queue_pos),
                touches=int(qo.touches),
                rng=rng,
            )
            qo.queue_pos = float(q_new_pos)
            qo.touches = int(q_new_touches)

            if q_filled:
                q_bkt = _edge_bucket(qo.edge_net)
                if q_bkt in bucket_counts:
                    bucket_counts[q_bkt] += 1
                    bucket_ev[q_bkt] += 0.50
                    bucket_entry_x[q_bkt] += float(abs(qo.limit_yes - mkt.mid))
                positions.append(
                    Position(
                        t_open=t,
                        side=qo.side,
                        stake=float(qo.stake),
                        fill_yes=float(qo.limit_yes),
                        spread=float(mkt.spread),
                        ev=0.50,
                        edge_bucket=q_bkt,
                        is_quote=True,
                        regime_mode=str(regime_mode),
                        ttl_bonus=0,
                        is_runner=False,
                        is_reentry=False,
                    )
                )
                quote_fills += 1
                n_trades += 1
                regime_trade_counts[regime_mode] += 1
                expected += 0.50
                if qo.tox_reg in tox_trades:
                    tox_trades[qo.tox_reg] += 1
                quote_fill_dists.append(float(abs(qo.limit_yes - float(mkt.mid))))
                passive_fills += 1
                passive_entry_dists.append(float(abs(qo.limit_yes - float(mkt.mid))))
                entry_extras.append(float(abs(qo.limit_yes - float(mkt.mid))))
                entry_dist_abs_list.append(float(abs(qo.limit_yes - float(mkt.mid))))
                edge_net_abs_list.append(float(abs(qo.edge_net)))
                if qo.tox_reg == "SOFT":
                    quote_trades_soft += 1
                elif qo.tox_reg == "NEUTRAL":
                    quote_trades_neutral += 1
                adverse_cluster_strength = max(adverse_cluster_strength, 0.5)
                adverse_cluster_ttl = max(adverse_cluster_ttl, 2)
                adverse_cluster_activations += 1
                quote_orders.remove(qo)
                continue

            if age >= qo.ttl:
                quote_cancels += 1
                quote_orders.remove(qo)

        quote_live_sum += float(len(quote_orders))

        # Compute signal (use execution-haircutted fair value)
        edge = float(p_exec - mkt.mid)
        edge_net = float(edge - (mkt.spread / 2.0))
        sig_raw = decide_trade(edge_net=edge_net, ci_width=ci_w, spread=mkt.spread, regime_thin=regime_thin)

        # Persistence filter: require the same directional signal 2 ticks in a row,
        # but allow a fast lane for strong edges ONLY when the market is stable.
        # Rationale: big edges often coincide with toxic flow; require low mid volatility.
        STRONG_EDGE_FASTLANE = 0.06
        STABLE_MID_MAX_D = 0.010

        # track mid change for stability (t==0 treated as stable)
        if t == 0:
            mid_delta = 0.0
        else:
            mid_delta = float(abs(mkt.mid - prev_mid))

        allow_fast_lane = (abs(p_exec - mkt.mid) >= STRONG_EDGE_FASTLANE) and (mid_delta <= STABLE_MID_MAX_D)

        # Persistence filter: only require confirmation for weaker edges.
        # Stronger edges can act immediately (still respecting the fast-lane guard).
        PERSIST_MIN_EDGE = 0.05

        if sig_raw in ("BUY", "SELL") and sig_raw != prev_sig_raw:
            if (abs(p_exec - mkt.mid) < PERSIST_MIN_EDGE) and (not allow_fast_lane):
                sig = "HOLD(persist)"
            else:
                sig = sig_raw
        else:
            sig = sig_raw

        # ----------------------------
        # Hard edge gate (execution-aware)
        # ----------------------------
        HARD_EDGE = 0.08
        SOFT_EDGE = 0.06
        TIGHT_SPREAD_MAX = 0.04
        STABLE_MID_MAX_D_SOFT = 0.006
        if tox_reg == "TOXIC":
            HARD_EDGE += 0.02
            SOFT_EDGE += 0.02
        elif tox_reg == "SOFT":
            HARD_EDGE -= 0.01
            SOFT_EDGE -= 0.01

        allow_soft = (not regime_thin) and (mkt.spread <= TIGHT_SPREAD_MAX) and (mid_delta <= STABLE_MID_MAX_D_SOFT)
        edge_gate = SOFT_EDGE if allow_soft else HARD_EDGE

        if sig_raw in ("BUY", "SELL"):
            if allow_soft:
                soft_gate_used_ticks += 1
            if allow_soft and (abs(p_exec - mkt.mid) >= SOFT_EDGE) and (abs(p_exec - mkt.mid) < HARD_EDGE):
                soft_gate_skips += 1

        if sig_raw in ("BUY", "SELL") and abs(p_exec - mkt.mid) < edge_gate:
            sig = "HOLD(edge_gate)"

        prev_sig_raw = sig_raw

        # ----------------------------
        # Reversion entry filter (relaxed)
        # ----------------------------
        # Avoid entering directly into sharp spikes, but allow slight adverse drift.
        # This is less strict than requiring full reversal of the last tick.
        EXTREME_EDGE_BYPASS = 0.12
        REVERSION_TOL = 0.003  # 0.3 prob points of tolerated adverse movement

        if sig in ("BUY", "SELL"):
            allow_reversion = True
            if sig == "BUY":
                # OK if price is flat/down OR only slightly up against us OR edge is extreme
                if not ((mkt.mid <= prev_mid + REVERSION_TOL) or ((p_exec - mkt.mid) >= EXTREME_EDGE_BYPASS)):
                    allow_reversion = False
            elif sig == "SELL":
                # OK if price is flat/up OR only slightly down against us OR edge is extreme
                if not ((mkt.mid >= prev_mid - REVERSION_TOL) or ((mkt.mid - p_exec) >= EXTREME_EDGE_BYPASS)):
                    allow_reversion = False

            if not allow_reversion:
                sig = "HOLD(revert)"

        # Decide stake
        if sig in ("BUY", "SELL"):
            stake = stake_ladder(edge_net=edge_net, regime_thin=regime_thin, min_stake=20.0)

            # Existing mild trend-strength scaler
            trend_scale = 1.0
            if regime_mode == "TREND":
                trend_scale = 0.90 + 0.35 * float(trend_strength)   # ~0.90x to ~1.25x
                stake *= trend_scale
                trend_scale_used.append(float(trend_scale))
            else:
                trend_scale_used.append(1.0)

            # New edge-weighted directional scaler
            dir_scale = directional_edge_scale(
                edge_net=float(edge_net),
                regime_mode=str(regime_mode),
                trend_strength=float(trend_strength),
            )
            stake *= dir_scale

            dir_edge_scale_used.append(float(dir_scale))
            if regime_mode == "TREND":
                dir_edge_scale_trend.append(float(dir_scale))
            else:
                dir_edge_scale_meanrev.append(float(dir_scale))

            # Keep practical bounds.
            stake = float(np.clip(stake, 0.0, 65.0))
        else:
            stake = 0.0

        # Risk/exposure
        exposure = (
            sum(p.stake for p in positions)
            + sum(po.stake for po in pending_orders)
            + sum(qo.stake for qo in quote_orders)
        )
        cap = bankroll * bad_night_pct
        rem = max(0.0, cap - exposure)

        action = "SKIP"
        why = sig

        if sig in ("BUY", "SELL") and stake > 0:
            # Post passive order first; fills are processed in subsequent ticks.
            p_post_yes = post_limit_yes_price(
                side=sig,
                mid=float(mkt.mid),
                spread=float(mkt.spread),
                tox_reg=str(tox_reg),
            )

            entry_extras.append(float(abs(p_post_yes - float(mkt.mid))))
            entry_dist_abs_list.append(float(abs(p_post_yes - float(mkt.mid))))
            edge_net_abs_list.append(float(abs(p_exec - mkt.mid)))

            # Roundtrip-cost gate: require enough raw edge to survive realistic fills
            RT_GATE_MULT = 0.40   # 0..1; 1.0 = strict full-roundtrip, 0.5 = allow half-roundtrip before skipping
            RT_GATE_BUF = 0.001   # small safety buffer in prob units
            req_edge = roundtrip_cost_gate(
                side=sig,
                mid=float(mkt.mid),
                spread=float(mkt.spread),
                slippage_bps=slippage_bps,
                p_fill_yes=float(p_post_yes),
                stake=float(stake),
                edge_net=float(edge_net),
                regime_thin=bool(regime_thin),
                mid_delta=float(mid_delta),
                buffer=float(RT_GATE_BUF),
            )
            raw_edge = float(abs(p_exec - mkt.mid))
            if raw_edge < (RT_GATE_MULT * req_edge):
                n_skips_ev += 1
                action = "SKIP"
                why = f"rt({raw_edge:.3f}<{(RT_GATE_MULT*req_edge):.3f})"
                continue

            # Stratum-conditional exit friction in EV gating (matches how we tend to exit)
            exit_mult_gate = 1.3
            if regime_thin:
                exit_mult_gate = 1.8
            if mid_delta >= STABLE_MID_MAX_D:
                exit_mult_gate = max(exit_mult_gate, 2.0)

            ev_old = trade_ev_from_fill(
                side=sig,
                stake=stake,
                p_fill_yes=float(p_post_yes),
                p_fair=float(p_exec),
                spread=float(mkt.spread),
                slippage_bps=slippage_bps,
                edge_net=float(edge_net),
                regime_thin=bool(regime_thin),
                exit_extra_mult=float(exit_mult_gate),
            )
            ev, p_exit_mid_exp, lam_exp = trade_ev_reversion_exit_from_fill(
                side=sig,
                stake=stake,
                p_fill_yes=float(p_post_yes),
                mid=float(mkt.mid),
                p_exec=float(p_exec),
                spread=float(mkt.spread),
                slippage_bps=slippage_bps,
                regime_thin=bool(regime_thin),
                tox_reg=str(tox_reg),
                mid_delta=float(mid_delta),
                vol=float(mkt.vol),
                regime_mode=str(regime_mode),
                exit_extra_mult=float(exit_mult_gate),
            )
            exp_exit_modes[regime_mode] += 1
            if regime_mode == "TREND":
                trend_kappas.append(
                    trend_continuation_kappa(
                        mid_delta=float(mid_delta),
                        tox_reg=str(tox_reg),
                        regime_thin=bool(regime_thin),
                    )
                )
            # Regime-aware execution realism haircut.
            if regime_mode == "TREND":
                ev *= EV_REALISM_MULT_TREND
            else:
                ev *= EV_REALISM_MULT_MEANREV
            regime_ev_haircuts[regime_mode].append(
                EV_REALISM_MULT_TREND if regime_mode == "TREND" else EV_REALISM_MULT_MEANREV
            )
            ev_old_assump.append(float(ev_old))
            ev_reversion_assump.append(float(ev))
            rev_lam_list.append(float(lam_exp))
            exp_exit_move_list.append(float(abs(p_exit_mid_exp - mkt.mid)))

            # EV gate: either absolute dollars (legacy) OR EV-per-dollar (preferred)
            if min_ev_per_dollar is not None:
                ev_req = float(stake) * float(min_ev_per_dollar)
                ev_req_label = f"{min_ev_per_dollar:.3f}*stk"
            else:
                ev_req = float(min_ev)
                ev_req_label = f"{min_ev:.2f}"

            ev_mult = 1.00
            if tox_reg == "TOXIC":
                ev_mult = 1.20
            elif tox_reg == "SOFT":
                ev_mult = 0.90
            ev_req *= float(ev_mult)
            # Additional conservatism in MEANREV mode: require somewhat higher EV.
            if regime_mode == "MEANREV":
                ev_req *= MEANREV_EV_MULT

            if ev < ev_req:
                n_skips_ev += 1
                action = "SKIP"
                why = f"ev({ev:+.2f}<{ev_req_label}*{ev_mult:.2f})"
            elif len(positions) >= max_open:
                action = "SKIP"
                why = "open_limit"
            elif stake > rem + 1e-9:
                action = "SKIP"
                why = "risk_cap"
            else:
                # post order first; count trade attribution only when it actually fills
                bkt = _edge_bucket(edge_net)
                pending_orders.append(
                    PendingOrder(
                        t_post=t,
                        side=sig,
                        stake=float(stake),
                        limit_yes=float(p_post_yes),
                        ttl=POST_TTL,
                        edge_bucket=bkt,
                        tox_reg=str(tox_reg),
                        edge_net=float(edge_net),
                        ev_at_post=float(ev),
                        queue_pos=initial_queue_position(
                            tox_reg=str(tox_reg),
                            spread=float(mkt.spread),
                            vol=float(mkt.vol),
                            rng=rng,
                        ),
                        touches=0,
                        is_reentry=False,
                    )
                )
                queue_pos_posted.append(float(pending_orders[-1].queue_pos))
                posted_orders += 1
                action = "POST"
                why = sig

        # Inventory quoting path (additive to directional flow)
        QUOTE_EDGE_MAX = 0.06
        QUOTE_TTL = 3
        MAX_QUOTE_ORDERS = 2
        QUOTE_ALPHA = 0.025
        QUOTE_MIDDELTA_MAX = 0.006   # skip quotes when tape is moving too fast
        QUOTE_VOL_MIN = 25.0         # avoid low-liquidity quote posting

        quote_edge_abs = abs(p_exec - mkt.mid)
        quote_edge_req = (mkt.spread / 2.0) + QUOTE_ALPHA

        if (
            action not in ("POST", "TRADE", "QUOTE")
            and sig_raw in ("BUY", "SELL")
            and tox_reg == "SOFT"
            and (quote_edge_abs >= quote_edge_req)
            and (quote_edge_abs < QUOTE_EDGE_MAX)
            and (mkt.spread <= 0.05)
            and (not regime_thin)
            and (mid_delta <= QUOTE_MIDDELTA_MAX)
            and (mkt.vol >= QUOTE_VOL_MIN)
            and (len(quote_orders) < MAX_QUOTE_ORDERS)
            and (not any(qo.side == sig_raw for qo in quote_orders))
        ):
            q_stake = quote_stake(edge_net=quote_edge_abs, regime_thin=regime_thin, min_stake=4.0)
            if q_stake > 0:
                q_exposure = (
                    sum(p.stake for p in positions)
                    + sum(po.stake for po in pending_orders)
                    + sum(qo.stake for qo in quote_orders)
                )
                q_cap = bankroll * bad_night_pct
                q_rem = max(0.0, q_cap - q_exposure)
                if q_stake <= q_rem + 1e-9:
                    q_limit_yes = post_quote_yes_price(
                        side=sig_raw,
                        mid=float(mkt.mid),
                        spread=float(mkt.spread),
                        tox_reg=str(tox_reg),
                    )
                    quote_orders.append(
                        QuoteOrder(
                            t_post=t,
                            side=sig_raw,
                            stake=float(q_stake),
                            limit_yes=float(q_limit_yes),
                            ttl=QUOTE_TTL,
                            edge_net=float(quote_edge_abs),
                            tox_reg=str(tox_reg),
                            queue_pos=initial_queue_position(
                                tox_reg=str(tox_reg),
                                spread=float(mkt.spread),
                                vol=float(mkt.vol),
                                rng=rng,
                            ),
                            touches=0,
                        )
                    )
                    quote_posts += 1
                    action = "QUOTE"
                    why = sig_raw

        # track drawdown
        peak = max(peak, bankroll)
        dd = (peak - bankroll)
        max_dd = max(max_dd, dd)

        prev_mid = float(mkt.mid)
        prev_spread = float(mkt.spread)

    # force-close at end (realistic nightly flatten)
    final_regime_mode = regime_modes_seen[-1] if len(regime_modes_seen) else "MEANREV"
    p_now_mid = float(mkt.mid)
    for pos in list(positions):
        edge_proxy = float(abs(p_now_mid - pos.fill_yes))
        p_now_fill_yes = exit_fill_yes(
            side=pos.side,
            mid=p_now_mid,
            spread=mkt.spread,
            edge_proxy=edge_proxy,
            regime_thin=(mkt.spread >= 0.06) or (mkt.vol <= 20),
            slippage_bps=slippage_bps,
            stake=float(pos.stake),
            vol=float(mkt.vol),
            exit_extra_mult=2.0,
        )
        exit_extras.append(float(abs(p_now_mid - p_now_fill_yes)))
        if getattr(pos, "edge_bucket", None) in bucket_exit_x:
            bucket_exit_x[pos.edge_bucket] += float(abs(p_now_mid - p_now_fill_yes))
        pnl = pos_mtm_pnl(pos, p_now_fill_yes)
        if getattr(pos, "is_runner", False):
            runner_realized += float(pnl)
        if getattr(pos, "is_reentry", False):
            trend_reentry_realized += float(pnl)
        realized_eod += float(pnl)
        if pos.is_quote:
            quote_realized += float(pnl)
        if last_tox_reg in tox_realized:
            tox_realized[last_tox_reg] += float(pnl)
        if getattr(pos, "edge_bucket", None) in bucket_pnl:
            bucket_pnl[pos.edge_bucket] += float(pnl)
        bankroll += pnl
        realized += pnl
        regime_realized[final_regime_mode] += float(pnl)
        positions.remove(pos)
    if len(pending_orders):
        posted_cancels += int(len(pending_orders))
        pending_orders.clear()
    if len(quote_orders):
        quote_cancels += int(len(quote_orders))
        quote_orders.clear()

    return {
        "ending_bankroll": bankroll,
        "pnl": bankroll - bankroll_start,
        "realized": realized,
        "expected": expected,
        "max_drawdown": max_dd,
        "n_trades": float(n_trades),
        "n_skips_ev": float(n_skips_ev),
        "exit_tp": float(exit_tp),
        "exit_flip": float(exit_flip),
        "exit_ttl": float(exit_ttl),
        "realized_tp": float(realized_tp),
        "realized_flip": float(realized_flip),
        "realized_ttl": float(realized_ttl),
        "realized_eod": float(realized_eod),
        "runner_extensions": float(runner_extensions),
        "runner_realized": float(runner_realized),
        "mean_runner_ttl_bonus": float(np.mean(runner_ttl_bonus_used)) if len(runner_ttl_bonus_used) else 0.0,
        "trend_reentries": float(trend_reentries),
        "trend_reentry_realized": float(trend_reentry_realized),
        "trend_reentry_attempts": float(trend_reentry_attempts),
        "exit_early": float(exit_early),
        "realized_early": float(realized_early),
        "mean_tp_frac_meanrev": float(np.mean(tp_frac_used_meanrev)) if len(tp_frac_used_meanrev) else 0.0,
        "mean_tp_frac_trend": float(np.mean(tp_frac_used_trend)) if len(tp_frac_used_trend) else 0.0,
        "mean_ttl_trend": float(np.mean(ttl_used_trend)) if len(ttl_used_trend) else 0.0,
        "mean_ttl_meanrev": float(np.mean(ttl_used_meanrev)) if len(ttl_used_meanrev) else 0.0,
        "avg_entry_extra": float(np.mean(entry_extras)) if len(entry_extras) else 0.0,
        "avg_exit_extra": float(np.mean(exit_extras)) if len(exit_extras) else 0.0,
        "mean_edge_raw_abs": float(np.mean(edge_raw_abs_list)) if len(edge_raw_abs_list) else 0.0,
        "mean_edge_exec_abs": float(np.mean(edge_exec_abs_list)) if len(edge_exec_abs_list) else 0.0,
        "mean_entry_dist_abs": float(np.mean(entry_dist_abs_list)) if len(entry_dist_abs_list) else 0.0,
        "mean_edge_net_abs": float(np.mean(edge_net_abs_list)) if len(edge_net_abs_list) else 0.0,
        "soft_gate_used_ticks": float(soft_gate_used_ticks),
        "soft_gate_trades": float(soft_gate_trades),
        "soft_gate_skips": float(soft_gate_skips),
        "mean_ev_old_assump": float(np.mean(ev_old_assump)) if len(ev_old_assump) else 0.0,
        "mean_ev_reversion_assump": float(np.mean(ev_reversion_assump)) if len(ev_reversion_assump) else 0.0,
        "mean_ev_haircut_meanrev": float(np.mean(regime_ev_haircuts["MEANREV"])) if len(regime_ev_haircuts["MEANREV"]) else 0.0,
        "mean_ev_haircut_trend": float(np.mean(regime_ev_haircuts["TREND"])) if len(regime_ev_haircuts["TREND"]) else 0.0,
        "mean_rev_lambda": float(np.mean(rev_lam_list)) if len(rev_lam_list) else 0.0,
        "mean_exp_exit_move": float(np.mean(exp_exit_move_list)) if len(exp_exit_move_list) else 0.0,
        "mean_trend_kappa": float(np.mean(trend_kappas)) if len(trend_kappas) else 0.0,
        "mean_trend_strength": float(np.mean(trend_strengths)) if len(trend_strengths) else 0.0,
        "mean_trend_strength_trade": float(np.mean(trend_strength_trade)) if len(trend_strength_trade) else 0.0,
        "mean_trend_scale_used": float(np.mean(trend_scale_used)) if len(trend_scale_used) else 1.0,
        "mean_dir_edge_scale": float(np.mean(dir_edge_scale_used)) if len(dir_edge_scale_used) else 1.0,
        "mean_dir_edge_scale_trend": float(np.mean(dir_edge_scale_trend)) if len(dir_edge_scale_trend) else 1.0,
        "mean_dir_edge_scale_meanrev": float(np.mean(dir_edge_scale_meanrev)) if len(dir_edge_scale_meanrev) else 1.0,
        "mean_trend_tp_strength": float(np.mean(trend_tp_strengths)) if len(trend_tp_strengths) else 0.0,
        "exp_exit_mode_meanrev": float(exp_exit_modes["MEANREV"]),
        "exp_exit_mode_trend": float(exp_exit_modes["TREND"]),
        "n_rev_attempts": float(len(rev_lam_list)),
        "sum_rev_lambda": float(np.sum(rev_lam_list)) if len(rev_lam_list) else 0.0,
        "sum_exp_exit_move": float(np.sum(exp_exit_move_list)) if len(exp_exit_move_list) else 0.0,
        "passive_fills": float(passive_fills),
        "taker_fills": float(taker_fills),
        "mean_passive_entry_dist": float(np.mean(passive_entry_dists)) if len(passive_entry_dists) else 0.0,
        "mean_taker_entry_dist": float(np.mean(taker_entry_dists)) if len(taker_entry_dists) else 0.0,
        "posted_orders": float(posted_orders),
        "posted_fills": float(posted_fills),
        "posted_cancels": float(posted_cancels),
        "taker_fallback_fills": float(taker_fallback_fills),
        "mean_pending_live": float(pending_live_sum / max(T, 1)),
        "mean_maker_entry_dist": float(np.mean(maker_entry_dists)) if len(maker_entry_dists) else 0.0,
        "mean_fallback_entry_dist": float(np.mean(fallback_entry_dists)) if len(fallback_entry_dists) else 0.0,
        "mean_queue_pos_posted": float(np.mean(queue_pos_posted)) if len(queue_pos_posted) else 0.0,
        "mean_queue_pos_filled": float(np.mean(queue_pos_filled)) if len(queue_pos_filled) else 0.0,
        "mean_queue_touches_filled": float(np.mean(queue_touches_filled)) if len(queue_touches_filled) else 0.0,
        "quote_posts": float(quote_posts),
        "quote_fills": float(quote_fills),
        "quote_cancels": float(quote_cancels),
        "mean_quote_fill_dist": float(np.mean(quote_fill_dists)) if len(quote_fill_dists) else 0.0,
        "mean_quote_live": float(quote_live_sum / max(T, 1)),
        "quote_realized": float(quote_realized),
        "quote_trades_soft": float(quote_trades_soft),
        "quote_trades_neutral": float(quote_trades_neutral),
        "shock_events": float(shock_events),
        "mean_shock_abs": float(np.mean(shock_flow_abs)) if len(shock_flow_abs) else 0.0,
        "adverse_cluster_activations": float(adverse_cluster_activations),
        "mean_adverse_cluster_flow": float(np.mean(np.abs(adverse_cluster_flow))) if len(adverse_cluster_flow) else 0.0,
        "adverse_cluster_ticks": float(adverse_cluster_ticks),
        "mean_pf_latency_mult": float(np.mean(pf_latency_mults)) if len(pf_latency_mults) else 1.0,
        "mean_pf_process_vol": float(np.mean(pf_process_vols)) if len(pf_process_vols) else 0.02,
        "regime_ticks_meanrev": float(regime_counts["MEANREV"]),
        "regime_ticks_trend": float(regime_counts["TREND"]),
        "regime_trades_meanrev": float(regime_trade_counts["MEANREV"]),
        "regime_trades_trend": float(regime_trade_counts["TREND"]),
        "regime_realized_meanrev": float(regime_realized["MEANREV"]),
        "regime_realized_trend": float(regime_realized["TREND"]),
        "trend_fb_blocked": float(trend_fb_blocked),
        "trend_fb_allowed": float(trend_fb_allowed),
        "avg_toxicity_score": float(np.mean(tox_scores)) if len(tox_scores) else 0.0,
        "tox_ticks_soft": float(tox_ticks["SOFT"]),
        "tox_ticks_neutral": float(tox_ticks["NEUTRAL"]),
        "tox_ticks_toxic": float(tox_ticks["TOXIC"]),
        "tox_trades_soft": float(tox_trades["SOFT"]),
        "tox_trades_neutral": float(tox_trades["NEUTRAL"]),
        "tox_trades_toxic": float(tox_trades["TOXIC"]),
        "tox_realized_soft": float(tox_realized["SOFT"]),
        "tox_realized_neutral": float(tox_realized["NEUTRAL"]),
        "tox_realized_toxic": float(tox_realized["TOXIC"]),
        "bucket_counts": bucket_counts,
        "bucket_ev": bucket_ev,
        "bucket_pnl": bucket_pnl,
        "bucket_entry_x": bucket_entry_x,
        "bucket_exit_x": bucket_exit_x,
    }

# ----------------------------
# Batch runner
# ----------------------------
def run_sim(
    N_nights: int = 2000,
    T: int = 60,
    bankroll_start: float = 1000.0,
    bad_night_pct: float = 0.15,
    min_ev: float = 0.50,
    min_ev_per_dollar: Optional[float] = 0.015,
    seed: int = 7,
    verbose: bool = True,
):
    rng = np.random.default_rng(seed)
    results = []
    for i in range(N_nights):
        s = int(rng.integers(1, 1_000_000_000))
        res = run_one_night(
            seed=s,
            T=T,
            bankroll_start=bankroll_start,
            bad_night_pct=bad_night_pct,
            min_ev=min_ev,
            min_ev_per_dollar=min_ev_per_dollar,
        )
        results.append(res)

    pnl = np.array([r["pnl"] for r in results], dtype=float)
    dd = np.array([r["max_drawdown"] for r in results], dtype=float)
    exp = np.array([r["expected"] for r in results], dtype=float)
    tr = np.array([r["n_trades"] for r in results], dtype=float)
    sk = np.array([r["n_skips_ev"] for r in results], dtype=float)
    ex_tp = np.array([r.get("exit_tp", 0.0) for r in results], dtype=float)
    ex_flip = np.array([r.get("exit_flip", 0.0) for r in results], dtype=float)
    ex_ttl = np.array([r.get("exit_ttl", 0.0) for r in results], dtype=float)
    rz_tp = np.array([r.get("realized_tp", 0.0) for r in results], dtype=float)
    rz_flip = np.array([r.get("realized_flip", 0.0) for r in results], dtype=float)
    rz_ttl = np.array([r.get("realized_ttl", 0.0) for r in results], dtype=float)
    rz_eod = np.array([r.get("realized_eod", 0.0) for r in results], dtype=float)
    runner_extensions = np.array([r.get("runner_extensions", 0.0) for r in results], dtype=float)
    runner_realized = np.array([r.get("runner_realized", 0.0) for r in results], dtype=float)
    mean_runner_ttl_bonus = np.array([r.get("mean_runner_ttl_bonus", 0.0) for r in results], dtype=float)
    trend_reentries = np.array([r.get("trend_reentries", 0.0) for r in results], dtype=float)
    trend_reentry_realized = np.array([r.get("trend_reentry_realized", 0.0) for r in results], dtype=float)
    trend_reentry_attempts = np.array([r.get("trend_reentry_attempts", 0.0) for r in results], dtype=float)
    exit_early = np.array([r.get("exit_early", 0.0) for r in results], dtype=float)
    realized_early = np.array([r.get("realized_early", 0.0) for r in results], dtype=float)
    mean_tp_frac_meanrev = np.array([r.get("mean_tp_frac_meanrev", 0.0) for r in results], dtype=float)
    mean_tp_frac_trend = np.array([r.get("mean_tp_frac_trend", 0.0) for r in results], dtype=float)
    mean_ttl_trend = np.array([r.get("mean_ttl_trend", 0.0) for r in results], dtype=float)
    mean_ttl_meanrev = np.array([r.get("mean_ttl_meanrev", 0.0) for r in results], dtype=float)
    ent_x = np.array([r.get("avg_entry_extra", 0.0) for r in results], dtype=float)
    ex_x = np.array([r.get("avg_exit_extra", 0.0) for r in results], dtype=float)
    edge_raw_abs = np.array([r.get("mean_edge_raw_abs", 0.0) for r in results], dtype=float)
    edge_exec_abs = np.array([r.get("mean_edge_exec_abs", 0.0) for r in results], dtype=float)
    entry_dist_abs = np.array([r.get("mean_entry_dist_abs", 0.0) for r in results], dtype=float)
    edge_net_abs = np.array([r.get("mean_edge_net_abs", 0.0) for r in results], dtype=float)
    soft_used = np.array([r.get("soft_gate_used_ticks", 0.0) for r in results], dtype=float)
    soft_trades = np.array([r.get("soft_gate_trades", 0.0) for r in results], dtype=float)
    soft_saved = np.array([r.get("soft_gate_skips", 0.0) for r in results], dtype=float)
    ev_old_assump = np.array([r.get("mean_ev_old_assump", 0.0) for r in results], dtype=float)
    ev_rev_assump = np.array([r.get("mean_ev_reversion_assump", 0.0) for r in results], dtype=float)
    mean_ev_haircut_meanrev = np.array([r.get("mean_ev_haircut_meanrev", 0.0) for r in results], dtype=float)
    mean_ev_haircut_trend = np.array([r.get("mean_ev_haircut_trend", 0.0) for r in results], dtype=float)
    rev_lam = np.array([r.get("mean_rev_lambda", 0.0) for r in results], dtype=float)
    exp_exit_move = np.array([r.get("mean_exp_exit_move", 0.0) for r in results], dtype=float)
    mean_trend_kappa = np.array([r.get("mean_trend_kappa", 0.0) for r in results], dtype=float)
    mean_trend_strength = np.array([r.get("mean_trend_strength", 0.0) for r in results], dtype=float)
    mean_trend_strength_trade = np.array([r.get("mean_trend_strength_trade", 0.0) for r in results], dtype=float)
    mean_trend_scale_used = np.array([r.get("mean_trend_scale_used", 1.0) for r in results], dtype=float)
    mean_dir_edge_scale = np.array([r.get("mean_dir_edge_scale", 1.0) for r in results], dtype=float)
    mean_dir_edge_scale_trend = np.array([r.get("mean_dir_edge_scale_trend", 1.0) for r in results], dtype=float)
    mean_dir_edge_scale_meanrev = np.array([r.get("mean_dir_edge_scale_meanrev", 1.0) for r in results], dtype=float)
    mean_trend_tp_strength = np.array([r.get("mean_trend_tp_strength", 0.0) for r in results], dtype=float)
    exp_exit_mode_meanrev = np.array([r.get("exp_exit_mode_meanrev", 0.0) for r in results], dtype=float)
    exp_exit_mode_trend = np.array([r.get("exp_exit_mode_trend", 0.0) for r in results], dtype=float)
    rev_n = np.array([r.get("n_rev_attempts", 0.0) for r in results], dtype=float)
    rev_lam_sum = np.array([r.get("sum_rev_lambda", 0.0) for r in results], dtype=float)
    exp_exit_move_sum = np.array([r.get("sum_exp_exit_move", 0.0) for r in results], dtype=float)
    passive_fills = np.array([r.get("passive_fills", 0.0) for r in results], dtype=float)
    taker_fills = np.array([r.get("taker_fills", 0.0) for r in results], dtype=float)
    mean_passive_entry_dist = np.array([r.get("mean_passive_entry_dist", 0.0) for r in results], dtype=float)
    mean_taker_entry_dist = np.array([r.get("mean_taker_entry_dist", 0.0) for r in results], dtype=float)
    posted_orders = np.array([r.get("posted_orders", 0.0) for r in results], dtype=float)
    posted_fills = np.array([r.get("posted_fills", 0.0) for r in results], dtype=float)
    posted_cancels = np.array([r.get("posted_cancels", 0.0) for r in results], dtype=float)
    taker_fallback_fills = np.array([r.get("taker_fallback_fills", 0.0) for r in results], dtype=float)
    mean_pending_live = np.array([r.get("mean_pending_live", 0.0) for r in results], dtype=float)
    mean_maker_entry_dist = np.array([r.get("mean_maker_entry_dist", 0.0) for r in results], dtype=float)
    mean_fallback_entry_dist = np.array([r.get("mean_fallback_entry_dist", 0.0) for r in results], dtype=float)
    mean_queue_pos_posted = np.array([r.get("mean_queue_pos_posted", 0.0) for r in results], dtype=float)
    mean_queue_pos_filled = np.array([r.get("mean_queue_pos_filled", 0.0) for r in results], dtype=float)
    mean_queue_touches_filled = np.array([r.get("mean_queue_touches_filled", 0.0) for r in results], dtype=float)
    quote_posts = np.array([r.get("quote_posts", 0.0) for r in results], dtype=float)
    quote_fills = np.array([r.get("quote_fills", 0.0) for r in results], dtype=float)
    quote_cancels = np.array([r.get("quote_cancels", 0.0) for r in results], dtype=float)
    mean_quote_fill_dist = np.array([r.get("mean_quote_fill_dist", 0.0) for r in results], dtype=float)
    mean_quote_live = np.array([r.get("mean_quote_live", 0.0) for r in results], dtype=float)
    quote_realized = np.array([r.get("quote_realized", 0.0) for r in results], dtype=float)
    quote_trades_soft = np.array([r.get("quote_trades_soft", 0.0) for r in results], dtype=float)
    quote_trades_neutral = np.array([r.get("quote_trades_neutral", 0.0) for r in results], dtype=float)
    shock_events = np.array([r.get("shock_events", 0.0) for r in results], dtype=float)
    mean_shock_abs = np.array([r.get("mean_shock_abs", 0.0) for r in results], dtype=float)
    adverse_cluster_activations = np.array([r.get("adverse_cluster_activations", 0.0) for r in results], dtype=float)
    mean_adverse_cluster_flow = np.array([r.get("mean_adverse_cluster_flow", 0.0) for r in results], dtype=float)
    adverse_cluster_ticks = np.array([r.get("adverse_cluster_ticks", 0.0) for r in results], dtype=float)
    pf_latency_mult = np.array([r.get("mean_pf_latency_mult", 1.0) for r in results], dtype=float)
    pf_process_vol = np.array([r.get("mean_pf_process_vol", 0.02) for r in results], dtype=float)
    regime_ticks_meanrev = np.array([r.get("regime_ticks_meanrev", 0.0) for r in results], dtype=float)
    regime_ticks_trend = np.array([r.get("regime_ticks_trend", 0.0) for r in results], dtype=float)
    regime_trades_meanrev = np.array([r.get("regime_trades_meanrev", 0.0) for r in results], dtype=float)
    regime_trades_trend = np.array([r.get("regime_trades_trend", 0.0) for r in results], dtype=float)
    regime_realized_meanrev = np.array([r.get("regime_realized_meanrev", 0.0) for r in results], dtype=float)
    regime_realized_trend = np.array([r.get("regime_realized_trend", 0.0) for r in results], dtype=float)
    trend_fb_blocked = np.array([r.get("trend_fb_blocked", 0.0) for r in results], dtype=float)
    trend_fb_allowed = np.array([r.get("trend_fb_allowed", 0.0) for r in results], dtype=float)
    total_rev_attempts = float(np.sum(rev_n))
    rev_lam_weighted = float(np.sum(rev_lam_sum) / total_rev_attempts) if total_rev_attempts > 0 else 0.0
    exp_exit_move_weighted = float(np.sum(exp_exit_move_sum) / total_rev_attempts) if total_rev_attempts > 0 else 0.0
    tox_avg = np.array([r.get("avg_toxicity_score", 0.0) for r in results], dtype=float)
    tox_ticks_soft = np.array([r.get("tox_ticks_soft", 0.0) for r in results], dtype=float)
    tox_ticks_neutral = np.array([r.get("tox_ticks_neutral", 0.0) for r in results], dtype=float)
    tox_ticks_toxic = np.array([r.get("tox_ticks_toxic", 0.0) for r in results], dtype=float)
    tox_trades_soft = np.array([r.get("tox_trades_soft", 0.0) for r in results], dtype=float)
    tox_trades_neutral = np.array([r.get("tox_trades_neutral", 0.0) for r in results], dtype=float)
    tox_trades_toxic = np.array([r.get("tox_trades_toxic", 0.0) for r in results], dtype=float)
    tox_realized_soft = np.array([r.get("tox_realized_soft", 0.0) for r in results], dtype=float)
    tox_realized_neutral = np.array([r.get("tox_realized_neutral", 0.0) for r in results], dtype=float)
    tox_realized_toxic = np.array([r.get("tox_realized_toxic", 0.0) for r in results], dtype=float)

    # Aggregate edge-bucket attribution across the whole simulation
    BUCKET_LABELS = ("0.04–0.05", "0.05–0.06", "0.06–0.08", "0.08+")
    b_counts: Dict[str, int] = {k: 0 for k in BUCKET_LABELS}
    b_ev: Dict[str, float] = {k: 0.0 for k in BUCKET_LABELS}
    b_pnl: Dict[str, float] = {k: 0.0 for k in BUCKET_LABELS}
    b_entx: Dict[str, float] = {k: 0.0 for k in BUCKET_LABELS}
    b_exx: Dict[str, float] = {k: 0.0 for k in BUCKET_LABELS}

    for r in results:
        bc = r.get("bucket_counts", {}) or {}
        be = r.get("bucket_ev", {}) or {}
        bp = r.get("bucket_pnl", {}) or {}
        bx_ent = r.get("bucket_entry_x", {}) or {}
        bx_ex = r.get("bucket_exit_x", {}) or {}
        for k in BUCKET_LABELS:
            b_counts[k] += int(bc.get(k, 0))
            b_ev[k] += float(be.get(k, 0.0))
            b_pnl[k] += float(bp.get(k, 0.0))
            b_entx[k] += float(bx_ent.get(k, 0.0))
            b_exx[k] += float(bx_ex.get(k, 0.0))

    def pct(x, q): return float(np.percentile(x, q))

    if verbose:
        print("\nAgent-based totals simulator (paper) — results")
        print("------------------------------------------------")
        print(f"N nights: {N_nights} | ticks/night: {T}")
        if min_ev_per_dollar is not None:
            gate_str = f"EV gate: {min_ev_per_dollar:.3f} per $1 (≈{min_ev_per_dollar*100:.1f}% of stake)"
        else:
            gate_str = f"EV gate: ${min_ev:.2f}"
        print(f"Bankroll start: ${bankroll_start:,.0f} | bad-night cap: {bad_night_pct*100:.0f}% | {gate_str}")
        print("")
        print(f"Avg nightly PnL:        ${pnl.mean():.2f}")
        print(f"Median nightly PnL:     ${np.median(pnl):.2f}")
        print(f"PnL 5th pct:            ${pct(pnl, 5):.2f}")
        print(f"PnL 1st pct:            ${pct(pnl, 1):.2f}")
        print(f"Avg max drawdown:       ${dd.mean():.2f}")
        print(f"DD 95th pct (bad):      ${pct(dd, 95):.2f}")
        print("")
        print(f"Avg trades/night:       {tr.mean():.2f}")
        print(f"Avg skips (EV gate):    {sk.mean():.2f}")
        print(f"Avg expected/night:     ${exp.mean():.2f}  (sum of trade EV at entry)")
        print(f"Avg exits/night:        TP={ex_tp.mean():.2f} | flip={ex_flip.mean():.2f} | ttl={ex_ttl.mean():.2f}")
        print(f"Avg realized/night:     TP=${rz_tp.mean():.2f} | flip=${rz_flip.mean():.2f} | ttl=${rz_ttl.mean():.2f} | eod=${rz_eod.mean():.2f}")
        tp_den = float(ex_tp.sum())
        flip_den = float(ex_flip.sum())
        ttl_den = float(ex_ttl.sum())
        tp_avg = float(rz_tp.sum() / tp_den) if tp_den > 0 else 0.0
        flip_avg = float(rz_flip.sum() / flip_den) if flip_den > 0 else 0.0
        ttl_avg = float(rz_ttl.sum() / ttl_den) if ttl_den > 0 else 0.0
        print(f"Avg pnl/exit event:     TP=${tp_avg:.2f} | flip=${flip_avg:.2f} | ttl=${ttl_avg:.2f}")
        print(f"Runner extensions/night: {runner_extensions.mean():.2f}")
        print(f"Runner realized/night:   ${runner_realized.mean():.2f}")
        print(f"Mean runner TTL bonus:  {mean_runner_ttl_bonus.mean():.2f}")
        print(f"TREND reentries/night:  {trend_reentries.mean():.2f}")
        print(f"TREND reentry attempts: {trend_reentry_attempts.mean():.2f}")
        print(f"TREND reentry realized: ${trend_reentry_realized.mean():.2f}")
        print(f"Early exits/night:      {exit_early.mean():.2f}")
        print(f"Early realized/night:   ${realized_early.mean():.2f}")
        print(f"Mean TP frac by regime: MEANREV={mean_tp_frac_meanrev.mean():.2f} | TREND={mean_tp_frac_trend.mean():.2f}")
        print(f"Mean TREND TP strength: {mean_trend_tp_strength.mean():.3f}")
        print(f"Mean TTL by regime:     TREND={mean_ttl_trend.mean():.2f} | MEANREV={mean_ttl_meanrev.mean():.2f}")
        print(f"EV model (entry):       old=${ev_old_assump.mean():.2f} | rev=${ev_rev_assump.mean():.2f} | d=${(ev_rev_assump.mean() - ev_old_assump.mean()):+.2f}")
        print(f"Expected reversion:     night_wtd λ={rev_lam.mean():.3f} | night_wtd |exit_mid-mid|={exp_exit_move.mean():.4f}")
        print(f"Expected reversion:     trade_wtd λ={rev_lam_weighted:.3f} | trade_wtd |exit_mid-mid|={exp_exit_move_weighted:.4f}")
        print(f"Reversion attempts/night:{rev_n.mean():.2f}  (total attempts={int(total_rev_attempts)})")
        print(f"Avg adverse extra:      entry={ent_x.mean():.4f} | exit={ex_x.mean():.4f} (abs prob units)")
        print(f"Mean |p_fair-mid|:      {edge_raw_abs.mean():.4f} per tick")
        print(f"Mean |p_exec-mid|:      {edge_exec_abs.mean():.4f} per tick")
        print(f"Mean |p_fill-mid|:      {entry_dist_abs.mean():.4f} on attempted trades")
        print(f"Mean |edge_net|:        {edge_net_abs.mean():.4f} on attempted trades")
        total_entry_fills = passive_fills.sum() + taker_fills.sum()
        passive_share = float(passive_fills.sum() / total_entry_fills) if total_entry_fills > 0 else 0.0
        print(f"Passive fills/night:    {passive_fills.mean():.2f}  (all passive fills incl. quotes)")
        print(f"Taker fills/night:      {taker_fills.mean():.2f}  (directional fallback/taker only)")
        print(f"Passive fill share:     {passive_share*100:.1f}%")
        print(f"Mean passive |fill-mid|:{mean_passive_entry_dist.mean():.4f}")
        print(f"Mean taker |fill-mid|:  {mean_taker_entry_dist.mean():.4f}")
        total_posts = posted_orders.sum()
        post_fill_rate = float(posted_fills.sum() / total_posts) if total_posts > 0 else 0.0
        fallback_share = float(taker_fallback_fills.sum() / max(posted_fills.sum() + taker_fallback_fills.sum(), 1.0))
        print(f"Posted orders/night:    {posted_orders.mean():.2f}")
        print(f"Posted fills/night:     {posted_fills.mean():.2f}")
        print(f"Posted cancels/night:   {posted_cancels.mean():.2f}")
        print(f"Fallback fills/night:   {taker_fallback_fills.mean():.2f}")
        print(f"Post fill rate:         {post_fill_rate*100:.1f}%")
        print(f"Fallback fill share:    {fallback_share*100:.1f}%")
        print(f"Mean pending live:      {mean_pending_live.mean():.2f}")
        print(f"Mean maker |fill-mid|:  {mean_maker_entry_dist.mean():.4f}")
        print(f"Mean fallback |fill-mid|:{mean_fallback_entry_dist.mean():.4f}")
        print(f"Mean queue pos posted:  {mean_queue_pos_posted.mean():.3f}")
        print(f"Mean queue pos filled:  {mean_queue_pos_filled.mean():.3f}")
        print(f"Mean touches/fill:      {mean_queue_touches_filled.mean():.2f}")
        total_quote_posts = quote_posts.sum()
        quote_fill_rate = float(quote_fills.sum() / total_quote_posts) if total_quote_posts > 0 else 0.0
        print(f"Quote posts/night:      {quote_posts.mean():.2f}")
        print(f"Quote fills/night:      {quote_fills.mean():.2f}")
        print(f"Quote cancels/night:    {quote_cancels.mean():.2f}")
        print(f"Quote fill rate:        {quote_fill_rate*100:.1f}%")
        print(f"Mean quote |fill-mid|:  {mean_quote_fill_dist.mean():.4f}")
        print(f"Mean quote live:        {mean_quote_live.mean():.2f}")
        print(f"Quote realized/night:   ${quote_realized.mean():.2f}")
        print(f"Quote trades/night:     SOFT={quote_trades_soft.mean():.2f} | NEUTRAL={quote_trades_neutral.mean():.2f}")
        print(f"Mean PF latency mult:   {pf_latency_mult.mean():.3f}")
        print(f"Mean PF process vol:    {pf_process_vol.mean():.4f}")
        print(f"Regime ticks/night:     MEANREV={regime_ticks_meanrev.mean():.2f} | TREND={regime_ticks_trend.mean():.2f}")
        print(f"Regime trades/night:    MEANREV={regime_trades_meanrev.mean():.2f} | TREND={regime_trades_trend.mean():.2f}")
        print(f"Regime realized/night:  MEANREV=${regime_realized_meanrev.mean():.2f} | TREND=${regime_realized_trend.mean():.2f}")
        total_trend_fb_checks = trend_fb_blocked.mean() + trend_fb_allowed.mean()
        print(f"TREND fb/night:         allow={trend_fb_allowed.mean():.2f} | block={trend_fb_blocked.mean():.2f}")
        if total_trend_fb_checks > 0:
            print(f"TREND fb allow rate:    {100.0 * trend_fb_allowed.mean() / total_trend_fb_checks:.1f}%")
        print(f"EV haircut by regime:   MEANREV={mean_ev_haircut_meanrev.mean():.3f} | TREND={mean_ev_haircut_trend.mean():.3f}")
        print(f"Exit model usage/night: MEANREV={exp_exit_mode_meanrev.mean():.2f} | TREND={exp_exit_mode_trend.mean():.2f}")
        print(f"Mean trend kappa:       {mean_trend_kappa.mean():.3f}")
        print(f"Mean trend strength:    {mean_trend_strength.mean():.3f}")
        print(f"Trend strength/trade:   {mean_trend_strength_trade.mean():.3f}")
        print(f"Mean trend stake scale: {mean_trend_scale_used.mean():.3f}")
        print(f"Mean dir edge scale:    {mean_dir_edge_scale.mean():.3f}")
        print(f"Dir edge scale/regime:  TREND={mean_dir_edge_scale_trend.mean():.3f} | MEANREV={mean_dir_edge_scale_meanrev.mean():.3f}")
        soft_trade_share = float(soft_trades.sum() / tr.sum()) if float(tr.sum()) > 0 else 0.0
        print(f"Avg toxicity score:     {tox_avg.mean():.4f}")
        print(f"Toxicity ticks/night:   SOFT={tox_ticks_soft.mean():.2f} | NEUTRAL={tox_ticks_neutral.mean():.2f} | TOXIC={tox_ticks_toxic.mean():.2f}")
        print(f"Toxicity trades/night:  SOFT={tox_trades_soft.mean():.2f} | NEUTRAL={tox_trades_neutral.mean():.2f} | TOXIC={tox_trades_toxic.mean():.2f}")
        print(f"Toxicity realized/night:SOFT=${tox_realized_soft.mean():.2f} | NEUTRAL=${tox_realized_neutral.mean():.2f} | TOXIC=${tox_realized_toxic.mean():.2f}")
        print(f"Adverse cluster acts/night:{adverse_cluster_activations.mean():.2f}")
        print(f"Adverse cluster ticks/night:{adverse_cluster_ticks.mean():.2f}")
        print(f"Mean adverse cluster flow:{mean_adverse_cluster_flow.mean():.4f}")
        print(f"Shock events/night:     {shock_events.mean():.2f}")
        print(f"Mean |shock flow|:      {mean_shock_abs.mean():.2f}")
        print(f"Soft-gate ticks/night:  {soft_used.mean():.2f}")
        print(f"Soft-gate trades/night: {soft_trades.mean():.2f}")
        print(f"Saved-by-soft/night:    {soft_saved.mean():.2f}")
        print(f"Soft-gate trade share:  {soft_trade_share*100:.1f}% of all trades")
        print(f"EV gap (exp-real):      ${(exp.mean() - pnl.mean()):.2f} per night")
        print(f"Exit mix (avg/night):   TP={ex_tp.mean():.2f} | flip={ex_flip.mean():.2f} | ttl={ex_ttl.mean():.2f}")
        print("")
        win_rate = float(np.mean(pnl > 0))
        print(f"Win-rate nights:        {win_rate*100:.1f}%")
        print("")
        print("Edge bucket attribution (all trades across sim)")
        print("------------------------------------------------")
        print(f"{'edge bucket':<12} {'trades':>7} {'avg EV':>10} {'realized':>10} {'entX':>8} {'exX':>8}")
        print("------------------------------------------------")
        for k in BUCKET_LABELS:
            c = b_counts[k]
            avg_ev = (b_ev[k] / c) if c > 0 else 0.0
            rz = b_pnl[k]
            avg_entx = (b_entx[k] / c) if c > 0 else 0.0
            avg_exx = (b_exx[k] / c) if c > 0 else 0.0
            print(f"{k:<12} {c:7d} {avg_ev:10.2f} {rz:10.2f} {avg_entx:8.4f} {avg_exx:8.4f}")
        if sum(b_counts.values()) == 0:
            print("(no bucketed trades)")

# ----------------------------
# Parameter grid sweep runner
# ----------------------------
def run_grid():
    """Quick sweep to compare risk caps and EV gates under adverse selection."""
    configs = []
    for bad in (0.10, 0.15):
        # EV-per-dollar gates (≈ percent-of-stake)
        for mev_per_dollar in (0.010, 0.015, 0.020):
            configs.append((bad, mev_per_dollar))

    N = 800
    T = 60
    seed = 7
    bankroll = 1000.0

    rows = []
    for bad, mev in configs:
        # run silently
        rng = np.random.default_rng(seed)
        results = []
        for i in range(N):
            s = int(rng.integers(1, 1_000_000_000))
            results.append(
                run_one_night(
                    seed=s,
                    T=T,
                    bankroll_start=bankroll,
                    bad_night_pct=bad,
                    min_ev=0.0,
                    min_ev_per_dollar=mev,
                )
            )

        pnl = np.array([r["pnl"] for r in results], dtype=float)
        dd = np.array([r["max_drawdown"] for r in results], dtype=float)
        tr = np.array([r["n_trades"] for r in results], dtype=float)
        sk = np.array([r["n_skips_ev"] for r in results], dtype=float)

        rows.append(
            (
                bad,
                mev,
                float(pnl.mean()),
                float(np.median(pnl)),
                float(np.percentile(pnl, 5)),
                float(np.percentile(pnl, 1)),
                float(np.mean(dd)),
                float(np.percentile(dd, 95)),
                float(tr.mean()),
                float(sk.mean()),
                float(np.mean(pnl > 0)) * 100.0,
            )
        )

    print("\nGrid sweep (adverse fills ON) — per-night summary")
    print("N=%d nights per config | ticks/night=%d" % (N, T))
    print("-----------------------------------------------------------------------------------------------")
    print(f"{'cap%':>5} {'minEV%':>7} {'mean$':>7} {'med$':>7} {'p5$':>7} {'p1$':>7} {'avgDD$':>8} {'DD95$':>7} {'tr/n':>6} {'skEV':>6} {'win%':>6}")
    print("-----------------------------------------------------------------------------------------------")
    for bad, mev, mean, med, p5, p1, avgdd, dd95, trn, skev, win in rows:
        print(f"{bad*100:5.0f} {mev*100:7.2f} {mean:7.2f} {med:7.2f} {p5:7.2f} {p1:7.2f} {avgdd:8.2f} {dd95:7.2f} {trn:6.2f} {skev:6.2f} {win:6.1f}")

if __name__ == "__main__":
    # quick sweep first
    run_grid()

    # then one detailed run with your preferred defaults
    run_sim(
        N_nights=2000,
        T=60,
        bankroll_start=1000.0,
        bad_night_pct=0.15,  # try 0.10 vs 0.15
        min_ev=0.50,         # legacy (ignored when min_ev_per_dollar is not None)
        min_ev_per_dollar=0.015,
        seed=7,
        verbose=True,
    )
