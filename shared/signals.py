import numpy as np


def exec_alpha_dynamic(regime_thin: bool, mid_delta: float, base: float = 0.75, thin: float = 0.50) -> float:
   """Haircut factor that shrinks fair value toward the market when execution is hard.


   Lower alpha => we assume we capture less of model-vs-market edge.
   """
   a = float(thin if regime_thin else base)
   # If the mid is moving quickly, assume toxic flow / latency => further haircut
   if mid_delta >= 0.010:
       a *= 0.80
   elif mid_delta >= 0.006:
       a *= 0.90
   return float(np.clip(a, 0.25, 0.95))


def toxicity_score(mid_delta: float, spread: float, vol: float, book_mid_gap: float) -> float:
   """Heuristic toxicity score in [0, 1].


   Higher when:
     - mid is moving fast
     - spread is wide
     - volume/liquidity is low
     - market is diverging from book
   """
   s_mid = np.clip(mid_delta / 0.015, 0.0, 1.0)
   s_spread = np.clip((spread - 0.02) / 0.06, 0.0, 1.0)
   s_liq = np.clip((50.0 - vol) / 50.0, 0.0, 1.0)
   s_gap = np.clip(book_mid_gap / 0.05, 0.0, 1.0)


   # weighted average
   score = 0.35 * s_mid + 0.30 * s_spread + 0.20 * s_liq + 0.15 * s_gap
   return float(np.clip(score, 0.0, 1.0))



def toxicity_regime(score: float) -> str:
   """Map toxicity score to SOFT / NEUTRAL / TOXIC."""
   if score >= 0.65:
       return "TOXIC"
   if score <= 0.30:
       return "SOFT"
   return "NEUTRAL"



def pf_latency_multiplier(mid_delta: float, tox_reg: str, spread: float) -> float:
   """Reduce PF responsiveness in fast / toxic tapes.


   Returns a multiplier in [0.35, 1.0] applied to observation weights.
   Lower => slower PF reaction.
   """
   mult = 1.0


   # Fast tape => slower assimilation
   if mid_delta >= 0.012:
       mult *= 0.45
   elif mid_delta >= 0.008:
       mult *= 0.65
   elif mid_delta >= 0.005:
       mult *= 0.80


   # Toxic regime => slower trust in incoming prints
   if tox_reg == "TOXIC":
       mult *= 0.70
   elif tox_reg == "SOFT":
       mult *= 1.05


   # Wider spreads => noisier microstructure => slower PF
   if spread >= 0.06:
       mult *= 0.80
   elif spread <= 0.025:
       mult *= 1.05


   return float(np.clip(mult, 0.35, 1.00))



def market_regime(mid_delta: float, tox_reg: str, run_signal: float, pace_signal: float) -> str:
   """Classify tape as MEANREV or TREND.


   Heuristics:
   - fast tape + toxic / strong run => TREND
   - calmer tape / softer conditions => MEANREV
   """
   trend_score = 0.0


   if mid_delta >= 0.012:
       trend_score += 1.2
   elif mid_delta >= 0.008:
       trend_score += 0.8
   elif mid_delta >= 0.005:
       trend_score += 0.4


   if tox_reg == "TOXIC":
       trend_score += 0.9
   elif tox_reg == "SOFT":
       trend_score -= 0.4


   if abs(run_signal) >= 2.0:
       trend_score += 0.8
   elif abs(run_signal) >= 1.0:
       trend_score += 0.4


   if abs(pace_signal) >= 2.5:
       trend_score += 0.4


   return "TREND" if trend_score >= 1.2 else "MEANREV"


def trend_strength_score(mid_delta: float, tox_reg: str, run_signal: float, pace_signal: float, spread: float) -> float:
   """Continuous TREND strength score in [0, 1].


   Higher means a cleaner / stronger continuation environment.
   This is not the same as regime classification; it refines TREND quality.
   """
   s = 0.0


   # Faster tapes tend to support continuation, but cap the effect.
   s += 0.35 * np.clip(mid_delta / 0.012, 0.0, 1.0)


   # Strong run / pace signals increase trend conviction.
   s += 0.25 * np.clip(abs(run_signal) / 3.0, 0.0, 1.0)
   s += 0.20 * np.clip(abs(pace_signal) / 3.0, 0.0, 1.0)


   # Toxic tapes can trend, but are harder to monetize cleanly.
   if tox_reg == "TOXIC":
       s -= 0.15
   elif tox_reg == "SOFT":
       s += 0.10


   # Extremely wide spreads reduce monetizable trend quality.
   if spread >= 0.06:
       s -= 0.10
   elif spread <= 0.03:
       s += 0.05


   return float(np.clip(s, 0.0, 1.0))
