import numpy as np


def stake_ladder(edge_net: float, regime_thin: bool, min_stake=20.0) -> float:
   """Smooth edge^2 sizing with practical bounds."""
   a = abs(edge_net)
   if a < 0.085:
       stake = 0.0
   else:
       # Map 0.085..0.14 to 20..50 with quadratic growth.
       x = (a - 0.085) / 0.055
       x = float(np.clip(x, 0.0, 1.0))
       stake = 20.0 + 30.0 * (x ** 2)
       stake = float(np.clip(stake, 20.0, 50.0))


   if regime_thin:
       stake *= 0.5


   stake = float(stake)
   return float(stake if stake >= float(min_stake) else 0.0)



def directional_edge_scale(edge_net: float, regime_mode: str, trend_strength: float) -> float:
   a = float(abs(edge_net))


   if a < 0.08:
       edge_mult = 1.00
   elif a < 0.10:
       edge_mult = 1.00
   elif a < 0.12:
       edge_mult = 1.08
   elif a < 0.14:
       edge_mult = 1.18
   else:
       edge_mult = 1.30


   if regime_mode == "TREND":
       regime_mult = 0.97 + 0.20 * float(np.clip(trend_strength, 0.0, 1.0))
   else:
       regime_mult = 0.90


   out = edge_mult * regime_mult
   return float(np.clip(out, 0.85, 1.35))


def quote_stake(edge_net: float, regime_thin: bool, min_stake: float = 4.0) -> float:
   """Small passive inventory quote sizing for modest edges.


   Intended for 0.02–0.06 edge range.
   """
   a = abs(edge_net)
   if a < 0.02:
       stake = 0.0
   elif a < 0.03:
       stake = 4.0
   elif a < 0.045:
       stake = 6.0
   else:
       stake = 8.0


   if regime_thin:
       stake *= 0.5


   return float(stake if stake >= float(min_stake) else 0.0)
