import numpy as np
from typing import Dict, Optional, Tuple

from shared.math_utils import expit, logit


class LogitBiasParticleFilter:
    def __init__(
        self,
        N: int = 4000,
        prior_bias: float = 0.0,
        process_vol: float = 0.03,
        seed: int = 42,
        resample_ess_frac: float = 0.5,
    ):
        self.N = int(N)
        self.base_process_vol = float(process_vol)
        self.resample_threshold = float(resample_ess_frac) * self.N
        self.rng = np.random.default_rng(seed)

        self.b = prior_bias + self.rng.normal(0, 0.5, size=self.N)
        self.w = np.ones(self.N) / self.N

    def ess(self) -> float:
        return float(1.0 / np.sum(self.w**2))

    def _systematic_resample(self) -> None:
        cdf = np.cumsum(self.w)
        u0 = self.rng.uniform(0, 1 / self.N)
        u = u0 + np.arange(self.N) / self.N
        idx = np.searchsorted(cdf, u)
        self.b = self.b[idx]
        self.w.fill(1.0 / self.N)

    def estimate(self, p_base: float) -> float:
        base_logit = float(logit(np.array([p_base]))[0])
        p = expit(base_logit + self.b)
        return float(np.average(p, weights=self.w))

    def credible_interval(self, p_base: float, alpha: float = 0.05) -> Tuple[float, float]:
        base_logit = float(logit(np.array([p_base]))[0])
        p = expit(base_logit + self.b)
        idx = np.argsort(p)
        p_sorted = p[idx]
        w_sorted = self.w[idx]
        cdf = np.cumsum(w_sorted)
        lo = p_sorted[np.searchsorted(cdf, alpha / 2)]
        hi = p_sorted[np.searchsorted(cdf, 1 - alpha / 2)]
        return float(lo), float(hi)

    def update(
        self,
        p_base: float,
        observations: Dict[str, Dict],
        process_vol: Optional[float] = None,
    ) -> None:
        if process_vol is None:
            process_vol = self.base_process_vol

        self.b += self.rng.normal(0, process_vol, size=self.N)

        base_logit = float(logit(np.array([p_base]))[0])
        p_particles = expit(base_logit + self.b)

        log_w = np.log(self.w + 1e-300)

        for _, obs in observations.items():
            y = float(obs["p"])
            noise = float(obs["obs_noise"])
            wt = float(obs.get("weight", 1.0))
            eff_noise = noise / np.sqrt(max(wt, 1e-6))
            log_like = -0.5 * ((y - p_particles) / eff_noise) ** 2
            log_w += log_like

        log_w -= np.max(log_w)
        self.w = np.exp(log_w)
        s = np.sum(self.w)
        if s == 0 or (not np.isfinite(s)):
            self.w.fill(1.0 / self.N)
        else:
            self.w /= s

        if self.ess() < self.resample_threshold:
            self._systematic_resample()