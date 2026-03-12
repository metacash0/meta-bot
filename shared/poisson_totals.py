from __future__ import annotations

import math


def poisson_pmf(k: int, lam: float) -> float:
    k = max(0, int(k))
    lam = max(0.0, float(lam))
    if lam == 0.0:
        return 1.0 if k == 0 else 0.0

    prob = math.exp(-lam)
    for i in range(1, k + 1):
        prob *= lam / float(i)
    return max(0.0, float(prob))


def poisson_cdf(k: int, lam: float) -> float:
    k = int(k)
    if k < 0:
        return 0.0
    lam = max(0.0, float(lam))
    total = 0.0
    for i in range(k + 1):
        total += poisson_pmf(i, lam)
    return min(1.0, max(0.0, total))


def poisson_over_prob(lambda_total: float, total_point: float) -> float:
    lambda_total = max(0.0, float(lambda_total))
    total_point = float(total_point)

    whole = math.floor(total_point)
    if abs(total_point - (whole + 0.5)) > 1e-9:
        raise ValueError("unsupported total_point; only x.5 lines are supported")

    threshold = int(whole + 1)
    return max(0.0, min(1.0, 1.0 - poisson_cdf(threshold - 1, lambda_total)))


def infer_lambda_total(total_point: float, p_over: float, tol: float = 1e-6) -> dict:
    total_point = float(total_point)
    p_over = float(p_over)
    tol = max(1e-12, float(tol))

    if p_over <= 0.0 or p_over >= 1.0:
        raise ValueError("p_over must be strictly between 0 and 1")

    low = 0.01
    high = 8.0
    fit_low = poisson_over_prob(low, total_point)
    fit_high = poisson_over_prob(high, total_point)
    if p_over < fit_low or p_over > fit_high:
        raise ValueError("target p_over is outside the solvable range")

    for _ in range(200):
        mid = (low + high) / 2.0
        fit_mid = poisson_over_prob(mid, total_point)
        if abs(fit_mid - p_over) <= tol:
            return {
                "lambda_total": float(mid),
                "total_point": total_point,
                "target_p_over": p_over,
                "fit_p_over": float(fit_mid),
                "fit_error": float(abs(fit_mid - p_over)),
            }
        if fit_mid < p_over:
            low = mid
        else:
            high = mid

    mid = (low + high) / 2.0
    fit_mid = poisson_over_prob(mid, total_point)
    return {
        "lambda_total": float(mid),
        "total_point": total_point,
        "target_p_over": p_over,
        "fit_p_over": float(fit_mid),
        "fit_error": float(abs(fit_mid - p_over)),
    }


if __name__ == "__main__":
    examples = [
        {"total_point": 2.5, "p_over": 0.55},
        {"total_point": 2.5, "p_over": 0.45},
        {"total_point": 3.5, "p_over": 0.41},
    ]

    for example in examples:
        print(infer_lambda_total(**example))
