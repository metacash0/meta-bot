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


def poisson_outcome_probs_from_lambdas(
    lambda_home: float,
    lambda_away: float,
    max_goals: int = 10,
) -> dict:
    lambda_home = max(0.0, float(lambda_home))
    lambda_away = max(0.0, float(lambda_away))
    max_goals = max(0, int(max_goals))

    home_win_prob = 0.0
    draw_prob = 0.0
    away_win_prob = 0.0

    for home_goals in range(max_goals + 1):
        p_home_goals = poisson_pmf(home_goals, lambda_home)
        for away_goals in range(max_goals + 1):
            p_away_goals = poisson_pmf(away_goals, lambda_away)
            joint_prob = max(0.0, p_home_goals * p_away_goals)
            if home_goals > away_goals:
                home_win_prob += joint_prob
            elif home_goals < away_goals:
                away_win_prob += joint_prob
            else:
                draw_prob += joint_prob

    total = home_win_prob + draw_prob + away_win_prob
    if total > 0.0:
        home_win_prob /= total
        draw_prob /= total
        away_win_prob /= total

    return {
        "home_win_prob": max(0.0, home_win_prob),
        "draw_prob": max(0.0, draw_prob),
        "away_win_prob": max(0.0, away_win_prob),
    }


def normalize_probs(p_home: float, p_draw: float, p_away: float) -> tuple[float, float, float]:
    values = [max(0.0, float(p_home)), max(0.0, float(p_draw)), max(0.0, float(p_away))]
    total = sum(values)
    if total <= 0.0:
        raise ValueError("probabilities sum to zero after clamping")
    return values[0] / total, values[1] / total, values[2] / total


def fit_lambdas_from_1x2_and_total(
    p_home: float,
    p_draw: float,
    p_away: float,
    lambda_total: float,
    min_share: float = 0.20,
    max_share: float = 0.80,
    step: float = 0.0025,
    max_goals: int = 10,
) -> dict:
    if float(lambda_total) <= 0.0:
        raise ValueError("lambda_total must be positive")
    if float(step) <= 0.0:
        raise ValueError("step must be positive")
    if float(min_share) > float(max_share):
        raise ValueError("min_share must be <= max_share")

    target_home, target_draw, target_away = normalize_probs(p_home, p_draw, p_away)

    best_result = None
    share = float(min_share)
    max_share = float(max_share)
    lambda_total = float(lambda_total)

    # Simple grid search is sufficient for this MVP and keeps dependencies at zero.
    while share <= max_share + 1e-12:
        lambda_home = lambda_total * share
        lambda_away = lambda_total * (1.0 - share)
        fitted = poisson_outcome_probs_from_lambdas(
            lambda_home=lambda_home,
            lambda_away=lambda_away,
            max_goals=max_goals,
        )
        error = (
            (fitted["home_win_prob"] - target_home) ** 2
            + (fitted["draw_prob"] - target_draw) ** 2
            + (fitted["away_win_prob"] - target_away) ** 2
        )

        candidate = {
            "lambda_home": float(lambda_home),
            "lambda_away": float(lambda_away),
            "lambda_total": float(lambda_total),
            "target_home_prob": float(target_home),
            "target_draw_prob": float(target_draw),
            "target_away_prob": float(target_away),
            "fit_home_prob": float(fitted["home_win_prob"]),
            "fit_draw_prob": float(fitted["draw_prob"]),
            "fit_away_prob": float(fitted["away_win_prob"]),
            "fit_error": float(error),
            "share_home": float(share),
        }
        if best_result is None or candidate["fit_error"] < best_result["fit_error"]:
            best_result = candidate
        share += float(step)

    if best_result is None:
        raise ValueError("failed to fit lambdas")
    return best_result


if __name__ == "__main__":
    examples = [
        {
            "label": "A",
            "args": {
                "p_home": 0.50,
                "p_draw": 0.26,
                "p_away": 0.24,
                "lambda_total": 2.80,
            },
        },
        {
            "label": "B",
            "args": {
                "p_home": 0.38,
                "p_draw": 0.29,
                "p_away": 0.33,
                "lambda_total": 2.20,
            },
        },
        {
            "label": "C",
            "args": {
                "p_home": 0.62,
                "p_draw": 0.22,
                "p_away": 0.16,
                "lambda_total": 3.05,
            },
        },
    ]

    for example in examples:
        print(example["label"])
        print(fit_lambdas_from_1x2_and_total(**example["args"]))
