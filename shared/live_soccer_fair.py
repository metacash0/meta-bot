from __future__ import annotations

import math


FINISHED_STATUSES = {"FT", "AET", "PEN", "ENDED", "FINISHED"}


def poisson_pmf(k: int, lam: float) -> float:
    k = max(0, int(k))
    lam = max(0.0, float(lam))
    if lam == 0.0:
        return 1.0 if k == 0 else 0.0

    prob = math.exp(-lam)
    for i in range(1, k + 1):
        prob *= lam / float(i)
    return max(0.0, float(prob))


def poisson_outcome_probs(
    score_home: int,
    score_away: int,
    lam_home_rem: float,
    lam_away_rem: float,
    max_future_goals: int = 8,
) -> dict:
    score_home = int(score_home)
    score_away = int(score_away)
    lam_home_rem = max(0.0, float(lam_home_rem))
    lam_away_rem = max(0.0, float(lam_away_rem))
    max_future_goals = max(0, int(max_future_goals))

    home_win_prob = 0.0
    draw_prob = 0.0
    away_win_prob = 0.0

    for home_goals in range(max_future_goals + 1):
        p_home = poisson_pmf(home_goals, lam_home_rem)
        for away_goals in range(max_future_goals + 1):
            p_away = poisson_pmf(away_goals, lam_away_rem)
            joint = max(0.0, p_home * p_away)
            final_home = score_home + home_goals
            final_away = score_away + away_goals
            if final_home > final_away:
                home_win_prob += joint
            elif final_home < final_away:
                away_win_prob += joint
            else:
                draw_prob += joint

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


def adjust_lambdas_for_state(
    lambda_home: float,
    lambda_away: float,
    minute: float,
    score_home: int,
    score_away: int,
    red_home: int,
    red_away: int,
    status: str = "",
) -> tuple[float, float]:
    lambda_home = max(0.0, float(lambda_home))
    lambda_away = max(0.0, float(lambda_away))
    minute = max(0.0, float(minute))
    score_home = int(score_home)
    score_away = int(score_away)
    red_home = max(0, int(red_home))
    red_away = max(0, int(red_away))
    _ = str(status or "")

    remaining_fraction = max(0.0, 90.0 - minute) / 90.0
    lam_home_rem = lambda_home * remaining_fraction
    lam_away_rem = lambda_away * remaining_fraction

    score_diff = score_home - score_away
    if score_diff == 1:
        lam_home_rem *= 0.92
        lam_away_rem *= 1.12
    elif score_diff == -1:
        lam_home_rem *= 1.15
        lam_away_rem *= 0.90
    elif score_diff >= 2:
        lam_home_rem *= 0.85
        lam_away_rem *= 1.18
    elif score_diff <= -2:
        lam_home_rem *= 1.18
        lam_away_rem *= 0.85
    elif score_diff == 0 and minute >= 75.0:
        lam_home_rem *= 0.92
        lam_away_rem *= 0.92

    red_diff = red_home - red_away
    if red_diff > 0:
        for _ in range(red_diff):
            lam_home_rem *= 0.72
            lam_away_rem *= 1.18
    elif red_diff < 0:
        for _ in range(-red_diff):
            lam_home_rem *= 1.18
            lam_away_rem *= 0.72

    return max(0.0, lam_home_rem), max(0.0, lam_away_rem)


def estimate_live_probs(
    lambda_home: float,
    lambda_away: float,
    minute: float,
    score_home: int,
    score_away: int,
    red_home: int,
    red_away: int,
    status: str = "",
) -> dict:
    status_text = str(status or "").strip().upper()
    score_home = int(score_home)
    score_away = int(score_away)

    if status_text in FINISHED_STATUSES:
        if score_home > score_away:
            return {
                "home_win_prob": 1.0,
                "draw_prob": 0.0,
                "away_win_prob": 0.0,
                "lam_home_rem": 0.0,
                "lam_away_rem": 0.0,
            }
        if score_home < score_away:
            return {
                "home_win_prob": 0.0,
                "draw_prob": 0.0,
                "away_win_prob": 1.0,
                "lam_home_rem": 0.0,
                "lam_away_rem": 0.0,
            }
        return {
            "home_win_prob": 0.0,
            "draw_prob": 1.0,
            "away_win_prob": 0.0,
            "lam_home_rem": 0.0,
            "lam_away_rem": 0.0,
        }

    lam_home_rem, lam_away_rem = adjust_lambdas_for_state(
        lambda_home=lambda_home,
        lambda_away=lambda_away,
        minute=minute,
        score_home=score_home,
        score_away=score_away,
        red_home=red_home,
        red_away=red_away,
        status=status,
    )
    outcome_probs = poisson_outcome_probs(
        score_home=score_home,
        score_away=score_away,
        lam_home_rem=lam_home_rem,
        lam_away_rem=lam_away_rem,
    )
    return {
        "home_win_prob": outcome_probs["home_win_prob"],
        "draw_prob": outcome_probs["draw_prob"],
        "away_win_prob": outcome_probs["away_win_prob"],
        "lam_home_rem": lam_home_rem,
        "lam_away_rem": lam_away_rem,
    }


if __name__ == "__main__":
    examples = [
        {
            "label": "0-0 at minute 0",
            "args": {
                "lambda_home": 1.45,
                "lambda_away": 1.10,
                "minute": 0.0,
                "score_home": 0,
                "score_away": 0,
                "red_home": 0,
                "red_away": 0,
            },
        },
        {
            "label": "1-0 at minute 70",
            "args": {
                "lambda_home": 1.45,
                "lambda_away": 1.10,
                "minute": 70.0,
                "score_home": 1,
                "score_away": 0,
                "red_home": 0,
                "red_away": 0,
            },
        },
        {
            "label": "0-1 with away red card at minute 35",
            "args": {
                "lambda_home": 1.45,
                "lambda_away": 1.10,
                "minute": 35.0,
                "score_home": 0,
                "score_away": 1,
                "red_home": 0,
                "red_away": 1,
            },
        },
        {
            "label": "2-2 at minute 88",
            "args": {
                "lambda_home": 1.45,
                "lambda_away": 1.10,
                "minute": 88.0,
                "score_home": 2,
                "score_away": 2,
                "red_home": 0,
                "red_away": 0,
            },
        },
    ]

    for example in examples:
        print(example["label"])
        print(estimate_live_probs(**example["args"]))
