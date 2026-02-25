import math


def compute_risk_score(actual_consumption, expected_consumption, *, min_score=0.3, max_score=0.95, default_score=0.8):
    """Compute a heuristic risk score in [0, 1] for demo alerting and ranking.

    This is a rule-based score (not a calibrated probability) intended for UI/API demonstration.

    Notes:
    - `default_score` is a conservative high-risk fallback used when inputs are invalid or missing.
      In a demo setting, we prefer to surface questionable readings for review rather than silently
      treating them as normal.
    - When `expected_consumption` is extremely small, the raw formula can produce negative scores.
      The `min_score` clamp acts as a low-risk floor so the UI remains stable.
    """
    if expected_consumption is None:
        # Conservative fallback: missing baseline -> return a "review-worthy" risk score.
        return float(default_score)

    try:
        expected = float(expected_consumption)
        actual = float(actual_consumption)
    except (TypeError, ValueError):
        # Conservative fallback for malformed readings.
        return float(default_score)

    if not math.isfinite(expected) or expected <= 0:
        # Conservative fallback: invalid/unreliable baseline.
        return float(default_score)

    if not math.isfinite(actual):
        # Conservative fallback: invalid reading.
        return float(default_score)

    if actual < 0:
        # Conservative fallback: negative consumption is treated as invalid input.
        return float(default_score)

    score = 1 - (actual / expected)
    if score < min_score:
        # Clamp to a low-risk floor to keep the score stable for UI display.
        return float(min_score)
    if score > max_score:
        return float(max_score)
    return float(score)
