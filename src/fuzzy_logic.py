"""Interpretable fuzzy memberships and a small fuzzy suspicion rule system."""

from __future__ import annotations

from typing import Any

import numpy as np


def triangular_membership(value: float, left: float, peak: float, right: float) -> float:
    """Return a triangular membership degree in [0, 1]."""
    value = float(value)
    if left == peak == right:
        return 1.0 if value == peak else 0.0
    if value == peak:
        return 1.0
    if value <= left or value >= right:
        return 0.0
    if value < peak:
        return (value - left) / (peak - left)
    return (right - value) / (right - peak)


def trapezoidal_membership(
    value: float, left: float, plateau_start: float, plateau_end: float, right: float
) -> float:
    """Return a trapezoidal membership degree in [0, 1]."""
    value = float(value)
    if plateau_start > plateau_end or left > plateau_start or plateau_end > right:
        raise ValueError("Trapezoid points must satisfy left <= plateau_start <= plateau_end <= right.")
    if plateau_start <= value <= plateau_end:
        return 1.0
    if value <= left or value >= right:
        return 0.0
    if value < plateau_start:
        return (value - left) / (plateau_start - left) if plateau_start != left else 1.0
    return (right - value) / (right - plateau_end) if right != plateau_end else 1.0


def membership_degrees(features: dict[str, Any]) -> dict[str, float]:
    """Fuzzify review length, promotion, repetition, punctuation, mismatch, and specificity."""
    words = max(float(features.get("word_count", 0) or 0), 0.0)
    length = max(float(features.get("review_length", 0) or 0), 0.0)
    promo_count = max(float(features.get("promotional_keyword_count", 0) or 0), 0.0)
    promo_rate = max(
        float(
            features.get(
                "promotional_keyword_rate",
                promo_count / max(float(features.get("word_count", 0) or 0), 1.0),
            )
            or 0
        ),
        0.0,
    )
    repetition = float(np.clip(float(features.get("repeated_word_ratio", 0) or 0), 0.0, 1.0))
    phrase_repetition = float(
        np.clip(float(features.get("repeated_phrase_ratio", 0) or 0), 0.0, 1.0)
    )
    exclamations = max(float(features.get("exclamation_count", 0) or 0), 0.0)
    punctuation_rate = exclamations / max(words, 1.0)
    mismatch = float(features.get("sentiment_rating_mismatch", 0) or 0)
    specificity = max(float(features.get("specificity_count", 0) or 0), 0.0)

    return {
        "length_short": trapezoidal_membership(length, 0, 0, 35, 75),
        "length_medium": triangular_membership(length, 35, 110, 240),
        "length_long": trapezoidal_membership(length, 180, 300, float("inf"), float("inf")),
        "promotion_low": trapezoidal_membership(promo_rate, 0, 0, 0.03, 0.1),
        "promotion_medium": triangular_membership(promo_rate, 0.02, 0.12, 0.3),
        "promotion_high": trapezoidal_membership(
            promo_rate, 0.08, 0.2, float("inf"), float("inf")
        ),
        "repetition_low": trapezoidal_membership(repetition, 0, 0, 0.03, 0.12),
        "repetition_medium": triangular_membership(repetition, 0.03, 0.18, 0.4),
        "repetition_high": trapezoidal_membership(repetition, 0.2, 0.4, 1, 1),
        "phrase_repetition_low": trapezoidal_membership(
            phrase_repetition, 0, 0, 0.02, 0.1
        ),
        "phrase_repetition_medium": triangular_membership(
            phrase_repetition, 0.02, 0.15, 0.4
        ),
        "phrase_repetition_high": trapezoidal_membership(
            phrase_repetition, 0.1, 0.3, 1, 1
        ),
        "punctuation_low": trapezoidal_membership(punctuation_rate, 0, 0, 0.02, 0.08),
        "punctuation_medium": triangular_membership(punctuation_rate, 0.02, 0.12, 0.35),
        "punctuation_high": trapezoidal_membership(
            punctuation_rate, 0.2, 0.4, float("inf"), float("inf")
        ),
        "mismatch_low": 1.0 - min(max(mismatch, 0.0), 1.0),
        "mismatch_high": min(max(mismatch, 0.0), 1.0),
        "specificity_low": 1.0 if specificity == 0 else 0.0,
        "specificity_high": 1.0 if specificity > 0 else 0.0,
    }


_RULE_WEIGHTS = {
    "high_promotion": 0.7,
    "high_repetition": 0.6,
    "promotion_and_repetition": 0.5,
    "high_mismatch": 1.0,
    "punctuation_and_promotion": 0.45,
    "promotional_but_nonspecific": 0.35,
    "short_and_promotional": 0.15,
    "unverified_purchase": 0.15,
}


def fuzzy_suspicion(features: dict[str, Any]) -> tuple[float, dict[str, float], dict[str, float]]:
    """Return normalized 0-100 suspicion, memberships, and rule activations.

    Short length alone has no suspicion rule. Verification has only a small
    contribution, so it cannot determine the result on its own.
    """
    memberships = membership_degrees(features)
    repetition_activation = max(
        memberships["repetition_high"], memberships["phrase_repetition_high"]
    )
    activations = {
        "high_promotion": memberships["promotion_high"],
        "high_repetition": repetition_activation,
        "promotion_and_repetition": min(
            memberships["promotion_high"], repetition_activation
        ),
        "high_mismatch": memberships["mismatch_high"],
        "punctuation_and_promotion": min(
            memberships["punctuation_high"], memberships["promotion_high"]
        ),
        "promotional_but_nonspecific": min(
            memberships["promotion_high"], memberships["specificity_low"]
        ),
        "short_and_promotional": min(
            memberships["length_short"], memberships["promotion_high"]
        ),
        "unverified_purchase": 1.0 - float(bool(features.get("verified_purchase_numeric", 0))),
    }
    weighted_activation = sum(_RULE_WEIGHTS[name] * activation for name, activation in activations.items())
    score = 100.0 * weighted_activation / sum(_RULE_WEIGHTS.values())
    return float(np.clip(score, 0.0, 100.0)), memberships, activations
