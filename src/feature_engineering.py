"""Transparent, rule-based feature extraction for product reviews."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import pandas as pd


PROMOTIONAL_KEYWORDS = {
    "amazing", "best", "buy", "deal", "excellent", "flawless", "guaranteed",
    "love", "perfect", "recommend", "today", "winner",
}
POSITIVE_WORDS = {
    "amazing", "best", "bright", "comfortable", "excellent", "good", "great",
    "love", "nice", "perfect", "recommend", "sturdy", "useful",
}
NEGATIVE_WORDS = {
    "awful", "bad", "bitter", "broken", "disappointed", "leak", "messy",
    "terrible", "thin", "uncomfortable", "weak", "worse",
}
SPECIFICITY_WORDS = {
    "cm", "day", "days", "hour", "hours", "inch", "inches",
    "mile", "miles", "minute", "minutes", "month", "months", "week", "weeks",
}
_WORD_PATTERN = re.compile(r"[A-Za-z0-9']+")


def _safe_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _verified_numeric(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if value is None or pd.isna(value):
        return 0
    if isinstance(value, (int, float)):
        return int(value != 0)
    normalized = str(value).strip().lower()
    if normalized in {"true", "yes", "y", "1"}:
        return 1
    if normalized in {"false", "no", "n", "0", ""}:
        return 0
    return 0


def _extract_one(review_text: Any, rating: Any, verified_purchase: Any) -> dict[str, float | int]:
    text = _safe_text(review_text)
    words = [word.lower() for word in _WORD_PATTERN.findall(text)]
    word_count = len(words)
    counts = Counter(words)
    repeated_word_ratio = (
        sum(count - 1 for count in counts.values() if count > 1) / word_count
        if word_count
        else 0.0
    )
    bigrams_and_trigrams = [
        tuple(words[index : index + size])
        for size in (2, 3)
        for index in range(max(word_count - size + 1, 0))
    ]
    phrase_counts = Counter(bigrams_and_trigrams)
    repeated_phrase_ratio = (
        sum(count - 1 for count in phrase_counts.values() if count > 1)
        / len(bigrams_and_trigrams)
        if bigrams_and_trigrams
        else 0.0
    )
    uppercase_letters = sum(character.isupper() for character in text if character.isalpha())
    letters = sum(character.isalpha() for character in text)
    rating_value = pd.to_numeric(pd.Series([rating]), errors="coerce").iloc[0]
    rating_value = float(rating_value) if pd.notna(rating_value) else 3.0

    positive = sum(word in POSITIVE_WORDS for word in words)
    negative = sum(word in NEGATIVE_WORDS for word in words)
    mismatch = int((positive > negative and rating_value <= 2) or (negative > positive and rating_value >= 4))

    return {
        "review_length": len(text),
        "word_count": word_count,
        "exclamation_count": text.count("!"),
        "uppercase_ratio": uppercase_letters / letters if letters else 0.0,
        "promotional_keyword_count": sum(word in PROMOTIONAL_KEYWORDS for word in words),
        "promotional_keyword_rate": (
            sum(word in PROMOTIONAL_KEYWORDS for word in words) / max(word_count, 1)
        ),
        "repeated_word_ratio": repeated_word_ratio,
        "repeated_phrase_ratio": repeated_phrase_ratio,
        "specificity_count": sum(
            word in SPECIFICITY_WORDS or any(character.isdigit() for character in word)
            for word in words
        ),
        "sentiment_score": (positive - negative) / max(word_count, 1),
        "sentiment_rating_mismatch": mismatch,
        "verified_purchase_numeric": _verified_numeric(verified_purchase),
    }


def extract_features(reviews: pd.DataFrame) -> pd.DataFrame:
    """Preserve input columns and append numeric features, tolerating missing values."""
    if "review_text" not in reviews.columns:
        raise ValueError("The input data must include a 'review_text' column.")

    result = reviews.copy()
    if "rating" not in result.columns:
        result["rating"] = 3
    if "verified_purchase" not in result.columns:
        result["verified_purchase"] = False

    extracted = [
        _extract_one(row.review_text, row.rating, row.verified_purchase)
        for row in result[["review_text", "rating", "verified_purchase"]].itertuples(index=False)
    ]
    feature_frame = pd.DataFrame(extracted, index=result.index)
    for column in feature_frame.columns:
        result[column] = feature_frame[column]
    return result
