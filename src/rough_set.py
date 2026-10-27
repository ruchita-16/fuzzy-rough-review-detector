"""Deterministic rough-set approximations over discretized review conditions."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd


CONDITION_ATTRIBUTES = (
    "promotional_keyword_rate",
    "repeated_phrase_ratio",
    "sentiment_rating_mismatch",
    "specificity_count",
)


def _discretize(attribute: str, value: Any) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    number = float(numeric) if pd.notna(numeric) else 0.0
    if attribute == "promotional_keyword_rate":
        return "low" if number < 0.04 else "medium" if number < 0.15 else "high"
    if attribute == "repeated_phrase_ratio":
        return "low" if number < 0.02 else "medium" if number < 0.15 else "high"
    if attribute == "sentiment_rating_mismatch":
        return "high" if number >= 1 else "low"
    if attribute == "specificity_count":
        return "low" if number == 0 else "high"
    raise ValueError(f"Unsupported rough-set condition attribute: {attribute}")


class RoughSetApproximator:
    """Compute Pawlak approximations for labeled examples and query groups."""

    def __init__(self, condition_attributes: tuple[str, ...] = CONDITION_ATTRIBUTES) -> None:
        if "review_label" in condition_attributes:
            raise ValueError("The target label cannot be a rough-set condition attribute.")
        self.condition_attributes = condition_attributes
        self.reference_groups: dict[tuple[str, ...], list[tuple[int, str]]] = {}
        self.lower_approximations: dict[str, set[int]] = {}
        self.upper_approximations: dict[str, set[int]] = {}
        self.classes: tuple[str, ...] = ()
        self._fitted = False

    def _key(self, row: dict[str, Any]) -> tuple[str, ...]:
        missing = [name for name in self.condition_attributes if name not in row]
        if missing:
            raise ValueError(f"Missing rough-set condition attributes: {', '.join(missing)}")
        return tuple(_discretize(name, row[name]) for name in self.condition_attributes)

    def fit(self, data: pd.DataFrame, label_column: str = "review_label") -> "RoughSetApproximator":
        if label_column not in data.columns:
            raise ValueError(f"Training data must include '{label_column}'.")
        missing = [name for name in self.condition_attributes if name not in data.columns]
        if missing:
            raise ValueError(f"Training data is missing condition attributes: {', '.join(missing)}")
        groups: dict[tuple[str, ...], list[tuple[int, str]]] = defaultdict(list)
        labels: set[str] = set()
        for position, (_, row) in enumerate(data.iterrows()):
            label = str(row[label_column])
            labels.add(label)
            groups[self._key(row.to_dict())].append((position, label))

        self.reference_groups = dict(groups)
        self.classes = tuple(sorted(labels))
        self.lower_approximations = {label: set() for label in self.classes}
        self.upper_approximations = {label: set() for label in self.classes}
        for members in self.reference_groups.values():
            member_positions = {position for position, _ in members}
            member_labels = {label for _, label in members}
            for label in member_labels:
                self.upper_approximations[label].update(member_positions)
            if len(member_labels) == 1:
                label = next(iter(member_labels))
                self.lower_approximations[label].update(member_positions)
        self._fitted = True
        return self

    def predict(self, row: dict[str, Any]) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("Fit the rough-set approximator before prediction.")
        key = self._key(row)
        members = self.reference_groups.get(key)
        if not members:
            return {
                "rough_membership": None,
                "approximation_status": "unknown",
                "group_size": 0,
                "lower_fake": False,
                "upper_fake": False,
                "lower_genuine": False,
                "upper_genuine": False,
            }

        labels = [label for _, label in members]
        fake_count = sum(label == "Fake" for label in labels)
        unique_labels = set(labels)
        status = (
            "lower_fake"
            if unique_labels == {"Fake"}
            else "lower_genuine"
            if unique_labels == {"Genuine"}
            else "boundary"
        )
        return {
            "rough_membership": fake_count / len(labels),
            "approximation_status": status,
            "group_size": len(labels),
            "lower_fake": status == "lower_fake",
            "upper_fake": fake_count > 0,
            "lower_genuine": status == "lower_genuine",
            "upper_genuine": any(label == "Genuine" for label in labels),
        }
