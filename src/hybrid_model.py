"""Fuzzy-rough hybrid model and leakage-safe deterministic holdout evaluation."""

from __future__ import annotations

import random
from typing import Any

import pandas as pd

from .feature_engineering import extract_features
from .fuzzy_logic import fuzzy_suspicion
from .rough_set import RoughSetApproximator


VALID_PREDICTIONS = ("Likely Genuine", "Suspicious", "Likely Fake")


class FuzzyRoughHybridModel:
    """Combine fuzzy suspicion with supported rough-set class evidence."""

    def __init__(self, minimum_supported_group_size: int = 3) -> None:
        if minimum_supported_group_size < 2:
            raise ValueError("minimum_supported_group_size must be at least 2.")
        self.minimum_supported_group_size = minimum_supported_group_size
        self.rough_set = RoughSetApproximator()
        self._fitted = False

    def fit(self, training_data: pd.DataFrame) -> "FuzzyRoughHybridModel":
        """Fit rough reference groups using labeled training rows only."""
        features = extract_features(training_data)
        self.rough_set.fit(features, label_column="review_label")
        self._fitted = True
        return self

    def predict_one(self, review: dict[str, Any] | pd.Series) -> dict[str, Any]:
        """Predict a review without requiring or reading a review_label input."""
        if not self._fitted:
            raise RuntimeError("Fit the model before prediction.")
        row = dict(review)
        features = extract_features(pd.DataFrame([row])).iloc[0].to_dict()
        fuzzy_score, memberships, activations = fuzzy_suspicion(features)
        rough = self.rough_set.predict(features)

        group_size = rough["group_size"]
        if group_size == 0:
            rough_weight = 0.0
            uncertainty = "No matching training equivalence class; rough evidence was not used."
        elif group_size < self.minimum_supported_group_size:
            rough_weight = 0.1
            uncertainty = "Small matching group; rough evidence has reduced influence."
        else:
            rough_weight = 0.3
            uncertainty = "Rough evidence uses a supported matching group."

        rough_score = rough["rough_membership"]
        final_score = (
            fuzzy_score
            if rough_score is None
            else (1.0 - rough_weight) * fuzzy_score + rough_weight * 100.0 * rough_score
        )
        if final_score < 35:
            prediction = "Likely Genuine"
        elif final_score < 65:
            prediction = "Suspicious"
        else:
            prediction = "Likely Fake"
        return {
            "prediction": prediction,
            "fuzzy_score": round(fuzzy_score, 2),
            "rough_membership": rough_score,
            "approximation_status": rough["approximation_status"],
            "group_size": group_size,
            "rough_weight": rough_weight,
            "hybrid_score": round(min(max(final_score, 0.0), 100.0), 2),
            "uncertainty": uncertainty,
            "memberships": memberships,
            "rule_activations": activations,
            "features": features,
        }

    def predict(self, reviews: pd.DataFrame) -> list[dict[str, Any]]:
        return [self.predict_one(row) for row in reviews.to_dict(orient="records")]


def _stratified_split(
    data: pd.DataFrame, label_column: str, test_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    data = data.reset_index(drop=True)
    rng = random.Random(seed)
    train_positions: list[int] = []
    test_positions: list[int] = []
    for _, group in data.groupby(label_column, sort=True):
        positions = list(group.index)
        rng.shuffle(positions)
        test_count = max(1, round(len(positions) * test_fraction)) if len(positions) > 1 else 0
        test_positions.extend(positions[:test_count])
        train_positions.extend(positions[test_count:])
    if not train_positions or not test_positions:
        raise ValueError("Not enough labeled rows for a stratified train/test split.")
    return data.loc[train_positions].copy(), data.loc[test_positions].copy()


def evaluate_dataset(
    data: pd.DataFrame, test_fraction: float = 0.3, seed: int = 42
) -> dict[str, Any]:
    """Evaluate on a deterministic stratified holdout, fitting only on its training rows."""
    if "review_label" not in data.columns:
        raise ValueError("Evaluation data must include a 'review_label' column.")
    train_data, test_data = _stratified_split(data, "review_label", test_fraction, seed)
    model = FuzzyRoughHybridModel().fit(train_data)
    predictions = model.predict(test_data)
    truth = test_data["review_label"].astype(str).tolist()
    predicted_labels = [result["prediction"] for result in predictions]
    matrix_labels = ("Genuine", "Fake")
    prediction_labels = ("Likely Genuine", "Suspicious", "Likely Fake")
    confusion = {actual: {predicted: 0 for predicted in prediction_labels} for actual in matrix_labels}
    for actual, predicted in zip(truth, predicted_labels):
        if actual in confusion:
            confusion[actual][predicted] += 1

    metrics: dict[str, dict[str, float | int]] = {}
    for label in matrix_labels:
        positive_prediction = "Likely Genuine" if label == "Genuine" else "Likely Fake"
        true_positive = confusion[label][positive_prediction]
        false_positive = sum(
            confusion[other][positive_prediction] for other in matrix_labels if other != label
        )
        false_negative = sum(
            count
            for prediction, count in confusion[label].items()
            if prediction != positive_prediction
        )
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion[label].values()),
        }
    evaluated = sum(sum(values.values()) for values in confusion.values())
    correct = confusion["Genuine"]["Likely Genuine"] + confusion["Fake"]["Likely Fake"]
    return {
        "confusion_matrix": confusion,
        "metrics": metrics,
        "accuracy": correct / evaluated if evaluated else 0.0,
        "test_size": len(test_data),
        "train_review_ids": train_data["review_id"].astype(str).tolist()
        if "review_id" in train_data
        else train_data.index.tolist(),
        "test_review_ids": test_data["review_id"].astype(str).tolist()
        if "review_id" in test_data
        else test_data.index.tolist(),
        "predictions": predictions,
        "actual_labels": truth,
    }
