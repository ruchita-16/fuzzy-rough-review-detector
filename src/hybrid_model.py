"""Fuzzy-rough hybrid model and leakage-safe deterministic holdout evaluation."""

from __future__ import annotations

import random
import re
from collections import Counter
from difflib import SequenceMatcher
from statistics import fmean, pstdev
from typing import Any

import pandas as pd

from .feature_engineering import extract_features
from .fuzzy_logic import fuzzy_suspicion
from .rough_set import RoughSetApproximator


VALID_PREDICTIONS = ("Likely Genuine", "Suspicious", "Likely Fake")


class FuzzyRoughHybridModel:
    """Combine fuzzy suspicion with supported rough-set class evidence."""

    def __init__(
        self,
        minimum_supported_group_size: int = 3,
        tune_thresholds: bool = True,
    ) -> None:
        if minimum_supported_group_size < 2:
            raise ValueError("minimum_supported_group_size must be at least 2.")
        self.minimum_supported_group_size = minimum_supported_group_size
        self.tune_thresholds = tune_thresholds
        self.genuine_threshold = 35.0
        self.fake_threshold = 65.0
        self.rough_set = RoughSetApproximator()
        self._fitted = False

    def fit(self, training_data: pd.DataFrame) -> "FuzzyRoughHybridModel":
        """Fit rough groups and select score cutoffs using training-only CV."""
        features = extract_features(training_data)
        self.rough_set.fit(features, label_column="review_label")
        if self.tune_thresholds:
            self.genuine_threshold, self.fake_threshold = _select_decision_thresholds(
                training_data
            )
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
        if final_score < self.genuine_threshold:
            prediction = "Likely Genuine"
        elif final_score < self.fake_threshold:
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
            "genuine_threshold": self.genuine_threshold,
            "fake_threshold": self.fake_threshold,
            "uncertainty": uncertainty,
            "memberships": memberships,
            "rule_activations": activations,
            "features": features,
        }

    def predict(self, reviews: pd.DataFrame) -> list[dict[str, Any]]:
        return [self.predict_one(row) for row in reviews.to_dict(orient="records")]


def _duplicate_groups(data: pd.DataFrame) -> list[list[int]]:
    """Cluster exact and near-duplicate review text without consulting labels."""
    texts = (
        data["review_text"].fillna("").astype(str).str.lower()
        .map(lambda text: re.sub(r"[^a-z0-9]+", " ", text).strip())
        .tolist()
    )
    parents = list(range(len(texts)))

    def find(position: int) -> int:
        while parents[position] != position:
            parents[position] = parents[parents[position]]
            position = parents[position]
        return position

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for first in range(len(texts)):
        if not texts[first]:
            continue
        for second in range(first + 1, len(texts)):
            if not texts[second]:
                continue
            if SequenceMatcher(None, texts[first], texts[second], autojunk=False).ratio() >= 0.92:
                union(first, second)

    groups: dict[int, list[int]] = {}
    for position in range(len(texts)):
        groups.setdefault(find(position), []).append(position)
    return list(groups.values())


def _assign_groups_to_folds(
    data: pd.DataFrame, label_column: str, fold_count: int, seed: int
) -> list[list[int]]:
    """Greedily balance class counts while keeping similar texts in one fold."""
    labels = sorted(data[label_column].astype(str).unique())
    target_per_fold = {
        label: int((data[label_column].astype(str) == label).sum()) / fold_count
        for label in labels
    }
    target_size = len(data) / fold_count
    grouped_counts = [
        (positions, Counter(data.iloc[positions][label_column].astype(str)))
        for positions in _duplicate_groups(data)
    ]
    rng = random.Random(seed)
    rng.shuffle(grouped_counts)
    grouped_counts.sort(key=lambda item: (len(item[0]), max(item[1].values())), reverse=True)

    fold_positions: list[list[int]] = [[] for _ in range(fold_count)]
    fold_class_counts = [Counter() for _ in range(fold_count)]

    def normalized_error(count: int, target: float) -> float:
        return ((count - target) / max(target, 1.0)) ** 2

    for positions, group_counts in grouped_counts:
        def placement_cost(fold_index: int) -> tuple[float, int, int]:
            class_cost = sum(
                normalized_error(
                    fold_class_counts[fold_index][label] + group_counts[label],
                    target_per_fold[label],
                )
                - normalized_error(
                    fold_class_counts[fold_index][label], target_per_fold[label]
                )
                for label in labels
            )
            size_cost = 0.25 * (
                normalized_error(
                    len(fold_positions[fold_index]) + len(positions), target_size
                )
                - normalized_error(len(fold_positions[fold_index]), target_size)
            )
            return class_cost + size_cost, len(fold_positions[fold_index]), fold_index

        chosen_fold = min(range(fold_count), key=placement_cost)
        fold_positions[chosen_fold].extend(positions)
        fold_class_counts[chosen_fold].update(group_counts)
    return fold_positions


def _stratified_split(
    data: pd.DataFrame, label_column: str, test_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    data = data.reset_index(drop=True)
    class_counts = data[label_column].value_counts()
    fold_count = max(2, round(1 / test_fraction))
    fold_count = min(fold_count, int(class_counts.min()), len(data))
    if fold_count < 2:
        raise ValueError("Not enough examples per class for a stratified holdout split.")
    folds = _assign_groups_to_folds(data, label_column, fold_count, seed)
    test_fold_index = random.Random(seed).randrange(fold_count)
    test_positions = folds[test_fold_index]
    train_positions = [
        position
        for fold_index, fold in enumerate(folds)
        if fold_index != test_fold_index
        for position in fold
    ]
    if not train_positions or not test_positions:
        raise ValueError("Not enough labeled rows for a stratified train/test split.")
    return data.loc[train_positions].copy(), data.loc[test_positions].copy()


def _metrics_for_predictions(
    truth: list[str], predictions: list[str]
) -> dict[str, Any]:
    matrix_labels = ("Genuine", "Fake")
    prediction_labels = ("Likely Genuine", "Suspicious", "Likely Fake")
    confusion = {actual: {predicted: 0 for predicted in prediction_labels} for actual in matrix_labels}
    for actual, predicted in zip(truth, predictions):
        if actual in confusion and predicted in confusion[actual]:
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
    supports = [metrics[label]["support"] for label in matrix_labels]
    total_support = sum(supports)
    macro = {
        metric_name: fmean(float(metrics[label][metric_name]) for label in matrix_labels)
        for metric_name in ("precision", "recall", "f1")
    }
    weighted = {
        metric_name: (
            sum(float(metrics[label][metric_name]) * int(metrics[label]["support"]) for label in matrix_labels)
            / total_support
            if total_support
            else 0.0
        )
        for metric_name in ("precision", "recall", "f1")
    }
    return {
        "confusion_matrix": confusion,
        "metrics": metrics,
        "macro_avg": macro,
        "weighted_avg": weighted,
        "accuracy": correct / evaluated if evaluated else 0.0,
        "test_size": evaluated,
        "actual_labels": truth.copy(),
        "predicted_labels": predictions.copy(),
    }


def _select_decision_thresholds(training_data: pd.DataFrame) -> tuple[float, float]:
    """Choose suspicion bands from inner grouped CV using only training examples."""
    data = training_data.reset_index(drop=True)
    class_counts = data["review_label"].value_counts()
    fold_count = min(4, int(class_counts.min()), len(_duplicate_groups(data)))
    if fold_count < 2:
        return 35.0, 65.0

    inner_folds = _assign_groups_to_folds(data, "review_label", fold_count, seed=73)
    out_of_fold_scores: dict[int, float] = {}
    for validation_positions in inner_folds:
        validation_set = set(validation_positions)
        fit_positions = [
            position for position in range(len(data)) if position not in validation_set
        ]
        inner_model = FuzzyRoughHybridModel(tune_thresholds=False).fit(
            data.iloc[fit_positions]
        )
        for position in validation_positions:
            row = data.iloc[position].to_dict()
            out_of_fold_scores[position] = inner_model.predict_one(row)["hybrid_score"]

    truth = data["review_label"].astype(str).tolist()
    best_thresholds = (35.0, 65.0)
    best_key: tuple[float, float, float, float] | None = None
    thresholds = range(0, 101, 5)
    for genuine_threshold in thresholds:
        for fake_threshold in range(genuine_threshold + 5, 101, 5):
            predictions = [
                "Likely Genuine"
                if out_of_fold_scores[position] < genuine_threshold
                else "Likely Fake"
                if out_of_fold_scores[position] >= fake_threshold
                else "Suspicious"
                for position in range(len(data))
            ]
            metrics = _metrics_for_predictions(truth, predictions)
            key = (
                metrics["accuracy"],
                metrics["macro_avg"]["f1"],
                metrics["metrics"]["Fake"]["precision"],
                -float(fake_threshold),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_thresholds = (float(genuine_threshold), float(fake_threshold))
    return best_thresholds


def evaluate_dataset(
    data: pd.DataFrame, test_fraction: float = 0.3, seed: int = 42, n_splits: int = 5
) -> dict[str, Any]:
    """Return a grouped holdout plus stratified CV, fitting independently per split.

    Near-duplicate review text is kept together in each partition. Thresholds
    remain fixed a priori, so held-out labels do not influence model decisions.
    """
    if "review_label" not in data.columns:
        raise ValueError("Evaluation data must include a 'review_label' column.")
    if data.empty:
        raise ValueError("Evaluation data must include at least one labeled review.")
    unexpected_labels = set(data["review_label"].dropna().astype(str)) - {"Genuine", "Fake"}
    if unexpected_labels or data["review_label"].isna().any():
        raise ValueError("Evaluation labels must be non-missing Genuine or Fake values.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    data = data.reset_index(drop=True)
    train_data, test_data = _stratified_split(data, "review_label", test_fraction, seed)
    model = FuzzyRoughHybridModel().fit(train_data)
    predictions = model.predict(test_data)
    truth = test_data["review_label"].astype(str).tolist()
    predicted_labels = [result["prediction"] for result in predictions]
    holdout = _metrics_for_predictions(truth, predicted_labels)

    minimum_class_count = int(data["review_label"].value_counts().min())
    fold_count = min(n_splits, minimum_class_count, len(_duplicate_groups(data)))
    if fold_count < 2:
        raise ValueError("Not enough independent labeled review groups for cross-validation.")
    fold_positions = _assign_groups_to_folds(data, "review_label", fold_count, seed)
    fold_results: list[dict[str, Any]] = []
    fold_train_ids: list[list[str]] = []
    fold_test_ids: list[list[str]] = []
    out_of_fold_predictions: dict[int, str] = {}
    for test_positions in fold_positions:
        test_position_set = set(test_positions)
        train_positions = [
            position for position in range(len(data)) if position not in test_position_set
        ]
        fold_train = data.iloc[train_positions].copy()
        fold_test = data.iloc[test_positions].copy()
        fold_train_ids.append(
            fold_train["review_id"].astype(str).tolist()
            if "review_id" in fold_train
            else fold_train.index.astype(str).tolist()
        )
        fold_test_ids.append(
            fold_test["review_id"].astype(str).tolist()
            if "review_id" in fold_test
            else fold_test.index.astype(str).tolist()
        )
        fold_model = FuzzyRoughHybridModel().fit(fold_train)
        fold_predictions = [
            result["prediction"] for result in fold_model.predict(fold_test)
        ]
        fold_truth = fold_test["review_label"].astype(str).tolist()
        metrics = _metrics_for_predictions(fold_truth, fold_predictions)
        fold_results.append(metrics)
        out_of_fold_predictions.update(zip(test_positions, fold_predictions))

    out_of_fold_truth = data["review_label"].astype(str).tolist()
    out_of_fold_labels = [out_of_fold_predictions[position] for position in range(len(data))]
    out_of_fold = _metrics_for_predictions(out_of_fold_truth, out_of_fold_labels)
    fold_summary = {
        "accuracy_mean": fmean(fold["accuracy"] for fold in fold_results),
        "accuracy_std": pstdev(fold["accuracy"] for fold in fold_results),
        "macro_f1_mean": fmean(fold["macro_avg"]["f1"] for fold in fold_results),
        "macro_f1_std": pstdev(fold["macro_avg"]["f1"] for fold in fold_results),
        "fake_precision_mean": fmean(fold["metrics"]["Fake"]["precision"] for fold in fold_results),
        "fake_precision_std": pstdev(fold["metrics"]["Fake"]["precision"] for fold in fold_results),
        "fake_recall_mean": fmean(fold["metrics"]["Fake"]["recall"] for fold in fold_results),
        "fake_recall_std": pstdev(fold["metrics"]["Fake"]["recall"] for fold in fold_results),
        "fake_f1_mean": fmean(fold["metrics"]["Fake"]["f1"] for fold in fold_results),
        "fake_f1_std": pstdev(fold["metrics"]["Fake"]["f1"] for fold in fold_results),
    }
    return {
        **holdout,
        "train_review_ids": train_data["review_id"].astype(str).tolist()
        if "review_id" in train_data
        else train_data.index.tolist(),
        "test_review_ids": test_data["review_id"].astype(str).tolist()
        if "review_id" in test_data
        else test_data.index.tolist(),
        "decision_thresholds": {
            "likely_genuine_below": model.genuine_threshold,
            "likely_fake_at_or_above": model.fake_threshold,
        },
        "predictions": predictions,
        "actual_labels": truth,
        "cross_validation": {
            "fold_count": fold_count,
            "summary": fold_summary,
            "fold_metrics": fold_results,
            "fold_train_ids": fold_train_ids,
            "fold_test_ids": fold_test_ids,
            "out_of_fold": out_of_fold,
        },
    }
