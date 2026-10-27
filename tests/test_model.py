from __future__ import annotations

import pandas as pd
import pytest

from src.dataset import generate_synthetic_dataset
from src.feature_engineering import extract_features
from src.fuzzy_logic import fuzzy_suspicion, membership_degrees
from src.hybrid_model import FuzzyRoughHybridModel, evaluate_dataset
from src.rough_set import RoughSetApproximator


def test_synthetic_dataset_has_thirty_rows_and_both_labels() -> None:
    data = generate_synthetic_dataset()
    assert len(data) == 30
    assert set(data["review_label"]) == {"Genuine", "Fake"}


def test_feature_extraction_handles_empty_and_missing_text() -> None:
    source = pd.DataFrame(
        {
            "review_text": ["", None],
            "rating": [3, None],
            "verified_purchase": [False, None],
        }
    )
    features = extract_features(source)
    assert features["review_length"].tolist() == [0, 0]
    assert features["word_count"].tolist() == [0, 0]
    assert features["repeated_word_ratio"].tolist() == [0.0, 0.0]
    assert features["sentiment_rating_mismatch"].tolist() == [0, 0]


def test_fuzzy_memberships_and_scores_are_bounded() -> None:
    features = extract_features(generate_synthetic_dataset())
    for _, row in features.iterrows():
        score, memberships, _ = fuzzy_suspicion(row.to_dict())
        assert 0 <= score <= 100
        assert all(0 <= degree <= 1 for degree in memberships.values())


def test_rough_lower_approximation_is_subset_of_upper() -> None:
    features = extract_features(generate_synthetic_dataset())
    rough = RoughSetApproximator().fit(features)
    for label in rough.classes:
        assert rough.lower_approximations[label] <= rough.upper_approximations[label]
    for row in features.to_dict(orient="records"):
        approximation = rough.predict(row)
        assert not approximation["lower_fake"] or approximation["upper_fake"]
        assert not approximation["lower_genuine"] or approximation["upper_genuine"]


def test_unseen_equivalence_class_is_explicitly_unknown() -> None:
    features = extract_features(generate_synthetic_dataset())
    rough = RoughSetApproximator().fit(features)
    unseen = {
        "promotional_keyword_count": 99,
        "repeated_word_ratio": 0.99,
        "exclamation_count": 99,
        "sentiment_rating_mismatch": 1,
        "verified_purchase_numeric": 1,
    }
    result = rough.predict(unseen)
    assert result["approximation_status"] == "unknown"
    assert result["rough_membership"] is None
    assert result["group_size"] == 0


def test_hybrid_predictions_have_valid_labels_and_ranges() -> None:
    data = generate_synthetic_dataset()
    model = FuzzyRoughHybridModel().fit(data)
    for result in model.predict(data.drop(columns=["review_label"])):
        assert result["prediction"] in {"Likely Genuine", "Suspicious", "Likely Fake"}
        assert 0 <= result["fuzzy_score"] <= 100
        assert 0 <= result["hybrid_score"] <= 100
        assert result["group_size"] >= 0


def test_inference_does_not_require_review_label() -> None:
    model = FuzzyRoughHybridModel().fit(generate_synthetic_dataset())
    review = generate_synthetic_dataset().drop(columns=["review_label"]).iloc[0].to_dict()
    assert model.predict_one(review)["prediction"] in {"Likely Genuine", "Suspicious", "Likely Fake"}


def test_evaluation_training_and_test_ids_are_disjoint(monkeypatch: pytest.MonkeyPatch) -> None:
    fitted_ids: list[str] = []
    original_fit = FuzzyRoughHybridModel.fit

    def tracking_fit(model: FuzzyRoughHybridModel, training_data: pd.DataFrame) -> FuzzyRoughHybridModel:
        fitted_ids.extend(training_data["review_id"].astype(str).tolist())
        return original_fit(model, training_data)

    monkeypatch.setattr(FuzzyRoughHybridModel, "fit", tracking_fit)
    result = evaluate_dataset(generate_synthetic_dataset())
    assert set(result["train_review_ids"]).isdisjoint(result["test_review_ids"])
    assert set(fitted_ids) == set(result["train_review_ids"])
    assert set(fitted_ids).isdisjoint(result["test_review_ids"])
    assert len(result["test_review_ids"]) == result["test_size"]
    assert sum(sum(row.values()) for row in result["confusion_matrix"].values()) == result["test_size"]


def test_rough_set_rejects_target_as_condition_attribute() -> None:
    with pytest.raises(ValueError, match="target label"):
        RoughSetApproximator(condition_attributes=("review_label",))
