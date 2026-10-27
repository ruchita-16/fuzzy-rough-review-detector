from __future__ import annotations

import pandas as pd
import pytest

from src.dataset import generate_synthetic_dataset
from src.feature_engineering import extract_features
from src.fuzzy_logic import fuzzy_suspicion, membership_degrees
from src.hybrid_model import (
    FuzzyRoughHybridModel,
    _assign_groups_to_folds,
    _duplicate_groups,
    _metrics_for_predictions,
    evaluate_dataset,
)
from src.rough_set import RoughSetApproximator


def test_synthetic_dataset_has_thirty_rows_and_both_labels() -> None:
    data = generate_synthetic_dataset()
    assert len(data) == 30
    assert set(data["review_label"]) == {"Genuine", "Fake"}
    assert list(data.columns) == [
        "review_id",
        "product_name",
        "review_text",
        "rating",
        "verified_purchase",
        "review_label",
    ]


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


def test_feature_extraction_captures_phrase_repetition_and_specificity() -> None:
    source = pd.DataFrame(
        {
            "review_text": ["works very well works very well for 4 hours", "clear sound, mild bass"],
            "rating": [5, 4],
            "verified_purchase": [True, False],
        }
    )
    features = extract_features(source)
    assert features.loc[0, "repeated_phrase_ratio"] > features.loc[1, "repeated_phrase_ratio"]
    assert features.loc[0, "specificity_count"] > features.loc[1, "specificity_count"]


def test_fuzzy_memberships_and_scores_are_bounded() -> None:
    features = extract_features(generate_synthetic_dataset())
    for _, row in features.iterrows():
        score, memberships, _ = fuzzy_suspicion(row.to_dict())
        assert 0 <= score <= 100
        assert all(0 <= degree <= 1 for degree in memberships.values())
        assert score == fuzzy_suspicion(row.to_dict())[0]


def test_short_review_alone_does_not_raise_fuzzy_suspicion() -> None:
    feature = extract_features(
        pd.DataFrame(
            [{"review_text": "Works.", "rating": 5, "verified_purchase": True}]
        )
    ).iloc[0].to_dict()
    score, _, _ = fuzzy_suspicion(feature)
    assert score == 0.0


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
        "promotional_keyword_rate": 0.99,
        "repeated_phrase_ratio": 0.99,
        "sentiment_rating_mismatch": 1,
        "specificity_count": 0,
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
    fitted_id_sets: list[set[str]] = []
    original_fit = FuzzyRoughHybridModel.fit

    def tracking_fit(model: FuzzyRoughHybridModel, training_data: pd.DataFrame) -> FuzzyRoughHybridModel:
        if model.tune_thresholds:
            fitted_id_sets.append(set(training_data["review_id"].astype(str).tolist()))
        return original_fit(model, training_data)

    monkeypatch.setattr(FuzzyRoughHybridModel, "fit", tracking_fit)
    data = generate_synthetic_dataset()
    result = evaluate_dataset(data)
    assert set(result["train_review_ids"]) == fitted_id_sets[0]
    assert set(result["train_review_ids"]).isdisjoint(result["test_review_ids"])
    assert len(result["test_review_ids"]) == result["test_size"]
    assert sum(sum(row.values()) for row in result["confusion_matrix"].values()) == result["test_size"]
    assert result["cross_validation"]["fold_count"] == 5
    assert len(result["cross_validation"]["fold_test_ids"]) == 5
    thresholds = result["decision_thresholds"]
    assert 0 <= thresholds["likely_genuine_below"] < thresholds["likely_fake_at_or_above"] <= 100

    identifier_by_position = data["review_id"].astype(str).tolist()
    duplicate_groups = _duplicate_groups(data)
    holdout_sides = {
        identifier: "test" if identifier in set(result["test_review_ids"]) else "train"
        for identifier in identifier_by_position
    }
    for group in duplicate_groups:
        group_sides = {holdout_sides[identifier_by_position[position]] for position in group}
        assert len(group_sides) == 1

    for fold_index, test_ids in enumerate(result["cross_validation"]["fold_test_ids"]):
        train_ids = result["cross_validation"]["fold_train_ids"][fold_index]
        assert set(train_ids).isdisjoint(test_ids)
        assert fitted_id_sets[fold_index + 1] == set(train_ids)
        test_id_set = set(test_ids)
        sides = {
            identifier: "test" if identifier in test_id_set else "train"
            for identifier in identifier_by_position
        }
        for group in duplicate_groups:
            assert len({sides[identifier_by_position[position]] for position in group}) == 1
    assert len(result["cross_validation"]["out_of_fold"]["actual_labels"]) == len(data)


def test_evaluation_metrics_handle_absent_fake_predictions_and_support() -> None:
    metrics = _metrics_for_predictions(["Genuine"], ["Likely Genuine"])
    assert metrics["metrics"]["Fake"] == {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "support": 0,
    }
    assert metrics["accuracy"] == 1.0
    assert metrics["test_size"] == 1


def test_near_duplicate_texts_stay_in_the_same_fold() -> None:
    data = pd.DataFrame(
        {
            "review_text": [
                "The lid fits my bottle well.",
                "The lid fits my bottle well!",
                "The coffee maker brews slowly.",
                "The lamp is bright for reading.",
            ],
            "review_label": ["Genuine", "Genuine", "Fake", "Fake"],
        }
    )
    folds = _assign_groups_to_folds(data, "review_label", fold_count=2, seed=7)
    folds_containing_pair = [
        fold for fold in folds if 0 in fold or 1 in fold
    ]
    assert len(folds_containing_pair) == 1
    assert {0, 1}.issubset(folds_containing_pair[0])


def test_rough_set_rejects_target_as_condition_attribute() -> None:
    with pytest.raises(ValueError, match="target label"):
        RoughSetApproximator(condition_attributes=("review_label",))
