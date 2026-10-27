"""Streamlit dashboard for the fuzzy-rough educational review detector."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dataset import ensure_dataset, generate_synthetic_dataset
from src.feature_engineering import extract_features
from src.hybrid_model import FuzzyRoughHybridModel, evaluate_dataset


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "reviews.csv"
PAGES = ("Dashboard Overview", "Review Explorer", "Analyze a Review", "Methodology", "Model Evaluation")


def _read_synthetic() -> pd.DataFrame:
    ensure_dataset(DATA_PATH)
    return pd.read_csv(DATA_PATH)


def _normalize_upload(upload: pd.DataFrame) -> pd.DataFrame:
    required = {"review_text", "rating"}
    missing = required - set(upload.columns)
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(sorted(missing))}.")

    normalized = upload.copy()
    normalized["review_text"] = normalized["review_text"].fillna("").astype(str)
    normalized["rating"] = pd.to_numeric(normalized["rating"], errors="coerce")
    if normalized["rating"].isna().any():
        raise ValueError("Every rating must be a number from 1 to 5.")
    if not normalized["rating"].between(1, 5).all():
        raise ValueError("Ratings must be between 1 and 5.")

    if "verified_purchase" not in normalized:
        normalized["verified_purchase"] = False
    valid_true = {"true", "yes", "y", "1"}
    valid_false = {"false", "no", "n", "0", ""}

    def parse_verified(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if pd.isna(value):
            return False
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        normalized_value = str(value).strip().lower()
        if normalized_value in valid_true:
            return True
        if normalized_value in valid_false:
            return False
        raise ValueError("verified_purchase values must be true/false, yes/no, or 1/0.")

    normalized["verified_purchase"] = normalized["verified_purchase"].map(parse_verified)
    if "review_label" in normalized:
        normalized["review_label"] = normalized["review_label"].astype(str).str.strip()
        if not normalized["review_label"].isin({"Genuine", "Fake"}).all():
            raise ValueError("review_label, when supplied, must contain only Genuine or Fake.")
    if "review_id" not in normalized:
        normalized["review_id"] = [f"Upload {index + 1}" for index in range(len(normalized))]
    if "product_name" not in normalized:
        normalized["product_name"] = "Uploaded review"
    return normalized


def _model_for(data: pd.DataFrame, fallback: pd.DataFrame) -> FuzzyRoughHybridModel:
    training = data if "review_label" in data.columns else fallback
    return FuzzyRoughHybridModel().fit(training)


def _analyze_frame(data: pd.DataFrame, model: FuzzyRoughHybridModel) -> pd.DataFrame:
    results = model.predict(data)
    summary = pd.DataFrame(
        [
            {
                "prediction": result["prediction"],
                "fuzzy_score": result["fuzzy_score"],
                "rough_membership": result["rough_membership"],
                "approximation_status": result["approximation_status"],
                "group_size": result["group_size"],
                "hybrid_score": result["hybrid_score"],
            }
            for result in results
        ],
        index=data.index,
    )
    return pd.concat([data.copy(), summary], axis=1)


def _render_dashboard(data: pd.DataFrame, analyzed: pd.DataFrame) -> None:
    st.title("Fuzzy-Rough Product Review Detector")
    st.caption("An interpretable soft-computing prototype for exploring review signals—not a fraud verdict.")
    labels_available = "review_label" in data.columns
    genuine_count = int((data["review_label"] == "Genuine").sum()) if labels_available else 0
    fake_count = int((data["review_label"] == "Fake").sum()) if labels_available else 0
    flagged_count = int(analyzed["prediction"].isin(["Suspicious", "Likely Fake"]).sum())
    columns = st.columns(4)
    columns[0].metric("Reviews", len(data))
    columns[1].metric("Genuine examples", genuine_count if labels_available else "—")
    columns[2].metric("Fake examples", fake_count if labels_available else "—")
    columns[3].metric("Flagged for review", flagged_count)

    left, right = st.columns(2)
    with left:
        counts = analyzed["prediction"].value_counts().reindex(
            ["Likely Genuine", "Suspicious", "Likely Fake"], fill_value=0
        )
        figure = px.pie(
            values=counts.values,
            names=counts.index,
            title="Prediction distribution",
            color=counts.index,
            color_discrete_map={
                "Likely Genuine": "#2a9d8f",
                "Suspicious": "#e9c46a",
                "Likely Fake": "#e76f51",
            },
            hole=0.4,
        )
        st.plotly_chart(figure, use_container_width=True)
    with right:
        rating_counts = data["rating"].value_counts().sort_index()
        figure = px.bar(
            x=rating_counts.index,
            y=rating_counts.values,
            labels={"x": "Rating", "y": "Reviews"},
            title="Rating distribution",
            color=rating_counts.index,
            color_continuous_scale="Teal",
        )
        st.plotly_chart(figure, use_container_width=True)

    figure = px.histogram(
        analyzed,
        x="fuzzy_score",
        nbins=12,
        title="Fuzzy suspicion scores",
        labels={"fuzzy_score": "Fuzzy suspicion score (0-100)", "count": "Reviews"},
        color_discrete_sequence=["#457b9d"],
    )
    st.plotly_chart(figure, use_container_width=True)
    if not labels_available:
        st.info("This upload has no labels. Synthetic class counts are not applicable to uploaded rows.")


def _render_explorer(data: pd.DataFrame, analyzed: pd.DataFrame, model: FuzzyRoughHybridModel) -> None:
    st.title("Review Explorer")
    search = st.text_input("Search review text or product")
    product_values = sorted(data["product_name"].fillna("Unknown").astype(str).unique())
    products = st.multiselect("Product", product_values, default=product_values)
    ratings = st.multiselect("Rating", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
    verified = st.selectbox("Verified purchase", ["All", "Verified", "Unverified"])

    filtered = analyzed[
        analyzed["product_name"].fillna("Unknown").astype(str).isin(products)
        & analyzed["rating"].isin(ratings)
    ]
    if verified == "Verified":
        filtered = filtered[filtered["verified_purchase"].map(bool)]
    elif verified == "Unverified":
        filtered = filtered[~filtered["verified_purchase"].map(bool)]
    if search.strip():
        query = search.strip().lower()
        matches = (
            filtered["review_text"].fillna("").astype(str).str.lower().str.contains(query, regex=False)
            | filtered["product_name"].fillna("").astype(str).str.lower().str.contains(query, regex=False)
        )
        filtered = filtered[matches]
    if "review_label" in filtered.columns:
        labels = ["All"] + sorted(filtered["review_label"].dropna().unique().tolist())
        chosen_label = st.selectbox("Original dataset label", labels)
        if chosen_label != "All":
            filtered = filtered[filtered["review_label"] == chosen_label]
    st.dataframe(
        filtered[
            [
                column
                for column in (
                    "review_id", "product_name", "review_text", "rating", "verified_purchase",
                    "review_label", "prediction", "hybrid_score", "fuzzy_score",
                    "approximation_status", "group_size",
                )
                if column in filtered.columns
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
    if not filtered.empty:
        selected_id = st.selectbox("Inspect a review", filtered["review_id"].astype(str).tolist())
        selected = filtered[filtered["review_id"].astype(str) == selected_id].iloc[0]
        result = model.predict_one(selected.to_dict())
        st.subheader(f"Explanation — {selected_id}")
        st.write(f"**Prediction:** {result['prediction']} · **Hybrid score:** {result['hybrid_score']:.1f}/100")
        st.write(result["uncertainty"])
        st.write(
            {
                "fuzzy_score": result["fuzzy_score"],
                "rough_membership_fake": result["rough_membership"],
                "approximation_status": result["approximation_status"],
                "matching_group_size": result["group_size"],
            }
        )
        st.dataframe(pd.DataFrame([result["features"]]), use_container_width=True, hide_index=True)


def _render_analyzer(model: FuzzyRoughHybridModel) -> None:
    st.title("Analyze a Review")
    st.caption("The prediction is an estimate based on review characteristics; it does not establish the review's true label.")
    with st.form("custom_review_form"):
        text = st.text_area("Review text", height=160, placeholder="Write or paste a product review...")
        rating = st.select_slider("Rating", options=[1, 2, 3, 4, 5], value=3)
        verified = st.selectbox("Verified purchase?", ["No", "Yes"])
        submitted = st.form_submit_button("Analyze review", type="primary")
    if submitted:
        review = {
            "review_text": text,
            "rating": rating,
            "verified_purchase": verified == "Yes",
            "review_id": "Custom review",
            "product_name": "Custom review",
        }
        result = model.predict_one(review)
        st.subheader(result["prediction"])
        st.progress(result["hybrid_score"] / 100)
        st.metric("Hybrid suspicion score", f"{result['hybrid_score']:.1f} / 100")
        metrics = st.columns(4)
        metrics[0].metric("Fuzzy score", f"{result['fuzzy_score']:.1f}")
        rough_value = (
            "Unknown" if result["rough_membership"] is None else f"{result['rough_membership']:.2f}"
        )
        metrics[1].metric("Rough membership (Fake)", rough_value)
        metrics[2].metric("Approximation", result["approximation_status"])
        metrics[3].metric("Matching group size", result["group_size"])
        st.info(result["uncertainty"])
        st.subheader("What contributed")
        rule_rows = [
            {"rule": name.replace("_", " ").title(), "activation": round(value, 3)}
            for name, value in result["rule_activations"].items()
            if value > 0
        ]
        if rule_rows:
            st.dataframe(pd.DataFrame(rule_rows), use_container_width=True, hide_index=True)
        else:
            st.write("No suspicion-increasing fuzzy rules activated.")
        st.caption(
            "A custom review has no known ground-truth label here. Rough membership describes only the matching "
            "training group, not the probability that this specific review is fraudulent."
        )


def _render_methodology(model: FuzzyRoughHybridModel, synthetic: pd.DataFrame) -> None:
    st.title("Methodology")
    st.markdown(
        """
        **Fuzzy logic** represents gradual concepts with degrees from 0 to 1 rather than hard boundaries.
        Review length, promotional language, repeated words, punctuation intensity, and rating-text mismatch
        are fuzzified into interpretable sets. The rules combine suspicious signals; a short review by itself
        adds no suspicion. A missing verified purchase adds only a small rule weight.

        **Rough-set theory** groups reviews with identical discretized condition attributes. A group wholly
        contained in Fake belongs to Fake's lower approximation; a group containing at least one Fake review
        belongs to its upper approximation. Mixed groups lie on the boundary. The rough membership is the
        fraction of Fake examples in the matching group. An unseen group is returned as `unknown`.

        **Hybrid score:** fuzzy suspicion is always retained. For matching groups of at least three examples,
        the score is 70% fuzzy score + 30% (100 × rough Fake membership). For smaller known groups, rough
        weight falls to 10%; for unknown groups it is 0%. Categories use fixed score bands: below 35 is
        Likely Genuine, 35–64.99 is Suspicious, and 65 or above is Likely Fake. These are decision bands,
        not calibrated probabilities.
        """
    )
    st.subheader("Worked example")
    example = synthetic.iloc[0].to_dict()
    result = model.predict_one(example)
    st.write(
        f"Review {example.get('review_id', 'example')} has fuzzy score {result['fuzzy_score']:.1f}. "
        f"Its matching group has {result['group_size']} training example(s), rough membership "
        f"{result['rough_membership']!r}, and status `{result['approximation_status']}`. "
        f"The resulting hybrid score is {result['hybrid_score']:.1f}."
    )
    st.subheader("Limitations")
    st.warning(
        "The 30 examples are synthetic and designed to demonstrate contrasting patterns. They do not prove "
        "real-world detection accuracy. Review text alone cannot establish fraud; context, reviewer history, "
        "product data, and human investigation would be needed."
    )


def _render_evaluation(synthetic: pd.DataFrame) -> None:
    st.title("Model Evaluation")
    st.caption(
        "Deterministic stratified 70/30 holdout. Every rough-set reference table is fitted only on its training "
        "fold; test labels are used only after prediction for scoring."
    )
    result = evaluate_dataset(synthetic, test_fraction=0.3, seed=42)
    st.warning("These educational results on synthetic examples are not representative of real-world performance.")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Accuracy", f"{result['accuracy']:.1%}")
    for index, label in enumerate(("Genuine", "Fake"), start=1):
        metric_columns[index].metric(f"{label} precision", f"{result['metrics'][label]['precision']:.1%}")
    metric_columns[3].metric("Test rows", result["test_size"])

    matrix = result["confusion_matrix"]
    figure = go.Figure(
        data=go.Heatmap(
            z=[
                [matrix[row][column] for column in ("Likely Genuine", "Suspicious", "Likely Fake")]
                for row in ("Genuine", "Fake")
            ],
            x=["Likely Genuine", "Suspicious", "Likely Fake"],
            y=["Actual Genuine", "Actual Fake"],
            colorscale="Teal",
            text=[
                [matrix[row][column] for column in ("Likely Genuine", "Suspicious", "Likely Fake")]
                for row in ("Genuine", "Fake")
            ],
            texttemplate="%{text}",
            showscale=False,
        )
    )
    figure.update_layout(title="Confusion matrix", xaxis_title="Prediction", yaxis_title="Ground truth")
    st.plotly_chart(figure, use_container_width=True)
    rows = []
    for label, metrics in result["metrics"].items():
        rows.append({"class": label, **metrics})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.write(f"**Overall accuracy:** {result['accuracy']:.3f}")


def main() -> None:
    st.set_page_config(
        page_title="Fuzzy-Rough Review Detector",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.8rem; padding-bottom: 2rem; max-width: 1280px;}
        [data-testid="stMetric"] {background: #f4f7f9; padding: 14px; border-radius: 10px;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    synthetic = _read_synthetic()
    if "active_reviews" not in st.session_state:
        st.session_state.active_reviews = synthetic.copy()
        st.session_state.active_source = "Built-in synthetic dataset"
        st.session_state.upload_widget_version = 0

    with st.sidebar:
        st.title("Review Lab")
        page = st.radio("Navigate", PAGES)
        st.caption(f"Active data: {st.session_state.active_source}")
        upload = st.file_uploader(
            "Upload review CSV",
            type=["csv"],
            key=f"review_upload_{st.session_state.upload_widget_version}",
            help="Required: review_text and rating. Optional: verified_purchase and review_label.",
        )
        reload_clicked = st.button("Reload synthetic dataset", use_container_width=True)

    if reload_clicked:
        st.session_state.active_reviews = generate_synthetic_dataset()
        st.session_state.active_source = "Built-in synthetic dataset"
        st.session_state.upload_widget_version += 1
        st.rerun()
    if upload is not None:
        try:
            active = _normalize_upload(pd.read_csv(upload))
            st.session_state.active_reviews = active
            st.session_state.active_source = f"Uploaded: {upload.name}"
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
            st.sidebar.error(str(error))

    active_data = st.session_state.active_reviews.copy()
    fallback = synthetic
    model = _model_for(active_data, fallback)
    analyzed = _analyze_frame(active_data, model)

    if page == "Dashboard Overview":
        _render_dashboard(active_data, analyzed)
    elif page == "Review Explorer":
        _render_explorer(active_data, analyzed, model)
    elif page == "Analyze a Review":
        _render_analyzer(model)
    elif page == "Methodology":
        _render_methodology(model, synthetic)
    else:
        _render_evaluation(synthetic)


if __name__ == "__main__":
    main()
