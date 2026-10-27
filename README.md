# Fuzzy-Rough Set Hybrid for Detecting Fake/Manipulated Product Reviews

An interactive, local Streamlit project that demonstrates how fuzzy logic and rough-set theory can be combined to flag review patterns as **Likely Genuine**, **Suspicious**, or **Likely Fake**. It uses transparent rules and a fixed 30-row synthetic dataset; it does not call an API or train a conventional machine-learning classifier.

> **Educational prototype only.** The data is synthetic and cannot establish real-world detection accuracy. Review characteristics alone cannot prove that a review is fraudulent.

## Objectives and soft-computing concepts

- Turn observable review characteristics into understandable numeric features.
- Use fuzzy membership degrees to represent gradual concepts such as “high promotion” or “short review.”
- Use Pawlak rough-set equivalence classes to reason from examples with matching categorical conditions.
- Combine both forms of evidence while exposing the score, group size, approximation status, and uncertainty.
- Provide an interactive explorer, custom-review analyzer, and leakage-safe holdout evaluation.

## Algorithm and workflow

1. Read the local CSV or generate it if it is missing. The built-in dataset has exactly 30 synthetic rows, multiple product types, both labels, verified and unverified purchases, and deliberately varied review styles.
2. Extract review length, word count, exclamation count, uppercase ratio, promotional keyword count, repeated-word ratio, sentiment-rating mismatch, and numeric verification status. The sentiment estimate uses small transparent positive/negative word lists.
3. Fuzzify five characteristics into linguistic sets. The fuzzy rules increase suspicion for promotion, repetition, rating mismatch, excessive punctuation, and combinations of these signals. A short review alone does not increase suspicion. Verification contributes only a small amount.
4. Discretize five selected features into categorical conditions. Rows equal on all conditions form an equivalence class; the target label is never a condition attribute.
5. Estimate rough membership as the fraction of Fake examples in a matching group. The hybrid suspicion score is the fuzzy score plus supported rough-set evidence. Rough-set weight is 30% for groups of at least three, 10% for smaller known groups, and 0% for unseen groups.
6. Map the score to fixed, interpretable bands: below 35 = Likely Genuine; 35 to less than 65 = Suspicious; 65 or higher = Likely Fake. These scores are not calibrated probabilities.

For each target class, the **lower approximation** contains rows in equivalence classes wholly belonging to that class. The **upper approximation** contains rows whose equivalence classes include at least one member of that class. Mixed groups therefore fall on the boundary of both upper approximations and neither lower approximation. Unseen groups are reported as `unknown`.

## Project structure

```text
fuzzy_rough_review_detector/
|-- app.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- data/
|   `-- reviews.csv
|-- src/
|   |-- __init__.py
|   |-- dataset.py
|   |-- feature_engineering.py
|   |-- fuzzy_logic.py
|   |-- rough_set.py
|   `-- hybrid_model.py
`-- tests/
    `-- test_model.py
```

## Installation and execution (Windows PowerShell)

From the project directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest
python -m streamlit run app.py
```

If PowerShell blocks activation, use the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app and tests require no API keys or external services. `data/reviews.csv` is created from the built-in synthetic data only when it does not already exist. **Reload synthetic dataset** resets the active dashboard data without overwriting an existing CSV. Uploaded CSV data is held in the app session and is not written to disk.

## Dataset schema

| Column | Meaning |
|---|---|
| `review_id` | Synthetic row identifier |
| `product_name` | Product category/name |
| `review_text` | Review text |
| `rating` | Integer rating from 1 to 5 |
| `verified_purchase` | Boolean purchase indicator |
| `review_label` | Demonstration target: `Genuine` or `Fake` |

The built-in labels are illustrative annotations, not ground truth from a real marketplace. An uploaded file requires `review_text` and `rating`; `verified_purchase` is optional and defaults to false. An optional `review_label` must contain only `Genuine` or `Fake`. Ratings must be numeric and within 1–5. Unlabeled uploads use the built-in labeled data as the rough reference table.

## Fuzzy membership sets and rules

The implementation provides reusable triangular and trapezoidal membership functions. It fuzzifies:

- Review length as short, medium, and long.
- Promotional keyword count, repeated-word ratio, and punctuation intensity as low, medium, and high.
- Sentiment-rating mismatch as low or high.

The boundary points are explicit in `src/fuzzy_logic.py`. Rule activations are combined as a weighted average on a 0–100 scale. The strongest contributions come from high promotional language, repetition, their conjunction, mismatch, and the conjunction of promotion with excessive exclamation punctuation. Shortness only contributes when paired with high promotion. Unverified status has the smallest rule weight and cannot determine the classification by itself.

## Evaluation methodology

The evaluation uses a deterministic, seeded, stratified 70/30 holdout. The test labels are accessed only after model fitting and prediction for metric calculation. The rough-set equivalence table is fitted on training rows only; condition discretization uses documented fixed cutoffs rather than all-data statistics.

The confusion matrix reports actual Genuine/Fake rows against all three prediction categories, including Suspicious. Per-class precision, recall, and F1 use Likely Genuine or Likely Fake as the corresponding positive prediction; Suspicious predictions count as errors for these two-class metrics. Accuracy counts only exact Likely Genuine/Genuine and Likely Fake/Fake matches. Missing test classes are handled with zero-valued metrics rather than errors. Results are computed at runtime and must not be interpreted as real-world performance.

## Limitations and future improvements

- Thirty synthetic examples are far too few and artificial for credible performance claims.
- Lexicon sentiment, promotion keywords, and fixed discretization can miss context, sarcasm, language variation, and evolving spam strategies.
- A rough membership value is a group proportion, not a calibrated probability for an individual review.
- Review text and purchase status cannot prove user intent or fraud; human review and contextual evidence are needed.
- Future study could use a consented, independently labeled dataset, stronger annotation protocols, multiple languages, temporal and reviewer-level context, and calibration/fairness analysis while retaining interpretable baselines.

## Viva questions (short answers)

1. **What is fuzzy logic?** It models partial membership in concepts, with degrees between zero and one.
2. **Why use fuzzy sets for review features?** Suspicious language is often gradual rather than separated by one natural cutoff.
3. **What is an indiscernibility relation?** Two examples are indiscernible when they agree on every selected condition attribute.
4. **What is a lower approximation?** The examples whose entire equivalence class belongs to the target class.
5. **What is an upper approximation?** The examples whose equivalence class contains at least one target-class member.
6. **What does a boundary region mean?** Its equivalence classes contain examples from multiple target classes.
7. **How is rough membership calculated?** Fake examples in a matching group divided by the group's total examples.
8. **What happens when a group is unseen?** The rough result is explicitly unknown and is not treated as Genuine.
9. **How is the hybrid score formed?** It combines the fuzzy score with rough membership using a group-size-dependent weight.
10. **Can verified purchase prove a review is genuine?** No; it has only a small influence and never decides the output alone.
11. **Does this project prove fake-review detection accuracy?** No; the synthetic data is for educational demonstration only.
12. **How is test-label leakage prevented?** A fresh rough-set table is fitted only on each split's training rows.
