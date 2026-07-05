"""
Train and evaluate churn classifiers on the leakage-free feature table from
churn_features.py.

Rigor checklist applied here:
- Stratified train/test split at the user level (each row already IS one user,
  so no risk of the same entity leaking across the split)
- Preprocessing (scaler, one-hot encoder) fit on the TRAIN split only, then
  applied to test -- fitting on the full dataset would leak test-set
  distribution into training
- Imbalance (4.6% churn) handled via class_weight='balanced' rather than
  accuracy-chasing
- Metrics: PR-AUC and macro-F1, not accuracy, since a "predict everyone stays"
  model would already score ~95% accuracy while being useless
"""
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

DATA_PATH = r"c:\Users\ashut\Downloads\VS code projects\Aamzon transactions data\Code files\churn_dataset.csv"

NUMERIC_FEATURES = [
    "n_orders",
    "tenure_days",
    "recency_days",
    "total_spend",
    "avg_order_value",
    "n_categories",
    "avg_days_between_orders",
    "orders_last_90d",
    "orders_prior_90d",
    "momentum",
]
CATEGORICAL_FEATURES = [
    "Q-demos-age",
    "Q-demos-education",
    "Q-demos-income",
    "Q-amazon-use-howmany",
    "Q-amazon-use-hh-size",
    "Q-amazon-use-how-oft",
]


def load_dataset():
    df = pd.read_csv(DATA_PATH, index_col="Survey ResponseID")
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["churn"]
    return X, y


def build_preprocessor():
    # median-based scaling isn't needed for RF but IS needed for logistic regression,
    # which is scale-sensitive; ColumnTransformer lets both models share one pipeline
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def evaluate(name, pipeline, X_test, y_test):
    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = pipeline.predict(X_test)

    pr_auc = average_precision_score(y_test, proba)
    roc_auc = roc_auc_score(y_test, proba)
    macro_f1 = f1_score(y_test, preds, average="macro")

    print(f"\n=== {name} ===")
    print(f"PR-AUC:   {pr_auc:.4f}")
    print(f"ROC-AUC:  {roc_auc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("\nClassification report:\n", classification_report(y_test, preds, target_names=["retained", "churned"]))
    print("Confusion matrix (rows=true, cols=pred):\n", confusion_matrix(y_test, preds))
    return {"model": name, "pr_auc": pr_auc, "roc_auc": roc_auc, "macro_f1": macro_f1}


def main():
    X, y = load_dataset()

    # stratify=y keeps the ~4.6% churn ratio consistent across train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train)} ({y_train.mean():.3f} churn rate)")
    print(f"Test:  {len(X_test)} ({y_test.mean():.3f} churn rate)")

    results = []

    logreg = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    logreg.fit(X_train, y_train)
    results.append(evaluate("Logistic Regression", logreg, X_test, y_test))

    rf = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1
        )),
    ])
    rf.fit(X_train, y_train)
    results.append(evaluate("Random Forest", rf, X_test, y_test))

    # feature importance -- sanity check that the model is leaning on sensible
    # behavioral signals (recency/momentum) rather than something spurious
    feature_names = rf.named_steps["prep"].get_feature_names_out()
    importances = rf.named_steps["clf"].feature_importances_
    top = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)
    print("\n=== Random Forest top 15 feature importances ===")
    print(top)

    print("\n=== Summary ===")
    print(pd.DataFrame(results).to_string(index=False))

    # The train/test split above exists to validate the modeling approach; the
    # deployed model is refit on ALL eligible users so the API serves scores
    # informed by the full labeled set, not just 75% of it.
    deployed_model = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1
        )),
    ])
    deployed_model.fit(X, y)
    joblib.dump(deployed_model, DATA_PATH.replace("churn_dataset.csv", "churn_model.joblib"))
    print(f"\nSaved deployed model to churn_model.joblib (trained on all {len(X)} eligible users)")


if __name__ == "__main__":
    main()
