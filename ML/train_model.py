"""
train_model_kaggle.py

STAGE 3b (part 3) - same comparison approach as train_model.py, but run
against REAL, anonymized credit card transaction data from Kaggle
(ULB Credit Card Fraud Detection dataset) instead of our synthetic data.

WHY THIS IS A MEANINGFUL ADDITION TO THE PROJECT:
Our synthetic data was designed by us, so it's always somewhat "easier"
than reality - we know exactly which features matter because we built
them that way. Real data has no such guarantees: relationships are
messier, and here the features are even anonymized (V1-V28), so we don't
even know what they represent - exactly like a real data scientist
working with a bank's genuinely sensitive customer data would face.

DATASET SPECIFICS:
  284,807 transactions, only 492 fraud (~0.17%) - far more imbalanced
  than our synthetic 8%. Columns:
    Time    - seconds elapsed since the first transaction in the dataset
    V1-V28  - anonymized features (PCA-transformed for privacy)
    Amount  - transaction amount
    Class   - 1 = fraud, 0 = genuine (this is our label)

HOW TO RUN:
    Make sure ml/creditcard.csv exists (downloaded from Kaggle)
    python ml/train_model_kaggle.py
"""

import joblib
import pandas as pd
from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, precision_score, recall_score,
    average_precision_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

DATA_FILE = Path(__file__).parent / "creditcard.csv"
MODEL_FILE = Path(__file__).parent / "kaggle_fraud_model.pkl"
COLUMNS_FILE = Path(__file__).parent / "kaggle_model_columns.pkl"
THRESHOLD_FILE = Path(__file__).parent / "kaggle_model_threshold.pkl"
SCALER_FILE = Path(__file__).parent / "kaggle_scaler.pkl"

LABEL_COLUMN = "Class"


def evaluate(name, model, X_test, y_test, results):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)

    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    pr_auc = average_precision_score(y_test, probabilities)
    roc_auc = roc_auc_score(y_test, probabilities)

    print(f"\n--- {name} ---")
    print(f"Precision: {precision:.2f}   Recall: {recall:.2f}   "
          f"PR-AUC: {pr_auc:.3f}   ROC-AUC: {roc_auc:.3f}")

    results.append({
        "model": name, "precision": precision, "recall": recall,
        "pr_auc": pr_auc, "roc_auc": roc_auc,
    })
    return probabilities


def main():
    print(f"Loading {DATA_FILE} - this is a larger file, may take a moment...")
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} transactions")

    fraud_total = df[LABEL_COLUMN].sum()
    print(f"Fraud: {fraud_total} ({fraud_total/len(df):.3%})  |  "
          f"Genuine: {len(df) - fraud_total} ({1 - fraud_total/len(df):.3%})\n")

    # V1-V28 are already PCA-transformed (roughly standardized already).
    # Amount and Time are NOT - they're on wildly different scales
    # (Amount in dollars, Time in raw seconds), so we standardize them too,
    # otherwise a model could be unfairly dominated by whichever raw column
    # happens to have the largest numbers, regardless of actual importance.
    feature_columns = [c for c in df.columns if c not in (LABEL_COLUMN,)]

    X = df[feature_columns].copy()
    y = df[LABEL_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
    X_test[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])

    fraud_count = sum(y_train)
    genuine_count = len(y_train) - fraud_count
    print(f"Training on {len(X_train)} transactions, testing on {len(X_test)}")
    print(f"Training set fraud ratio: {fraud_count} of {len(y_train)} ({fraud_count/len(y_train):.3%})\n")

    scale_pos_weight = genuine_count / fraud_count

    print("Applying SMOTE (this dataset is large, may take a minute)...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print("Done.\n")

    results = []

    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
    log_reg.fit(X_train, y_train)
    evaluate("Logistic Regression + class_weight", log_reg, X_test, y_test, results)

    rf_smote = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rf_smote.fit(X_train_smote, y_train_smote)
    evaluate("Random Forest + SMOTE", rf_smote, X_test, y_test, results)

    xgb_weighted = XGBClassifier(
        scale_pos_weight=scale_pos_weight, random_state=42, eval_metric="logloss"
    )
    xgb_weighted.fit(X_train, y_train)
    xgb_weighted_probs = evaluate("XGBoost + scale_pos_weight", xgb_weighted, X_test, y_test, results)

    xgb_smote = XGBClassifier(random_state=42, eval_metric="logloss")
    xgb_smote.fit(X_train_smote, y_train_smote)
    evaluate("XGBoost + SMOTE", xgb_smote, X_test, y_test, results)

    print("\n" + "=" * 70)
    print("=== MODEL COMPARISON (sorted by PR-AUC) ===")
    print("=" * 70)
    comparison = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    print(comparison.to_string(index=False))

    best_name = comparison.iloc[0]["model"]
    best_model, best_probs = {
        "Logistic Regression + class_weight": (log_reg, None),
        "Random Forest + SMOTE": (rf_smote, None),
        "XGBoost + scale_pos_weight": (xgb_weighted, xgb_weighted_probs),
        "XGBoost + SMOTE": (xgb_smote, None),
    }[best_name]
    if best_probs is None:
        best_probs = best_model.predict_proba(X_test)[:, 1]

    print(f"\n>>> Best model: {best_name}")
    print("\n=== Threshold trade-off for best model ===")
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'Fraud Caught':>14} {'False Alarms':>14}")
    for t in [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]:
        preds_at_t = (best_probs >= t).astype(int)
        p = precision_score(y_test, preds_at_t, zero_division=0)
        r = recall_score(y_test, preds_at_t, zero_division=0)
        caught = ((preds_at_t == 1) & (y_test == 1)).sum()
        false_alarms = ((preds_at_t == 1) & (y_test == 0)).sum()
        print(f"{t:>10.2f} {p:>10.2f} {r:>10.2f} {caught:>14} {false_alarms:>14}")

    CHOSEN_THRESHOLD = 0.5   # start conservative on real data - adjust after seeing the table
    final_preds = (best_probs >= CHOSEN_THRESHOLD).astype(int)
    print(f"\n=== Final report: {best_name} @ threshold {CHOSEN_THRESHOLD} ===")
    print(classification_report(y_test, final_preds, target_names=["Genuine", "Fraud"]))

    joblib.dump(best_model, MODEL_FILE)
    joblib.dump(feature_columns, COLUMNS_FILE)
    joblib.dump(CHOSEN_THRESHOLD, THRESHOLD_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"\nSaved best model ({best_name}) to {MODEL_FILE}")


if __name__ == "__main__":
    main()