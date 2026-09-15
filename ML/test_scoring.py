"""
test_scoring.py

A standalone test - NO Spark, NO Kafka. Just: build one fake transaction,
try to score it with your model, see if it works or throws a clear error.

PURPOSE: to figure out whether our Spark integration is broken, or the
model/preprocessor loading itself is broken - without Spark's confusing,
multi-layered error messages in the way.

HOW TO RUN:
    python ml/test_scoring.py
"""

import joblib
import pandas as pd
from pathlib import Path

MODEL_DIR = Path(__file__).parent
PREPROCESSOR_PATH = MODEL_DIR / "fraud_preprocessor1.pkl"
MODEL_PATH = MODEL_DIR / "fraud_xgboost_model1.pkl"

CATEGORICAL_FEATURES = [
    "merchant_category", "transaction_country", "device_type",
    "transaction_type", "geo_location_region",
]
BASE_NUMERICAL_FEATURES = [
    "transaction_hour", "transaction_day_of_week", "account_age_days",
    "previous_chargebacks", "is_international", "is_high_risk_merchant_category",
    "is_weekend", "customer_total_transactions_30d", "customer_risk_score",
    "transaction_amount", "avg_transaction_amount_30d_customer",
    "transaction_velocity_1h", "transaction_velocity_24h",
]

print("Step 1: Loading preprocessor...")
preprocessor = joblib.load(PREPROCESSOR_PATH)
print("OK\n")

print("Step 2: Loading model...")
model = joblib.load(MODEL_PATH)
print("OK\n")

print("Step 3: Building one fake transaction...")
fake_txn = {
    "transaction_hour": 14,
    "transaction_day_of_week": 2,
    "account_age_days": 400,
    "previous_chargebacks": 0,
    "merchant_category": "grocery",
    "transaction_country": "US",
    "device_type": "mobile",
    "transaction_type": "pos",
    "geo_location_region": "North America",
    "is_international": 0,
    "is_high_risk_merchant_category": 0,
    "is_weekend": 0,
    "customer_total_transactions_30d": 12,
    "customer_risk_score": 25.0,
    "transaction_amount": 45.50,
    "avg_transaction_amount_30d_customer": 60.0,
    "transaction_velocity_1h": 1,
    "transaction_velocity_24h": 3,
}
pdf = pd.DataFrame([fake_txn])
print("OK\n")

print("Step 4: Adding engineered features...")
pdf["amount_deviation"] = pdf["transaction_amount"] / (pdf["avg_transaction_amount_30d_customer"] + 1)
pdf["velocity_ratio"] = pdf["transaction_velocity_1h"] / (pdf["transaction_velocity_24h"] + 1)
pdf["high_amount_flag"] = (pdf["transaction_amount"] > pdf["avg_transaction_amount_30d_customer"] * 2).astype(int)
pdf["late_night_flag"] = ((pdf["transaction_hour"] >= 0) & (pdf["transaction_hour"] <= 5)).astype(int)
pdf["weekend_international"] = ((pdf["is_weekend"] == 1) & (pdf["is_international"] == 1)).astype(int)
print("OK\n")

print("Step 5: Selecting feature columns...")
all_raw_features = CATEGORICAL_FEATURES + BASE_NUMERICAL_FEATURES + [
    "amount_deviation", "velocity_ratio", "high_amount_flag",
    "late_night_flag", "weekend_international",
]
X = pdf[all_raw_features]
print("OK\n")

print("Step 6: Running preprocessor.transform()...")
X_processed = preprocessor.transform(X)
print("OK - shape:", X_processed.shape, "\n")

print("Step 7: Running model.predict_proba()...")
proba = model.predict_proba(X_processed)
print("OK - fraud probability:", proba[0][1])

print("\n=== ALL STEPS PASSED ===")