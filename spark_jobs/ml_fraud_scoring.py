"""
ml_fraud_scoring.py

STAGE 3c: consumes the enriched Kafka stream and scores each transaction
using YOUR trained XGBoost model + preprocessor from Colab.

HOW THIS WORKS, STEP BY STEP:
1. Loads fraud_preprocessor1.pkl and fraud_xgboost_model1.pkl ONCE, when
   this script starts (not per-transaction - that would be slow).
2. Reads the live Kafka stream, same as before.
3. Spark's Structured Streaming processes data in small "micro-batches"
   (a few transactions at a time, not one by one). For EACH micro-batch,
   foreachBatch() hands us a small Spark DataFrame, which we convert to
   pandas (.toPandas()) - because your model and preprocessor are
   scikit-learn/XGBoost objects, which only understand pandas/numpy,
   not Spark's distributed data format.
4. We recreate the 5 engineered features EXACTLY as your notebook did
   (amount_deviation, velocity_ratio, etc.) - the model expects them.
5. We run preprocessor.transform() then model.predict_proba() - the
   exact same steps your notebook used to evaluate the test set.
6. Anything over our chosen threshold gets printed as a fraud alert.

WHY foreachBatch INSTEAD OF A NORMAL STREAMING WRITE:
Spark's built-in streaming writers (console, Delta, etc.) don't know how
to run arbitrary Python ML code. foreachBatch is the standard escape
hatch: it gives you a plain batch of data and lets you do ANYTHING with
it in regular Python - including calling a non-Spark model.

HOW TO RUN:
    Make sure ml/fraud_preprocessor1.pkl and ml/fraud_xgboost_model1.pkl exist
    docker compose up -d
    python spark_jobs/ml_fraud_scoring.py
    (then, in another terminal) python producer/kafka_producer_ml.py
"""

import joblib
import pandas as pd
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType, BooleanType,
)

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "bank-transactions-ml"

MODEL_DIR = Path(__file__).parent.parent / "ml"
PREPROCESSOR_PATH = MODEL_DIR / "fraud_preprocessor1.pkl"
MODEL_PATH = MODEL_DIR / "fraud_xgboost_model1.pkl"

FRAUD_THRESHOLD = 0.5   # matches the default your notebook's classification_report used

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

transaction_schema = StructType([
    StructField("transaction_id", StringType()),
    StructField("account_id", StringType()),
    StructField("transaction_hour", IntegerType()),
    StructField("transaction_day_of_week", IntegerType()),
    StructField("account_age_days", IntegerType()),
    StructField("previous_chargebacks", IntegerType()),
    StructField("merchant_category", StringType()),
    StructField("transaction_country", StringType()),
    StructField("device_type", StringType()),
    StructField("transaction_type", StringType()),
    StructField("geo_location_region", StringType()),
    StructField("is_international", IntegerType()),
    StructField("is_high_risk_merchant_category", IntegerType()),
    StructField("is_weekend", IntegerType()),
    StructField("customer_total_transactions_30d", IntegerType()),
    StructField("customer_risk_score", DoubleType()),
    StructField("transaction_amount", DoubleType()),
    StructField("avg_transaction_amount_30d_customer", DoubleType()),
    StructField("transaction_velocity_1h", IntegerType()),
    StructField("transaction_velocity_24h", IntegerType()),
    StructField("timestamp", StringType()),
    StructField("is_fraud_actual", BooleanType()),
])


def add_engineered_features(pdf: pd.DataFrame) -> pd.DataFrame:
    """Recreates the exact 5 features engineered in the notebook's
    'FEATURE ENGINEERING' cell - the model was trained expecting these."""
    pdf["amount_deviation"] = pdf["transaction_amount"] / (pdf["avg_transaction_amount_30d_customer"] + 1)
    pdf["velocity_ratio"] = pdf["transaction_velocity_1h"] / (pdf["transaction_velocity_24h"] + 1)
    pdf["high_amount_flag"] = (
        pdf["transaction_amount"] > pdf["avg_transaction_amount_30d_customer"] * 2
    ).astype(int)
    pdf["late_night_flag"] = (
        (pdf["transaction_hour"] >= 0) & (pdf["transaction_hour"] <= 5)
    ).astype(int)
    pdf["weekend_international"] = (
        (pdf["is_weekend"] == 1) & (pdf["is_international"] == 1)
    ).astype(int)
    return pdf


def make_batch_processor(preprocessor, model):
    """Returns the function foreachBatch will call for every micro-batch."""

    def process_batch(spark_df, batch_id):
        pdf = spark_df.toPandas()
        if pdf.empty:
            return
        pdf = add_engineered_features(pdf)

        all_raw_features = CATEGORICAL_FEATURES + BASE_NUMERICAL_FEATURES + [
            "amount_deviation", "velocity_ratio", "high_amount_flag",
            "late_night_flag", "weekend_international",
        ]
        X = pdf[all_raw_features]

        X_processed = preprocessor.transform(X)
        fraud_probabilities = model.predict_proba(X_processed)[:, 1]

        pdf["fraud_probability"] = fraud_probabilities
        pdf["predicted_fraud"] = (fraud_probabilities >= FRAUD_THRESHOLD).astype(int)

        flagged = pdf[pdf["predicted_fraud"] == 1]
        if len(flagged) > 0:
            print(f"\n--- Batch {batch_id}: {len(flagged)} flagged out of {len(pdf)} ---")
            print(flagged[[
                "transaction_id", "account_id", "transaction_amount",
                "merchant_category", "customer_risk_score", "fraud_probability", "is_fraud_actual",
            ]].to_string(index=False))
        else:
            print(f"Batch {batch_id}: {len(pdf)} transactions, none flagged.")

    return process_batch


def main():
    print(f"Loading model and preprocessor from {MODEL_DIR} ...")
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    print("Loaded successfully.\n")

    spark = (
        SparkSession.builder
        .appName("MLFraudScoring")
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    raw_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "latest")
        .load()
    )

    transactions = (
        raw_stream
        .selectExpr("CAST(value AS STRING) as json_str")
        .select(from_json(col("json_str"), transaction_schema).alias("data"))
        .select("data.*")
    )

    query = (
        transactions
        .writeStream
        .foreachBatch(make_batch_processor(preprocessor, model))
        .outputMode("update")
        .start()
    )

    print(f"Listening on Kafka topic '{KAFKA_TOPIC}', scoring with fraud_xgboost_model1.pkl")
    print(f"Fraud threshold: {FRAUD_THRESHOLD}")
    print("Press Ctrl+C to stop.\n")

    query.awaitTermination()


if __name__ == "__main__":
    main()