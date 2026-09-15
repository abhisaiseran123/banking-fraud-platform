"""
fraud_detection_stream.py

STAGE 3 of the pipeline: the "PySpark - Stream Processing & Transformations"
box in the Data Flow lane.

WHAT THIS SCRIPT DOES (plain English):
1. Connects to Kafka as a CONSUMER, reading from the 'bank-transactions'
   topic that kafka_producer.py sends to.
2. Parses each message from raw JSON text into a structured table (Spark's
   native format - like a spreadsheet with typed columns, not just text).
3. Applies two fraud rules:
     RULE 1 - Large amount: flags any single transaction over a threshold.
     RULE 2 - Velocity: counts how many transactions each account makes in
              a rolling 5-minute window; flags accounts over a count threshold.
4. Prints both results live to the terminal.

WHY SPARK (not just plain Python) FOR THIS:
Rule 1 is easy in plain Python. Rule 2 is not - it requires grouping and
counting across a moving time window, continuously, on a live, unbounded
stream of data. That kind of continuous windowed aggregation is exactly
what Spark Structured Streaming is built for.

HOW TO RUN:
    Make sure Kafka is running:      docker compose up -d
    Then run this script:            python spark_jobs/fraud_detection_stream.py
    Then, in ANOTHER terminal, run:  python producer/kafka_producer.py
    Watch transactions and fraud flags appear here as they're produced.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, window, count
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, BooleanType

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "bank-transactions"

# Fraud rule thresholds - tweak these as you experiment
LARGE_AMOUNT_THRESHOLD = 500.00     # flag any transaction over this amount
VELOCITY_WINDOW = "5 minutes"       # rolling time window for counting transactions
VELOCITY_THRESHOLD = 5              # flag if an account has more than this many txns in the window


# This schema describes the shape of the JSON each transaction arrives as.
# Spark needs this defined explicitly - it can't safely guess types from
# a live stream the way it could from a static file.
transaction_schema = StructType([
    StructField("transaction_id", StringType()),
    StructField("account_id", StringType()),
    StructField("amount", DoubleType()),
    StructField("currency", StringType()),
    StructField("merchant_category", StringType()),
    StructField("location", StringType()),
    StructField("device_id", StringType()),
    StructField("timestamp", StringType()),
    StructField("is_fraud_actual", BooleanType()),
])


def main():
    spark = (
        SparkSession.builder
        .appName("FraudDetectionStream")
        # This tells Spark to pull in the Kafka connector library automatically
        # (Spark's core doesn't include Kafka support by default).
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")  # hide noisy INFO logs, keep only warnings/errors

    # --- Read the raw stream from Kafka ---
    # Kafka messages arrive as raw bytes with 'key' and 'value' columns.
    # We only care about 'value' (our JSON transaction), cast to a string.
    raw_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "latest")   # only read NEW messages from when this job starts
        .load()
    )

    # --- Parse the JSON string into structured columns ---
    transactions = (
        raw_stream
        .selectExpr("CAST(value AS STRING) as json_str")
        .select(from_json(col("json_str"), transaction_schema).alias("data"))
        .select("data.*")
        # Spark needs a proper timestamp type (not text) to do time-window
        # aggregations, so we convert the timestamp string here.
        .withColumn("event_time", col("timestamp").cast("timestamp"))
    )

    # --- RULE 1: Large amount ---
    flagged_by_amount = transactions.withColumn(
        "flag_reason",
        # A simple readable rule: if amount is over the threshold, flag it.
        (col("amount") > LARGE_AMOUNT_THRESHOLD).cast("string")
    ).filter(col("amount") > LARGE_AMOUNT_THRESHOLD)

    query_amount = (
        flagged_by_amount
        .select("transaction_id", "account_id", "amount", "location", "is_fraud_actual")
        .writeStream
        .outputMode("append")     # show each new flagged row as it appears
        .format("console")
        .option("truncate", "false")
        .queryName("large_amount_alerts")
        .start()
    )

    # --- RULE 2: Velocity - too many transactions per account in a time window ---
    velocity_counts = (
        transactions
        .withWatermark("event_time", "1 minute")   # tells Spark how late data is allowed to arrive
        .groupBy(
            window(col("event_time"), VELOCITY_WINDOW),
            col("account_id"),
        )
        .agg(count("*").alias("txn_count"))
        .filter(col("txn_count") > VELOCITY_THRESHOLD)
    )

    query_velocity = (
        velocity_counts
        .writeStream
        .outputMode("update")     # show updated counts as they change within the window
        .format("console")
        .option("truncate", "false")
        .queryName("velocity_alerts")
        .start()
    )

    print(f"\nListening on Kafka topic '{KAFKA_TOPIC}'...")
    print(f"Rule 1: flagging transactions over ${LARGE_AMOUNT_THRESHOLD}")
    print(f"Rule 2: flagging accounts with more than {VELOCITY_THRESHOLD} transactions per {VELOCITY_WINDOW}")
    print("Press Ctrl+C to stop.\n")

    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()