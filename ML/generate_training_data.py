"""
generate_training_data.py

STAGE 3b (part 1) - generates a labeled dataset for training an ML model.

WHAT'S DIFFERENT FROM kafka_producer.py:
That script sends raw transactions into Kafka for the LIVE pipeline.
This script instead writes a CSV file of transactions with ENGINEERED
FEATURES already computed - numeric signals a machine learning model
can actually learn from, rather than raw text fields like 'location'.

WHY FEATURE ENGINEERING MATTERS:
A model can't directly learn from a location string like "Paris, France".
But it CAN learn from a number like is_foreign_location = 1. Turning raw,
messy fields into clean numeric signals is most of the real work in
building any ML model - the model itself is often the easy part.

FEATURES WE ENGINEER PER TRANSACTION:
  amount                  - the raw transaction amount
  amount_to_avg_ratio     - how many times bigger than usual this is
                            (a $50 charge is normal for one person,
                            suspicious for another - this ratio captures
                            "unusual FOR THIS ACCOUNT", not just "big")
  is_foreign_location     - 1 if the transaction includes a country
                            (our fake foreign transactions always do)
  is_new_device           - 1 if the device doesn't match the account's
                            usual device pattern
  hour_of_day             - 0-23, sometimes fraud clusters at odd hours
  merchant_category       - what type of purchase this was

LABEL:
  is_fraud_actual         - what we're trying to predict

HOW TO RUN:
    python ml/generate_training_data.py
Produces: ml/training_data.csv
"""

import csv
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from faker import Faker

fake = Faker()

NUM_ACCOUNTS = 300
TOTAL_TRANSACTIONS = 5000
FRAUD_RATE = 0.08   # higher than our live producer - models need enough
                    # fraud examples to actually learn the pattern from
OUTPUT_FILE = Path(__file__).parent / "training_data.csv"

MERCHANT_CATEGORIES = [
    "grocery", "electronics", "restaurant", "travel",
    "fuel", "online_retail", "utilities", "entertainment",
]


def build_account_pool(n: int):
    accounts = []
    for _ in range(n):
        accounts.append({
            "account_id": f"ACC{uuid.uuid4().hex[:8].upper()}",
            "home_city": fake.city(),
            "avg_spend": round(random.uniform(15, 200), 2),
            "usual_device": f"DEV{uuid.uuid4().hex[:6].upper()}",
        })
    return accounts


def random_timestamp():
    # Spread transactions across the last 30 days, all hours of the day -
    # a real bank's transactions aren't clustered in one narrow moment.
    now = datetime.now(timezone.utc)
    delta = timedelta(
        days=random.randint(0, 30),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
    )
    return now - delta


def make_transaction(account, is_fraud: bool):
    ts = random_timestamp()

    amount = round(max(1, random.gauss(account["avg_spend"], account["avg_spend"] * 0.35)), 2)
    location = account["home_city"]
    device_id = account["usual_device"]

    if is_fraud:
        # ~25% of fraud is "stealthy" - it deliberately looks like a normal
        # transaction. This is realistic: some real fraud has no obvious
        # signal at all, and no model can catch what genuinely isn't there.
        # Keeping this in the data means our model's recall will honestly
        # be < 100%, which is what a trustworthy fraud model looks like.
        if random.random() < 0.25:
            fraud_type = "stealthy"
            # amount/location/device all stay normal - no signal injected
        else:
            fraud_type = random.choice(["huge_amount", "foreign_location", "new_device"])
            if fraud_type == "huge_amount":
                # Softer multiplier than before (was 15-40x) - now overlaps
                # with the natural high end of genuine spending instead of
                # being trivially separable.
                amount = round(account["avg_spend"] * random.uniform(3, 12), 2)
            elif fraud_type == "foreign_location":
                # Not every fraud attempt is obviously "foreign" - 85% of the time
                if random.random() < 0.85:
                    location = fake.city() + ", " + fake.country()
            elif fraud_type == "new_device":
                if random.random() < 0.85:
                    device_id = f"DEV{uuid.uuid4().hex[:6].upper()}"
    else:
        fraud_type = None
        # Realistic false-positive-prone genuine behavior: some genuine
        # customers travel (foreign location) or get a new phone (new device)
        # without being fraudulent at all. Without this, "foreign location"
        # and "new device" become perfect fraud signals by construction,
        # which never happens with real data.
        if random.random() < 0.06:
            location = fake.city() + ", " + fake.country()
        if random.random() < 0.06:
            device_id = f"DEV{uuid.uuid4().hex[:6].upper()}"
        # Genuine spending occasionally spikes too (big purchase, rent, etc.)
        if random.random() < 0.04:
            amount = round(account["avg_spend"] * random.uniform(2, 5), 2)

    return {
        "transaction_id": str(uuid.uuid4()),
        "account_id": account["account_id"],
        "amount": amount,
        "amount_to_avg_ratio": round(amount / account["avg_spend"], 3),
        "is_foreign_location": 1 if "," in location else 0,
        "is_new_device": 1 if device_id != account["usual_device"] else 0,
        "hour_of_day": ts.hour,
        "merchant_category": random.choice(MERCHANT_CATEGORIES),
        "is_fraud_actual": 1 if is_fraud else 0,
        "fraud_type": fraud_type if fraud_type else "genuine",  # diagnostic only - NOT a model feature
    }


def main():
    accounts = build_account_pool(NUM_ACCOUNTS)
    rows = []

    for _ in range(TOTAL_TRANSACTIONS):
        account = random.choice(accounts)
        is_fraud = random.random() < FRAUD_RATE
        rows.append(make_transaction(account, is_fraud))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    fraud_count = sum(r["is_fraud_actual"] for r in rows)
    print(f"Wrote {len(rows)} labeled transactions to {OUTPUT_FILE}")
    print(f"Fraud: {fraud_count} ({fraud_count/len(rows):.1%})  |  "
          f"Genuine: {len(rows)-fraud_count} ({1-fraud_count/len(rows):.1%})")


if __name__ == "__main__":
    main()