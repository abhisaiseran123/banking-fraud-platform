"""
kafka_producer.py

STAGE 2 of the pipeline: sends transactions into Kafka in real time,
instead of writing them to a file.

WHAT CHANGED FROM generate_transactions.py:
Same fake-transaction logic as before (normal_transaction,
fraudulent_transaction, build_account_pool) - only the *destination*
changed. Instead of f.write(json.dumps(txn)), we now do
producer.send("bank-transactions", txn).

WHAT "PRODUCER" MEANS:
In Kafka terms, anything that sends messages INTO a topic is called a
producer. Anything that reads messages OUT of a topic is a consumer.
Right now, this script is the only producer. Later, PySpark will be
our consumer.

HOW TO RUN THIS:
    Make sure Kafka is running first:  docker compose up -d
    Then:                              python producer/kafka_producer.py
"""

import json
import random
import time
import uuid
from datetime import datetime, timezone

from faker import Faker
from kafka import KafkaProducer

fake = Faker()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
NUM_ACCOUNTS = 200
TOTAL_TRANSACTIONS = 2000
FRAUD_RATE = 0.03
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"   # where our Kafka container listens
KAFKA_TOPIC = "bank-transactions"

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
        })
    return accounts


def normal_transaction(account):
    amount = round(max(1, random.gauss(account["avg_spend"], account["avg_spend"] * 0.3)), 2)
    return {
        "transaction_id": str(uuid.uuid4()),
        "account_id": account["account_id"],
        "amount": amount,
        "currency": "USD",
        "merchant_category": random.choice(MERCHANT_CATEGORIES),
        "location": account["home_city"],
        "device_id": f"DEV{account['account_id'][-4:]}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "is_fraud_actual": False,
    }


def fraudulent_transaction(account):
    fraud_type = random.choice(["huge_amount", "foreign_location", "new_device"])
    txn = normal_transaction(account)

    if fraud_type == "huge_amount":
        txn["amount"] = round(account["avg_spend"] * random.uniform(15, 40), 2)
    elif fraud_type == "foreign_location":
        txn["location"] = fake.city() + ", " + fake.country()
    elif fraud_type == "new_device":
        txn["device_id"] = f"DEV{uuid.uuid4().hex[:6].upper()}"

    txn["is_fraud_actual"] = True
    return txn


def generate(num_transactions: int, fraud_rate: float, accounts: list):
    for _ in range(num_transactions):
        account = random.choice(accounts)
        if random.random() < fraud_rate:
            yield fraudulent_transaction(account)
        else:
            yield normal_transaction(account)


def main():
    accounts = build_account_pool(NUM_ACCOUNTS)

    # A KafkaProducer connects to the Kafka broker (our Docker container)
    # and gives us a .send() method to push messages into a topic.
    # value_serializer tells it HOW to convert our Python dict into bytes
    # (Kafka only understands raw bytes, not Python objects directly).
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    fraud_count = 0
    for i, txn in enumerate(generate(TOTAL_TRANSACTIONS, FRAUD_RATE, accounts), start=1):
        # key=account_id ensures all transactions for the same account
        # land on the same Kafka partition, keeping their order consistent -
        # important later for detecting "rapid transactions from one account."
        producer.send(KAFKA_TOPIC, key=txn["account_id"].encode("utf-8"), value=txn)

        if txn["is_fraud_actual"]:
            fraud_count += 1

        tag = "FRAUD" if txn["is_fraud_actual"] else "  ok  "
        print(f"[{i:>4}/{TOTAL_TRANSACTIONS}] {tag} | {txn['account_id']} | "
              f"${txn['amount']:>9,.2f} | {txn['location']}")

        time.sleep(0.05)  # slower than before - lets you watch it stream live

    producer.flush()   # make sure every message is actually sent before exiting
    producer.close()

    print(f"\nDone. Sent {TOTAL_TRANSACTIONS} transactions to Kafka topic '{KAFKA_TOPIC}'")
    print(f"Injected {fraud_count} fraudulent transactions ({fraud_count/TOTAL_TRANSACTIONS:.1%})")


if __name__ == "__main__":
    main()