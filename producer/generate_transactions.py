"""
generate_transactions.py

STAGE 1 of the pipeline: the "Python - Data Generator / Source" box
in the Data Flow lane of the architecture diagram.

WHAT THIS SCRIPT DOES (in plain English):
1. Pretends to be thousands of bank customers making transactions.
2. Writes each transaction as one line of JSON - a simple text format
   that looks like: {"transaction_id": "...", "amount": 42.50, ...}
3. On purpose, makes a small percentage of transactions look suspicious
   (huge amounts, rapid-fire spending, weird locations) so that later,
   when we build the fraud detector, we have real anomalies to catch.

WHY JSON: Kafka (our next stage) moves messages around as raw text.
JSON is the standard, human-readable way to structure that text so
every downstream tool (Spark, Delta Lake) can parse it consistently.

HOW TO RUN THIS:
    pip install -r requirements.txt
    python producer/generate_transactions.py
"""

import json
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from faker import Faker

fake = Faker()

# ---------------------------------------------------------------------------
# CONFIG - tweak these to change how much / what kind of data you generate
# ---------------------------------------------------------------------------
NUM_ACCOUNTS = 200          # how many "customers" exist in our fake bank
TOTAL_TRANSACTIONS = 2000   # how many transactions to generate this run
FRAUD_RATE = 0.03           # 3% of transactions will be intentionally suspicious
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "transactions.jsonl"

MERCHANT_CATEGORIES = [
    "grocery", "electronics", "restaurant", "travel",
    "fuel", "online_retail", "utilities", "entertainment",
]


def build_account_pool(n: int):
    """
    Create a fixed set of fake 'customers' with a home city and a typical
    spending amount. Real fraud detection compares each new transaction
    against a customer's *normal* behavior - so we need that baseline first.
    """
    accounts = []
    for _ in range(n):
        accounts.append({
            "account_id": f"ACC{uuid.uuid4().hex[:8].upper()}",
            "home_city": fake.city(),
            "avg_spend": round(random.uniform(15, 200), 2),  # their "normal" amount
        })
    return accounts


def normal_transaction(account):
    """A transaction that looks like everyday, legitimate spending."""
    amount = round(max(1, random.gauss(account["avg_spend"], account["avg_spend"] * 0.3)), 2)
    return {
        "transaction_id": str(uuid.uuid4()),
        "account_id": account["account_id"],
        "amount": amount,
        "currency": "USD",
        "merchant_category": random.choice(MERCHANT_CATEGORIES),
        "location": account["home_city"],
        "device_id": f"DEV{account['account_id'][-4:]}",  # customer's usual device
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "is_fraud_actual": False,  # our "ground truth" label, for testing later
    }


def fraudulent_transaction(account):
    """
    A transaction with one of a few classic fraud signatures.
    We label it is_fraud_actual=True so that later we can measure how
    many of these our PySpark rules actually catch (precision/recall).
    """
    fraud_type = random.choice(["huge_amount", "foreign_location", "new_device"])
    txn = normal_transaction(account)

    if fraud_type == "huge_amount":
        # Amount wildly outside the customer's normal range
        txn["amount"] = round(account["avg_spend"] * random.uniform(15, 40), 2)
    elif fraud_type == "foreign_location":
        # Sudden transaction far from the customer's home city
        txn["location"] = fake.city() + ", " + fake.country()
    elif fraud_type == "new_device":
        # A device that has never been associated with this account
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
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    accounts = build_account_pool(NUM_ACCOUNTS)

    fraud_count = 0
    with open(OUTPUT_FILE, "w") as f:
        for i, txn in enumerate(generate(TOTAL_TRANSACTIONS, FRAUD_RATE, accounts), start=1):
            f.write(json.dumps(txn) + "\n")
            if txn["is_fraud_actual"]:
                fraud_count += 1

            # Print a live preview to the terminal, like a real-time feed
            tag = "FRAUD" if txn["is_fraud_actual"] else "  ok  "
            print(f"[{i:>4}/{TOTAL_TRANSACTIONS}] {tag} | {txn['account_id']} | "
                  f"${txn['amount']:>9,.2f} | {txn['location']}")

            time.sleep(0.005)  # tiny delay so the terminal output is readable

    print(f"\nDone. Wrote {TOTAL_TRANSACTIONS} transactions to {OUTPUT_FILE}")
    print(f"Injected {fraud_count} fraudulent transactions ({fraud_count/TOTAL_TRANSACTIONS:.1%})")


if __name__ == "__main__":
    main()