"""
kafka_producer_ml.py

STAGE 3c: an enriched producer that emits ALL 23 raw features your
Colab-trained XGBoost model needs, not just the simple 7 fields our
original producer sends.

WHY THIS IS DIFFERENT FROM kafka_producer.py:
Your model was trained on real historical account data: chargeback
history, account age, rolling 30-day averages, transaction velocity.
A single incoming transaction can't carry that on its own - it requires
KNOWING THE ACCOUNT'S PAST. In a real bank, that history lives in a
database. Here, since WE control the entire simulation, we track it
directly in memory as each account "lives" through this script -
no separate database needed for a learning project like this.

ACCOUNT RISK TIERS:
90% of accounts are "normal" (few/no chargebacks, low risk score).
10% are "risky" (some chargeback history, higher risk score) - and
fraud is deliberately MORE likely (not guaranteed) to come from risky
accounts. This mirrors reality: repeat offenders exist, but so does
occasional fraud from otherwise-clean accounts, and risky accounts
still make mostly-genuine purchases too.

HOW TO RUN:
    docker compose up -d
    python producer/kafka_producer_ml.py
"""

import json
import random
import time
import uuid
from collections import deque
from datetime import datetime, timezone

from kafka import KafkaProducer

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "bank-transactions-ml"   # separate topic from the simple stream

NUM_ACCOUNTS = 150
TOTAL_TRANSACTIONS = 400
RISKY_ACCOUNT_RATE = 0.10

# Exact category vocabulary from your training data - matching this matters,
# since your OneHotEncoder only recognizes values it saw during training.
MERCHANT_CATEGORIES = [
    "digital_subscriptions", "electronics", "fashion", "fuel", "gambling",
    "grocery", "luxury_goods", "online_services", "restaurant", "travel",
]
HIGH_RISK_CATEGORIES = {"gambling", "luxury_goods"}   # reasonable real-world assumption
COUNTRIES = ["AU", "BR", "CA", "DE", "FR", "IN", "MX", "NG", "UK", "US"]
DEVICE_TYPES = ["desktop", "mobile", "tablet", "wearable"]
TRANSACTION_TYPES = ["atm_withdrawal", "card_present_moto", "online", "pos"]
REGIONS = ["Africa", "Asia", "Europe", "North America", "South America"]

# Best-effort country -> region mapping. Real datasets don't always keep
# these perfectly correlated (e.g. VPNs, card issued in one country used
# abroad) so we mostly follow this mapping but occasionally deviate -
# see build_transaction() below.
COUNTRY_REGION = {
    "AU": "Asia", "BR": "South America", "CA": "North America",
    "DE": "Europe", "FR": "Europe", "IN": "Asia", "MX": "North America",
    "NG": "Africa", "UK": "Europe", "US": "North America",
}


def build_account_pool(n: int):
    accounts = []
    for _ in range(n):
        is_risky = random.random() < RISKY_ACCOUNT_RATE
        accounts.append({
            "account_id": f"ACC{uuid.uuid4().hex[:8].upper()}",
            "home_country": random.choice(COUNTRIES),
            "usual_device": random.choice(DEVICE_TYPES),
            "avg_spend": round(random.uniform(15, 200), 2),
            "account_age_days": random.randint(30, 3650),
            "is_risky": is_risky,
            "previous_chargebacks": random.randint(1, 5) if is_risky else random.choices([0, 1], weights=[0.9, 0.1])[0],
            "risk_score": round(random.uniform(50, 95), 1) if is_risky else round(random.uniform(0, 40), 1),
            "recent_txns": deque(),   # (datetime, amount) - our in-memory "history"
        })
    return accounts


def rolling_stats(account, now):
    """Computes velocity/average features from this account's in-memory
    transaction history - the same idea as a real 'customer_total_transactions_30d'
    SQL query, just against memory instead of a database."""
    recent = account["recent_txns"]

    count_1h = sum(1 for ts, _ in recent if (now - ts).total_seconds() <= 3600)
    count_24h = sum(1 for ts, _ in recent if (now - ts).total_seconds() <= 86400)
    count_30d = sum(1 for ts, _ in recent if (now - ts).days <= 30)
    amounts_30d = [amt for ts, amt in recent if (now - ts).days <= 30]
    avg_30d = round(sum(amounts_30d) / len(amounts_30d), 2) if amounts_30d else account["avg_spend"]

    return count_1h, count_24h, count_30d, avg_30d


def build_transaction(account, force_fraud_signal=False):
    now = datetime.now(timezone.utc)

    amount = round(max(1, random.gauss(account["avg_spend"], account["avg_spend"] * 0.35)), 2)
    country = account["home_country"]
    device = account["usual_device"]
    merchant = random.choice(MERCHANT_CATEGORIES)

    if force_fraud_signal:
        # Nudge a few fields toward suspicious values - NOT guaranteed
        # detectable, same overlap principle as our earlier synthetic data.
        signal = random.choice(["amount", "location_device", "merchant"])
        if signal == "amount":
            amount = round(account["avg_spend"] * random.uniform(3, 10), 2)
        elif signal == "location_device":
            country = random.choice([c for c in COUNTRIES if c != account["home_country"]])
            device = random.choice([d for d in DEVICE_TYPES if d != account["usual_device"]])
        elif signal == "merchant":
            merchant = random.choice(list(HIGH_RISK_CATEGORIES))

    # Region mostly follows country, but not always (VPNs, card-not-present
    # transactions can legitimately show a mismatched region).
    region = COUNTRY_REGION[country] if random.random() < 0.85 else random.choice(REGIONS)

    count_1h, count_24h, count_30d, avg_30d = rolling_stats(account, now)

    txn = {
        "transaction_id": str(uuid.uuid4()),
        "account_id": account["account_id"],
        "transaction_hour": now.hour,
        "transaction_day_of_week": now.weekday(),   # Monday=0 ... Sunday=6
        "account_age_days": account["account_age_days"],
        "previous_chargebacks": account["previous_chargebacks"],
        "merchant_category": merchant,
        "transaction_country": country,
        "device_type": device,
        "transaction_type": random.choice(TRANSACTION_TYPES),
        "geo_location_region": region,
        "is_international": 1 if country != account["home_country"] else 0,
        "is_high_risk_merchant_category": 1 if merchant in HIGH_RISK_CATEGORIES else 0,
        "is_weekend": 1 if now.weekday() >= 5 else 0,
        "customer_total_transactions_30d": count_30d,
        "customer_risk_score": account["risk_score"],
        "transaction_amount": amount,
        "avg_transaction_amount_30d_customer": avg_30d,
        "transaction_velocity_1h": count_1h,
        "transaction_velocity_24h": count_24h,
        "timestamp": now.isoformat(),
        "is_fraud_actual": force_fraud_signal,   # our own ground truth, NOT a model input
    }

    account["recent_txns"].append((now, amount))
    return txn


def main():
    accounts = build_account_pool(NUM_ACCOUNTS)

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    fraud_count = 0
    for i in range(1, TOTAL_TRANSACTIONS + 1):
        account = random.choice(accounts)

        # Risky accounts produce fraud far more often than normal ones,
        # but neither is deterministic - matching the real, imperfect
        # correlation between "past chargebacks" and "will commit fraud again".
        fraud_chance = 0.25 if account["is_risky"] else 0.02
        is_fraud = random.random() < fraud_chance

        txn = build_transaction(account, force_fraud_signal=is_fraud)
        producer.send(KAFKA_TOPIC, key=account["account_id"].encode("utf-8"), value=txn)

        if is_fraud:
            fraud_count += 1

        tag = "FRAUD" if is_fraud else "  ok  "
        print(f"[{i:>4}/{TOTAL_TRANSACTIONS}] {tag} | {account['account_id']} | "
              f"${txn['transaction_amount']:>9,.2f} | {txn['merchant_category']:<22} | "
              f"risk_score={txn['customer_risk_score']}")

        time.sleep(0.1)

    producer.flush()
    producer.close()

    print(f"\nDone. Sent {TOTAL_TRANSACTIONS} transactions to '{KAFKA_TOPIC}'")
    print(f"Injected {fraud_count} fraudulent transactions ({fraud_count/TOTAL_TRANSACTIONS:.1%})")


if __name__ == "__main__":
    main()