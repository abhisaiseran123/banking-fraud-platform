"""
test_generate_transactions.py

Unit tests for producer/generate_transactions.py - the Stage 1 file-based
transaction generator. These test PURE LOGIC only: no Kafka, no network,
no cloud credentials needed. This is deliberate - CI should never depend
on live infrastructure or secrets to pass.

WHAT "UNIT TEST" MEANS HERE:
Each test checks ONE small, specific behavior of a function in isolation -
e.g. "does a fraudulent transaction actually get labeled as fraud?" -
rather than testing the whole pipeline end to end.

HOW TO RUN:
    pytest tests/test_generate_transactions.py -v
"""

from producer.generate_transactions import (
    build_account_pool,
    normal_transaction,
    fraudulent_transaction,
)


def test_build_account_pool_creates_correct_count():
    accounts = build_account_pool(10)
    assert len(accounts) == 10


def test_build_account_pool_accounts_are_unique():
    accounts = build_account_pool(50)
    account_ids = [a["account_id"] for a in accounts]
    assert len(account_ids) == len(set(account_ids)), "account_ids should all be unique"


def test_build_account_pool_avg_spend_is_positive():
    accounts = build_account_pool(20)
    for account in accounts:
        assert account["avg_spend"] > 0


def test_normal_transaction_has_required_fields():
    accounts = build_account_pool(1)
    txn = normal_transaction(accounts[0])

    required_fields = [
        "transaction_id", "account_id", "amount", "currency",
        "merchant_category", "location", "device_id", "timestamp",
        "is_fraud_actual",
    ]
    for field in required_fields:
        assert field in txn, f"Missing expected field: {field}"


def test_normal_transaction_is_not_labeled_fraud():
    accounts = build_account_pool(1)
    txn = normal_transaction(accounts[0])
    assert txn["is_fraud_actual"] is False


def test_normal_transaction_amount_is_positive():
    accounts = build_account_pool(5)
    for account in accounts:
        txn = normal_transaction(account)
        assert txn["amount"] > 0


def test_fraudulent_transaction_is_labeled_fraud():
    accounts = build_account_pool(1)
    txn = fraudulent_transaction(accounts[0])
    assert txn["is_fraud_actual"] is True


def test_fraudulent_huge_amount_eventually_produces_large_amounts():
    # fraud_type is chosen randomly INSIDE the function, not passed in -
    # so we generate several and confirm at least one shows the expected
    # "way above average" signal, rather than forcing one specific type.
    accounts = build_account_pool(1)
    account = accounts[0]
    amounts = [fraudulent_transaction(account)["amount"] for _ in range(30)]
    assert max(amounts) > account["avg_spend"] * 3, (
        "Expected at least one huge_amount fraud transaction in 30 tries"
    )


def test_fraudulent_transaction_eventually_produces_foreign_location():
    accounts = build_account_pool(1)
    account = accounts[0]
    locations = [fraudulent_transaction(account)["location"] for _ in range(30)]
    # Our convention: foreign locations are "City, Country" - a comma
    # signals a foreign transaction elsewhere in the pipeline too.
    assert any("," in loc for loc in locations), (
        "Expected at least one foreign_location fraud transaction in 30 tries"
    )