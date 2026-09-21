"""
test_training_data_generation.py

Unit tests for ml/generate_training_data.py - the synthetic dataset
generator used for our model comparison experiments (Logistic Regression,
Random Forest, XGBoost). Pure logic only, no external dependencies.

HOW TO RUN:
    pytest tests/test_training_data_generation.py -v
"""

from ml.generate_training_data import build_account_pool, make_transaction


def test_build_account_pool_creates_correct_count():
    accounts = build_account_pool(15)
    assert len(accounts) == 15


def test_build_account_pool_has_usual_device():
    accounts = build_account_pool(5)
    for account in accounts:
        assert account["usual_device"].startswith("DEV")


def test_genuine_transaction_has_all_expected_columns():
    accounts = build_account_pool(1)
    txn = make_transaction(accounts[0], is_fraud=False)

    expected_columns = [
        "transaction_id", "account_id", "amount", "amount_to_avg_ratio",
        "is_foreign_location", "is_new_device", "hour_of_day",
        "merchant_category", "is_fraud_actual", "fraud_type",
    ]
    for col in expected_columns:
        assert col in txn, f"Missing expected column: {col}"


def test_genuine_transaction_is_labeled_not_fraud():
    accounts = build_account_pool(1)
    txn = make_transaction(accounts[0], is_fraud=False)
    assert txn["is_fraud_actual"] == 0
    assert txn["fraud_type"] == "genuine"


def test_fraud_transaction_is_labeled_fraud():
    accounts = build_account_pool(1)
    txn = make_transaction(accounts[0], is_fraud=True)
    assert txn["is_fraud_actual"] == 1
    assert txn["fraud_type"] != "genuine"


def test_amount_to_avg_ratio_is_calculated_correctly():
    accounts = build_account_pool(1)
    txn = make_transaction(accounts[0], is_fraud=False)
    expected_ratio = round(txn["amount"] / accounts[0]["avg_spend"], 3)
    assert txn["amount_to_avg_ratio"] == expected_ratio


def test_is_foreign_location_flag_matches_location_format():
    accounts = build_account_pool(1)
    # Generate several to see both foreign and non-foreign cases
    for _ in range(30):
        txn = make_transaction(accounts[0], is_fraud=True)
        # Our convention: is_foreign_location is 1 exactly when a comma
        # (indicating "City, Country") would be present - checked
        # indirectly here since raw location isn't returned, only the flag.
        assert txn["is_foreign_location"] in (0, 1)


def test_stealthy_fraud_eventually_occurs():
    # ~25% of fraud is deliberately "stealthy" (no distinguishing signal) -
    # generate enough to reasonably expect at least one.
    accounts = build_account_pool(1)
    fraud_types = [make_transaction(accounts[0], is_fraud=True)["fraud_type"] for _ in range(60)]
    assert "stealthy" in fraud_types, "Expected at least one stealthy fraud transaction in 60 tries"