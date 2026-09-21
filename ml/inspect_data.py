"""
inspect_data.py

DIAGNOSTIC SCRIPT - not part of the pipeline itself.

Purpose: directly check whether 'stealthy' fraud transactions are actually
statistically indistinguishable from genuine ones in our engineered
features, as intended. If the model is somehow achieving perfect scores
even with stealthy fraud in the mix, this will show us whether that's
because the data isn't as overlapping as we think, or something else.

HOW TO RUN:
    python ml/inspect_data.py
"""

import pandas as pd
from pathlib import Path

DATA_FILE = Path(__file__).parent / "training_data.csv"


def main():
    df = pd.read_csv(DATA_FILE)

    print(f"Total rows: {len(df)}")
    print("\nBreakdown by fraud_type:")
    print(df["fraud_type"].value_counts().to_string())

    print("\n=== amount_to_avg_ratio: mean and std by fraud_type ===")
    print(df.groupby("fraud_type")["amount_to_avg_ratio"].agg(["mean", "std", "count"]).to_string())

    print("\n=== amount: mean and std by fraud_type ===")
    print(df.groupby("fraud_type")["amount"].agg(["mean", "std", "count"]).to_string())

    print("\n=== is_foreign_location: rate by fraud_type ===")
    print(df.groupby("fraud_type")["is_foreign_location"].mean().to_string())

    print("\n=== is_new_device: rate by fraud_type ===")
    print(df.groupby("fraud_type")["is_new_device"].mean().to_string())

    # The critical check: compare 'stealthy' fraud directly against 'genuine'.
    # If our design is working, these two rows should look nearly identical.
    print("\n=== Direct comparison: genuine vs stealthy fraud ===")
    subset = df[df["fraud_type"].isin(["genuine", "stealthy"])]
    print(subset.groupby("fraud_type")[
        ["amount", "amount_to_avg_ratio", "is_foreign_location", "is_new_device", "hour_of_day"]
    ].mean().to_string())


if __name__ == "__main__":
    main()