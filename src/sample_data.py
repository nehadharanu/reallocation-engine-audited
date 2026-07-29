"""
Script 1: Stratified sampling of the PaySim dataset.

Reads the full ~6.3M row PaySim CSV in chunks (never loads it all into memory
at once). Pass 1 counts fraud vs non-fraud rows using only the isFraud column.
Pass 2 keeps every fraud row and takes a per-chunk random fraction of
non-fraud rows sized to hit the ~45,000 non-fraud target, then writes a
50,000-row stratified sample with engineered features to data/raw/.
"""
import os
import pandas as pd
import numpy as np

RAW_PATH = "PS_20174392719_1491204439457_log.csv"
OUT_PATH = "data/raw/paysim_sample.csv"

TARGET_FRAUD = 5000
TARGET_NONFRAUD = 45000
CHUNK_SIZE = 500_000
RANDOM_STATE = 42

def main():
    rng = np.random.default_rng(RANDOM_STATE)

    # Pass 1: count fraud / non-fraud rows cheaply (single column only)
    fraud_total = 0
    nonfraud_total = 0
    for chunk in pd.read_csv(RAW_PATH, usecols=["isFraud"], chunksize=CHUNK_SIZE):
        fraud_total += int(chunk["isFraud"].sum())
        nonfraud_total += int((chunk["isFraud"] == 0).sum())

    nonfraud_frac = min(1.0, TARGET_NONFRAUD / nonfraud_total)

    # Pass 2: keep all fraud rows, sample nonfraud rows at nonfraud_frac per chunk
    fraud_chunks = []
    nonfraud_chunks = []
    for chunk in pd.read_csv(RAW_PATH, chunksize=CHUNK_SIZE):
        fraud_rows = chunk[chunk["isFraud"] == 1]
        if len(fraud_rows) > 0:
            fraud_chunks.append(fraud_rows)

        nonfraud_rows = chunk[chunk["isFraud"] == 0]
        if len(nonfraud_rows) > 0:
            sampled = nonfraud_rows.sample(frac=nonfraud_frac, random_state=RANDOM_STATE)
            nonfraud_chunks.append(sampled)

    fraud_df = pd.concat(fraud_chunks, ignore_index=True)
    if len(fraud_df) > TARGET_FRAUD:
        fraud_df = fraud_df.sample(n=TARGET_FRAUD, random_state=RANDOM_STATE)

    nonfraud_df = pd.concat(nonfraud_chunks, ignore_index=True)
    if len(nonfraud_df) > TARGET_NONFRAUD:
        nonfraud_df = nonfraud_df.sample(n=TARGET_NONFRAUD, random_state=RANDOM_STATE)

    sample = pd.concat([fraud_df, nonfraud_df], ignore_index=True)
    sample = sample.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # Engineered features
    sample["balance_change_orig"] = sample["newbalanceOrig"] - sample["oldbalanceOrg"]
    sample["balance_change_dest"] = sample["newbalanceDest"] - sample["oldbalanceDest"]
    sample["amount_to_balance_ratio"] = sample["amount"] / (sample["oldbalanceOrg"] + 1)
    sample["zero_balance_after"] = (sample["newbalanceOrig"] == 0).astype(int)

    os.makedirs("data/raw", exist_ok=True)
    sample.to_csv(OUT_PATH, index=False)

    fraud_count = int(sample["isFraud"].sum())
    row_count = len(sample)
    fraud_rate = fraud_count / row_count

    print(f"Full dataset: {fraud_total + nonfraud_total} rows ({fraud_total} fraud, {nonfraud_total} non-fraud)")
    print(f"Row count: {row_count}")
    print(f"Fraud count: {fraud_count}")
    print(f"Fraud rate: {fraud_rate:.4f}")
    print(f"Columns: {list(sample.columns)}")
    print(f"Saved to: {OUT_PATH}")

if __name__ == "__main__":
    main()
