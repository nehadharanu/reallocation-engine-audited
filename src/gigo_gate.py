"""
Script 2: GIGO (garbage-in-garbage-out) quality gate.

Applies five hard rejection rules to the sampled data before anything is
allowed to train a model or feed a recommendation. Rows failing any rule
are dropped and counted per-rule. Also names three hidden assumptions the
gate does NOT and cannot check.
"""
import os
import pandas as pd

IN_PATH = "data/raw/paysim_sample.csv"
OUT_PATH = "data/gated_transactions.csv"
REPORT_PATH = "reports/gigo_gate_report.md"

VALID_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

def main():
    df = pd.read_csv(IN_PATH)
    start_count = len(df)

    rule_fail = {
        "R1_amount_le_0": df["amount"] <= 0,
        "R2_oldbalanceOrg_negative": df["oldbalanceOrg"] < 0,
        "R3_amount_extreme_outlier_gt_10M": df["amount"] > 10_000_000,
        "R4_invalid_type": ~df["type"].isin(VALID_TYPES),
        "R5_newbalanceOrig_negative": df["newbalanceOrig"] < 0,
    }

    rejection_counts = {name: int(mask.sum()) for name, mask in rule_fail.items()}

    combined_fail = pd.Series(False, index=df.index)
    for mask in rule_fail.values():
        combined_fail = combined_fail | mask

    gated = df[~combined_fail].reset_index(drop=True)
    rejected_total = int(combined_fail.sum())
    kept_total = len(gated)

    os.makedirs("data", exist_ok=True)
    gated.to_csv(OUT_PATH, index=False)

    # --- Threshold sensitivity check for R3 (extreme-outlier amount cutoff) ---
    # Re-run just R3 at three cutoffs, holding the other four rules fixed,
    # so the comparison isolates the effect of the outlier threshold choice.
    other_rules_fail = pd.Series(False, index=df.index)
    for name, mask in rule_fail.items():
        if name != "R3_amount_extreme_outlier_gt_10M":
            other_rules_fail = other_rules_fail | mask

    sensitivity_thresholds = [5_000_000, 10_000_000, 20_000_000]
    sensitivity_results = {}
    for th in sensitivity_thresholds:
        r3_mask_th = df["amount"] > th
        sensitivity_results[th] = {
            "r3_only": int(r3_mask_th.sum()),
            "total_rejected": int((r3_mask_th | other_rules_fail).sum()),
        }

    amount_percentiles = df["amount"].quantile([0.95, 0.99, 0.995, 0.999, 0.9999]).to_dict()
    amount_max = df["amount"].max()

    hidden_assumptions = [
        (
            "Fraud labels are accurate — uninvestigated transactions are assumed "
            "non-fraud (survivorship bias). PaySim's isFraud=0 does not mean "
            "'confirmed legitimate,' it means 'not simulated as fraud in this run.' "
            "In a real deployment, isFraud=0 would mean 'not caught,' which is a "
            "very different claim. This gate cannot detect or correct that; it "
            "only checks internal consistency of the fields it can see."
        ),
        (
            "Conversion/exchange rates and currency units are stable across time "
            "steps. PaySim's 'step' field spans 744 simulated hours; the gate does "
            "not check for regime shifts in amount distributions over time, so a "
            "structural change mid-dataset would pass silently."
        ),
        (
            "Transaction type is consistently recorded (i.e. the same real-world "
            "action is never logged under two different type labels, and type "
            "values are not manually miscoded). The gate only checks that the type "
            "value is one of the five known strings, not that it was assigned "
            "correctly."
        ),
    ]

    what_we_did = (
        "We enforced only checks we can verify from the data itself (field-level "
        "consistency), and rejected outright rather than imputing or silently "
        "coercing bad rows. We did NOT attempt to fix survivorship bias, temporal "
        "drift, or type mislabeling here — those require external validation data "
        "the gate does not have, so they are named as open assumptions instead of "
        "being papered over. All rejected rows are excluded from data/gated_transactions.csv, "
        "and rejection counts are reported per rule so downstream users know exactly "
        "how much data (and what kind) was discarded before modeling."
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# GIGO Gate Report\n\n")
        f.write(f"Input rows: {start_count}\n\n")
        f.write(f"Rows rejected (any rule): {rejected_total}\n\n")
        f.write(f"Rows kept: {kept_total}\n\n")
        f.write("## Rejection counts per rule\n\n")
        f.write("| Rule | Description | Rows rejected |\n")
        f.write("|---|---|---|\n")
        descriptions = {
            "R1_amount_le_0": "amount <= 0 (invalid transaction amount)",
            "R2_oldbalanceOrg_negative": "oldbalanceOrg < 0 (impossible negative balance)",
            "R3_amount_extreme_outlier_gt_10M": "amount > 10,000,000 (extreme outlier, flagged as suspicious data)",
            "R4_invalid_type": "type not in [CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER]",
            "R5_newbalanceOrig_negative": "newbalanceOrig < 0 (impossible negative balance after)",
        }
        for name, count in rejection_counts.items():
            f.write(f"| {name} | {descriptions[name]} | {count} |\n")

        f.write("\n## Threshold sensitivity: R3 extreme-outlier cutoff\n\n")
        f.write(
            "R3 rejects transactions with `amount` above a cutoff, currently "
            "$10,000,000. We re-ran R3 alone at $5M, $10M, and $20M (holding "
            "R1/R2/R4/R5 fixed) to see how sensitive the gate is to this choice:\n\n"
        )
        f.write("| Threshold | Rows rejected by R3 alone | Total rows rejected (all rules) |\n")
        f.write("|---|---|---|\n")
        for th in sensitivity_thresholds:
            r = sensitivity_results[th]
            f.write(f"| ${th:,.0f} | {r['r3_only']} | {r['total_rejected']} |\n")
        f.write("\n**Amount distribution context (this sample):**\n\n")
        f.write("| Percentile | Amount |\n|---|---|\n")
        for pct, val in amount_percentiles.items():
            f.write(f"| p{pct*100:g} | ${val:,.2f} |\n")
        f.write(f"| max | ${amount_max:,.2f} |\n\n")
        r3_5m = sensitivity_results[5_000_000]["r3_only"]
        r3_10m = sensitivity_results[10_000_000]["r3_only"]
        r3_20m = sensitivity_results[20_000_000]["r3_only"]
        f.write(
            f"Threshold sensitivity: at $5M cutoff, {r3_5m} rows rejected; at $10M "
            f"(chosen), {r3_10m} rows rejected; at $20M, {r3_20m} rows rejected. This "
            f"threshold is more load-bearing than we expected going in: moving from "
            f"$5M to $10M drops the rejection count by {r3_5m - r3_10m} rows "
            f"({(r3_5m - r3_10m) / start_count * 100:.2f}% of the 50,000-row sample), "
            f"and moving from $10M to $20M drops it by a further {r3_10m - r3_20m} "
            f"rows. Even the largest of these ({r3_5m} rows at $5M) is only "
            f"{r3_5m / start_count * 100:.2f}% of the sample, so no choice in this "
            "range materially changes the downstream engine's training set size - but "
            f"the ~{r3_5m / max(r3_20m, 1):.0f}x swing in rejected-row count between "
            "$5M and $20M means the specific cutoff does determine which individual "
            "transactions are treated as suspicious data versus large legitimate "
            f"outliers. We chose $10M because it sits above the p99.9 amount "
            f"(${amount_percentiles.get(0.999, float('nan')):,.2f}) in this sample - "
            "high enough that it is not routinely rejecting large-but-plausible "
            "legitimate transfers, while still catching amounts an order of "
            "magnitude beyond the bulk of the distribution. This is a defensible "
            "default, not a validated optimum - a real deployment should set this "
            "from domain knowledge of what a legitimate large transfer looks like "
            "for its actual customer base, not from a percentile alone.\n\n"
        )

        f.write("## Hidden assumptions (not enforceable by this gate)\n\n")
        for i, assumption in enumerate(hidden_assumptions, 1):
            f.write(f"{i}. {assumption}\n\n")

        f.write("## What we did about it\n\n")
        f.write(what_we_did + "\n")

    print(f"Input rows: {start_count}")
    print("Rejection counts per rule:")
    for name, count in rejection_counts.items():
        print(f"  {name}: {count}")
    print(f"Total rejected: {rejected_total}")
    print(f"Rows kept: {kept_total}")
    print("Threshold sensitivity (R3 outlier cutoff):")
    for th in sensitivity_thresholds:
        r = sensitivity_results[th]
        print(f"  ${th:,.0f}: R3-only={r['r3_only']}, total_rejected={r['total_rejected']}")
    print(f"Saved gated data to: {OUT_PATH}")
    print(f"Saved report to: {REPORT_PATH}")

if __name__ == "__main__":
    main()
