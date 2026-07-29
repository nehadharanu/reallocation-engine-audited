"""
Script 7: Adversarial / stress test.

Perturbation 1 - threshold shift: move the INVESTIGATE cutoff from 0.3 to
0.25, a plausible outcome of a routine model update, and count how many
recommendations flip.

Perturbation 2 - amount inflation: add 10% to `amount` for all TRANSFER
transactions (simulating a data entry error or a gaming attempt) and
re-score with the real trained model, then compare recommendation
distributions before/after.
"""
import os
import joblib
import numpy as np
import pandas as pd

IN_PATH = "data/scored_transactions.csv"
MODEL_DIR = "models"
REPORT_PATH = "reports/adversarial_report.md"

def recommend(p, invest_th=0.3, monitor_th=0.1):
    if p > invest_th:
        return "INVESTIGATE"
    elif p > monitor_th:
        return "MONITOR"
    else:
        return "CLEAR"

def main():
    df = pd.read_csv(IN_PATH)
    model = joblib.load(os.path.join(MODEL_DIR, "rf_model.joblib"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.joblib"))

    # --- Perturbation 1: threshold shift 0.3 -> 0.25 ---
    df["recommendation_shifted"] = df["fraud_probability"].apply(lambda p: recommend(p, invest_th=0.25))
    changed_mask = df["recommendation"] != df["recommendation_shifted"]
    n_changed = int(changed_mask.sum())
    direction_counts = (
        df.loc[changed_mask]
        .groupby(["recommendation", "recommendation_shifted"])
        .size()
        .to_dict()
    )

    # --- Perturbation 2: +10% amount on TRANSFER, re-scored with the real model ---
    df_mod = df.copy()
    transfer_mask = df_mod["type"] == "TRANSFER"
    df_mod.loc[transfer_mask, "amount"] = df_mod.loc[transfer_mask, "amount"] * 1.10
    df_mod.loc[transfer_mask, "amount_to_balance_ratio"] = (
        df_mod.loc[transfer_mask, "amount"] / (df_mod.loc[transfer_mask, "oldbalanceOrg"] + 1)
    )
    # balance_change_orig/dest, zero_balance_after, type_encoded are left as observed:
    # we have no ledger data for what the *new* balances would be under a synthetic
    # 10% amount inflation, so this perturbation only changes amount and its direct
    # derivative (amount_to_balance_ratio) - a documented limitation, not an oversight.

    X_mod = df_mod[feature_cols]
    new_proba = model.predict_proba(X_mod)[:, 1]
    df_mod["fraud_probability_inflated"] = new_proba
    df_mod["recommendation_inflated"] = [recommend(p) for p in new_proba]

    transfer_before = df.loc[transfer_mask, "recommendation"].value_counts().to_dict()
    transfer_after = df_mod.loc[transfer_mask, "recommendation_inflated"].value_counts().to_dict()
    transfer_changed_mask = transfer_mask & (df["recommendation"] != df_mod["recommendation_inflated"])
    n_transfer_changed = int(transfer_changed_mask.sum())

    overall_before = df["recommendation"].value_counts().to_dict()
    overall_after = df_mod["recommendation_inflated"].value_counts().to_dict()

    # --- Cross-component check: does the amount-inflation perturbation widen
    # or narrow the demographic-parity gap found in bias_audit.py? OTHER is
    # untouched by this perturbation (only TRANSFER amounts were inflated), so
    # its selection rate is identical before/after - only TRANSFER moves. ---
    other_mask = ~transfer_mask
    other_selection_rate = (df.loc[other_mask, "recommendation"] == "INVESTIGATE").mean()
    transfer_selection_rate_before = (df.loc[transfer_mask, "recommendation"] == "INVESTIGATE").mean()
    transfer_selection_rate_after = (df_mod.loc[transfer_mask, "recommendation_inflated"] == "INVESTIGATE").mean()
    dp_gap_before = abs(transfer_selection_rate_before - other_selection_rate)
    dp_gap_after = abs(transfer_selection_rate_after - other_selection_rate)
    dp_gap_delta = dp_gap_after - dp_gap_before
    dp_gap_direction = "WIDENS" if dp_gap_delta > 0 else ("NARROWS" if dp_gap_delta < 0 else "UNCHANGED")

    # --- Failure condition ---
    near_boundary_1 = df[(df["fraud_probability"] > 0.25) & (df["fraud_probability"] <= 0.3)]
    n_near_boundary_1 = len(near_boundary_1)

    failure_condition = (
        f"The engine's recommendation is fragile for any transaction whose "
        f"fraud_probability sits within a narrow band around the decision "
        f"boundary. A 5-point threshold shift (0.30 -> 0.25) flipped "
        f"{n_changed} of {len(df)} recommendations ({n_changed/len(df)*100:.2f}%) "
        f"purely from transactions with probability in (0.25, 0.30] "
        f"({n_near_boundary_1} such rows) - none of these transactions changed "
        f"in any real way, only the threshold moved. Separately, a 10% amount "
        f"inflation on TRANSFER transactions (representing a plausible data "
        f"entry error or an adversary padding amounts to see how the system "
        f"reacts) changed the recommendation for {n_transfer_changed} of "
        f"{int(transfer_mask.sum())} TRANSFER transactions. Both failure modes "
        f"point to the same root cause: hard threshold cutoffs on a continuous "
        f"probability score are inherently unstable for transactions near the "
        f"cutoff, regardless of whether the underlying risk actually changed."
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Adversarial Report\n\n")
        f.write("## Perturbation 1: Threshold shift (0.30 -> 0.25)\n\n")
        f.write(f"Recommendations changed: **{n_changed}** / {len(df)} "
                f"({n_changed/len(df)*100:.2f}%)\n\n")
        f.write("Direction of change (old -> new : count):\n\n")
        f.write("| From | To | Count |\n|---|---|---|\n")
        for (old, new), count in direction_counts.items():
            f.write(f"| {old} | {new} | {count} |\n")
        f.write(f"\nTransactions with fraud_probability in (0.25, 0.30]: {n_near_boundary_1}\n\n")

        f.write("## Perturbation 2: +10% amount on TRANSFER transactions\n\n")
        f.write(f"TRANSFER transactions affected: {int(transfer_mask.sum())}\n\n")
        f.write(f"Recommendations changed among TRANSFER rows: **{n_transfer_changed}**\n\n")
        f.write("Recommendation distribution, TRANSFER only, before -> after:\n\n")
        f.write("| Recommendation | Before | After |\n|---|---|---|\n")
        for rec in ["INVESTIGATE", "MONITOR", "CLEAR"]:
            f.write(f"| {rec} | {transfer_before.get(rec, 0)} | {transfer_after.get(rec, 0)} |\n")
        f.write("\nRecommendation distribution, full test set, before -> after:\n\n")
        f.write("| Recommendation | Before | After |\n|---|---|---|\n")
        for rec in ["INVESTIGATE", "MONITOR", "CLEAR"]:
            f.write(f"| {rec} | {overall_before.get(rec, 0)} | {overall_after.get(rec, 0)} |\n")

        f.write("\n## Failure condition\n\n")
        f.write(failure_condition + "\n")

        f.write("\n## Cross-component: does this perturbation worsen the fairness gap?\n\n")
        f.write(
            "OTHER's selection rate is unaffected by this perturbation (only "
            "TRANSFER amounts were inflated), so any change in the demographic "
            "parity gap comes entirely from the TRANSFER side.\n\n"
        )
        f.write("| | Before inflation | After +10% TRANSFER inflation |\n|---|---|---|\n")
        f.write(f"| TRANSFER selection rate | {transfer_selection_rate_before:.4f} | {transfer_selection_rate_after:.4f} |\n")
        f.write(f"| OTHER selection rate | {other_selection_rate:.4f} | {other_selection_rate:.4f} (unchanged) |\n")
        f.write(f"| Demographic parity gap | {dp_gap_before:.4f} | {dp_gap_after:.4f} |\n")
        f.write(f"\n**Gap {dp_gap_direction} by {abs(dp_gap_delta):.4f}** "
                f"({'a plausible data error makes the fairness problem worse' if dp_gap_direction == 'WIDENS' else 'a plausible data error incidentally makes the measured gap smaller, driven by the same non-linear MONITOR-shift seen in perturbation 2, not by any fairness-aware behavior' if dp_gap_direction == 'NARROWS' else 'no measurable change'}).\n")

    print("Perturbation 1 (threshold 0.30 -> 0.25):")
    print(f"  Recommendations changed: {n_changed} / {len(df)} ({n_changed/len(df)*100:.2f}%)")
    print(f"  Direction breakdown: {direction_counts}")
    print()
    print("Perturbation 2 (+10% amount on TRANSFER):")
    print(f"  TRANSFER rows affected: {int(transfer_mask.sum())}")
    print(f"  TRANSFER recommendations changed: {n_transfer_changed}")
    print(f"  TRANSFER before: {transfer_before}")
    print(f"  TRANSFER after:  {transfer_after}")
    print()
    print("Failure condition:")
    print(f"  {failure_condition}")
    print()
    print("Cross-component: fairness gap before/after inflation:")
    print(f"  TRANSFER selection rate: {transfer_selection_rate_before:.4f} -> {transfer_selection_rate_after:.4f}")
    print(f"  OTHER selection rate (unchanged): {other_selection_rate:.4f}")
    print(f"  Demographic parity gap: {dp_gap_before:.4f} -> {dp_gap_after:.4f} ({dp_gap_direction} by {abs(dp_gap_delta):.4f})")
    print(f"\nSaved report to: {REPORT_PATH}")

if __name__ == "__main__":
    main()
