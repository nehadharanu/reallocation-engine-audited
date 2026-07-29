"""
Script 4: Bias audit.

Protected axis: transaction TYPE, collapsed to TRANSFER vs all-other-types.
Computes two competing fairness definitions on the real scored test set:
  1. Demographic parity - selection rate (INVESTIGATE) equal across groups
  2. Equalized odds - true positive rate equal across groups
and shows, with the actual numbers, that both cannot hold at once when the
groups have different base fraud rates.
"""
import numpy as np
import pandas as pd

IN_PATH = "data/scored_transactions.csv"
REPORT_PATH = "reports/bias_audit_report.md"

def rate(mask_num, mask_den):
    denom = mask_den.sum()
    if denom == 0:
        return None
    return mask_num.sum() / denom

def main():
    df = pd.read_csv(IN_PATH)
    df["is_transfer"] = df["type"] == "TRANSFER"
    df["selected"] = df["recommendation"] == "INVESTIGATE"

    groups = {"TRANSFER": df["is_transfer"], "OTHER": ~df["is_transfer"]}

    results = {}
    for name, mask in groups.items():
        g = df[mask]
        n = len(g)
        base_fraud_rate = g["isFraud"].mean() if n > 0 else None
        selection_rate = g["selected"].mean() if n > 0 else None

        fraud_mask = g["isFraud"] == 1
        nonfraud_mask = g["isFraud"] == 0
        tpr = rate(g.loc[fraud_mask, "selected"], fraud_mask) if fraud_mask.any() else None
        fpr = rate(g.loc[nonfraud_mask, "selected"], nonfraud_mask) if nonfraud_mask.any() else None

        results[name] = {
            "n": n,
            "base_fraud_rate": base_fraud_rate,
            "selection_rate": selection_rate,
            "tpr": tpr,
            "fpr": fpr,
        }

    dp_gap = abs(results["TRANSFER"]["selection_rate"] - results["OTHER"]["selection_rate"])
    tpr_vals = [results["TRANSFER"]["tpr"], results["OTHER"]["tpr"]]
    eo_gap = abs(tpr_vals[0] - tpr_vals[1]) if None not in tpr_vals else None
    fpr_vals = [results["TRANSFER"]["fpr"], results["OTHER"]["fpr"]]
    fpr_gap = abs(fpr_vals[0] - fpr_vals[1]) if None not in fpr_vals else None

    base_rate_gap = abs(results["TRANSFER"]["base_fraud_rate"] - results["OTHER"]["base_fraud_rate"])

    # Try fairlearn for a cross-check on selection_rate / TPR (MetricFrame),
    # falls back silently to the manual numbers above if unavailable.
    fairlearn_available = False
    try:
        from fairlearn.metrics import MetricFrame, selection_rate as fl_selection_rate, true_positive_rate as fl_tpr
        mf = MetricFrame(
            metrics={"selection_rate": fl_selection_rate, "tpr": fl_tpr},
            y_true=df["isFraud"],
            y_pred=df["selected"].astype(int),
            sensitive_features=df["is_transfer"].map({True: "TRANSFER", False: "OTHER"}),
        )
        fairlearn_frame = mf.by_group
        fairlearn_available = True
    except Exception as e:
        fairlearn_frame = None
        fairlearn_error = str(e)

    intervention = (
        "Highest-leverage intervention point: the DECISION THRESHOLD applied to "
        "fraud_probability, not the underlying model. Because base fraud rates "
        "differ sharply between TRANSFER and OTHER transaction types in this "
        f"sample ({results['TRANSFER']['base_fraud_rate']:.4f} vs "
        f"{results['OTHER']['base_fraud_rate']:.4f}), a single global threshold "
        "(0.3) cannot equalize both selection rate and true positive rate across "
        "groups simultaneously. The threshold is also the cheapest point to "
        "intervene on: it requires no retraining, is auditable, and can be set "
        "per-group if the organization decides which fairness definition it is "
        "willing to prioritize."
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Bias Audit Report\n\n")
        f.write("Protected axis: transaction TYPE, collapsed to TRANSFER vs OTHER.\n\n")
        f.write("## Group summary\n\n")
        f.write("| Group | n | Base fraud rate | Selection rate (INVESTIGATE) | TPR | FPR |\n")
        f.write("|---|---|---|---|---|---|\n")
        for name, r in results.items():
            tpr_str = "n/a" if r["tpr"] is None else f"{r['tpr']:.4f}"
            fpr_str = "n/a" if r["fpr"] is None else f"{r['fpr']:.4f}"
            f.write(
                f"| {name} | {r['n']} | {r['base_fraud_rate']:.4f} | "
                f"{r['selection_rate']:.4f} | {tpr_str} | {fpr_str} |\n"
            )
        f.write(f"\nBase fraud rate gap (TRANSFER - OTHER): {base_rate_gap:.4f}\n\n")
        f.write("## Fairness definition 1: Demographic parity\n\n")
        f.write(
            f"Selection rate gap between groups: **{dp_gap:.4f}** "
            f"(TRANSFER: {results['TRANSFER']['selection_rate']:.4f}, "
            f"OTHER: {results['OTHER']['selection_rate']:.4f})\n\n"
        )
        f.write("## Fairness definition 2: Equalized odds (TPR component)\n\n")
        if eo_gap is not None:
            f.write(
                f"True positive rate gap between groups: **{eo_gap:.4f}** "
                f"(TRANSFER: {results['TRANSFER']['tpr']:.4f}, "
                f"OTHER: {results['OTHER']['tpr']:.4f})\n\n"
            )
        else:
            f.write("TPR could not be computed for one group (no fraud cases in that group in the test split).\n\n")
        if fpr_gap is not None:
            f.write(f"False positive rate gap between groups (for completeness): **{fpr_gap:.4f}**\n\n")

        f.write("## The tradeoff, quantified\n\n")
        f.write(
            f"The two groups have different base fraud rates "
            f"({results['TRANSFER']['base_fraud_rate']:.4f} vs "
            f"{results['OTHER']['base_fraud_rate']:.4f}, a gap of {base_rate_gap:.4f}). "
            "This is the textbook condition under which demographic parity and "
            "equalized odds are mathematically incompatible for a non-trivial "
            "classifier: if selection rates are forced equal across groups with "
            "different true fraud rates, the group with the lower true rate will "
            "necessarily have a higher false positive rate (or the higher-rate "
            "group will have a lower true positive rate) than a threshold "
            "optimized per-group would produce. Our engine currently uses one "
            f"global threshold (0.3), which produces a demographic parity gap of "
            f"{dp_gap:.4f}"
            + (f" and an equalized-odds (TPR) gap of {eo_gap:.4f}.\n\n" if eo_gap is not None else ".\n\n")
        )
        f.write("## " + intervention.split(":")[0] + "\n\n")
        f.write(intervention + "\n\n")

        f.write("## Fairlearn cross-check\n\n")
        if fairlearn_available:
            f.write("fairlearn was available; MetricFrame results:\n\n")
            f.write(fairlearn_frame.to_markdown() + "\n")
        else:
            f.write(f"fairlearn MetricFrame could not be used ({fairlearn_error}); metrics above were computed manually with pandas.\n")

    print("Group summary:")
    for name, r in results.items():
        print(f"  {name}: n={r['n']}, base_fraud_rate={r['base_fraud_rate']:.4f}, "
              f"selection_rate={r['selection_rate']:.4f}, "
              f"tpr={'n/a' if r['tpr'] is None else round(r['tpr'],4)}, "
              f"fpr={'n/a' if r['fpr'] is None else round(r['fpr'],4)}")
    print(f"Demographic parity gap (selection rate): {dp_gap:.4f}")
    print(f"Equalized odds gap (TPR): {'n/a' if eo_gap is None else round(eo_gap,4)}")
    print(f"fairlearn available: {fairlearn_available}")
    print(f"Saved report to: {REPORT_PATH}")

if __name__ == "__main__":
    main()
