"""
Script 9: Uncertainty communication.

Bar chart of transaction counts per fraud_probability bucket, colored by
recommendation. Error bars are NOT invented - they are derived from each
transaction's real 90% bootstrap CI (ci_lower_90 / ci_upper_90 from
engine.py): we re-bucket every transaction using its CI lower bound and
again using its CI upper bound, and use the resulting counts as the
asymmetric error range around the point-estimate bucket count. That shows,
concretely, how much each bucket's count could shift if every transaction's
true probability sat at the edge of its own confidence interval.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_PATH = "data/scored_transactions.csv"
CHART_PATH = "reports/uncertainty_chart.png"
REPORT_PATH = "reports/uncertainty_communication.md"

BINS = [0, 0.1, 0.3, 0.7, 1.0]
LABELS = ["0-0.1", "0.1-0.3", "0.3-0.7", "0.7-1.0"]
COLOR_BY_REC = {"CLEAR": "#2ca02c", "MONITOR": "#f4c542", "INVESTIGATE": "#d62728"}
BUCKET_TO_REC = {"0-0.1": "CLEAR", "0.1-0.3": "MONITOR", "0.3-0.7": "INVESTIGATE", "0.7-1.0": "INVESTIGATE"}

def main():
    df = pd.read_csv(IN_PATH)

    df["bucket"] = pd.cut(df["fraud_probability"], bins=BINS, labels=LABELS, include_lowest=True, right=True)
    point_counts = df["bucket"].value_counts().reindex(LABELS).fillna(0).astype(int)

    df["bucket_lo"] = pd.cut(df["ci_lower_90"], bins=BINS, labels=LABELS, include_lowest=True, right=True)
    df["bucket_hi"] = pd.cut(df["ci_upper_90"], bins=BINS, labels=LABELS, include_lowest=True, right=True)
    lo_counts = df["bucket_lo"].value_counts().reindex(LABELS).fillna(0).astype(int)
    hi_counts = df["bucket_hi"].value_counts().reindex(LABELS).fillna(0).astype(int)

    err_lower = (point_counts - lo_counts).clip(lower=0)
    err_upper = (hi_counts - point_counts).clip(lower=0)

    colors = [COLOR_BY_REC[BUCKET_TO_REC[label]] for label in LABELS]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(
        LABELS, point_counts.values, color=colors,
        yerr=[err_lower.values, err_upper.values], capsize=6, ecolor="black",
    )
    ax.set_xlabel("Fraud probability bucket")
    ax.set_ylabel("Transaction count")
    ax.set_title("Transaction counts by fraud-probability bucket\n(error bars = 90% CI re-bucketing range)")
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=COLOR_BY_REC["INVESTIGATE"], label="INVESTIGATE"),
        Patch(facecolor=COLOR_BY_REC["MONITOR"], label="MONITOR"),
        Patch(facecolor=COLOR_BY_REC["CLEAR"], label="CLEAR"),
    ]
    ax.legend(handles=legend_elems)
    for bar, count in zip(bars, point_counts.values):
        ax.annotate(str(count), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    textcoords="offset points", xytext=(0, 8), ha="center")
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150)
    plt.close()

    trust_sentence = (
        "What a non-specialist would trust: the tool is good at telling you "
        "\"this transaction looks nothing like the fraud patterns we've seen "
        "before\" (the CLEAR bucket is large, stable, and has a narrow CI "
        "re-bucketing range), so trusting it to deprioritize the bulk of "
        "obviously-routine transactions is reasonable."
    )
    distrust_sentence = (
        "Where I would NOT trust this tool: (1) for any single transaction near "
        "a decision boundary (fraud_probability just above or below 0.1, 0.3, or "
        "0.7) - the adversarial test showed a 10% amount change alone can flip "
        "the recommendation; (2) as evidence that investigating a transaction "
        "will actually prevent fraud - the causal analysis could not establish "
        "that; (3) as a fairness-neutral tool - the bias audit found a large "
        "demographic-parity gap by transaction type; (4) on any transaction "
        "type or amount range not well represented in the training sample, "
        "since the model has never seen it."
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Uncertainty Communication Report\n\n")
        f.write(f"Chart saved to: {CHART_PATH}\n\n")
        f.write("## Bucket counts (point estimate, and 90% CI re-bucketing range)\n\n")
        f.write("| Bucket | Recommendation | Count | Count if all at CI lower bound | Count if all at CI upper bound |\n")
        f.write("|---|---|---|---|---|\n")
        for label in LABELS:
            f.write(f"| {label} | {BUCKET_TO_REC[label]} | {point_counts[label]} | {lo_counts[label]} | {hi_counts[label]} |\n")
        f.write("\n## Plain language\n\n")
        f.write(f"**{trust_sentence}**\n\n")
        f.write(f"**{distrust_sentence}**\n\n")

    print("Bucket counts (point / CI-lower-rebucket / CI-upper-rebucket):")
    for label in LABELS:
        print(f"  {label} ({BUCKET_TO_REC[label]}): {point_counts[label]} / {lo_counts[label]} / {hi_counts[label]}")
    print(f"Saved chart to: {CHART_PATH}")
    print(f"Saved report to: {REPORT_PATH}")
    print()
    print(trust_sentence)
    print()
    print(distrust_sentence)

if __name__ == "__main__":
    main()
