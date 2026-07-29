"""
Script 6: Causal reasoning audit, using Pearl's ladder of causation.

Rung 1 (association): correlations actually observed in the scored data.
Rung 2 (intervention): named, data-grounded confounders that could make the
    Rung 1 correlation vanish under a real intervention.
Rung 3 (counterfactual): an honest refusal to invent a counterfactual number
    for a specific transaction, with the concrete reasons why it cannot be
    computed from this data.
"""
import numpy as np
import pandas as pd

IN_PATH = "data/scored_transactions.csv"
REPORT_PATH = "reports/causal_reasoning_report.md"

def cramers_v(confusion_matrix):
    chi2 = 0.0
    n = confusion_matrix.values.sum()
    row_sums = confusion_matrix.sum(axis=1)
    col_sums = confusion_matrix.sum(axis=0)
    expected = np.outer(row_sums, col_sums) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((confusion_matrix.values - expected) ** 2 / expected)
    r, k = confusion_matrix.shape
    denom = min(r - 1, k - 1)
    if denom == 0 or n == 0:
        return float("nan")
    return np.sqrt((chi2 / n) / denom)

def main():
    df = pd.read_csv(IN_PATH)

    # --- Rung 1: Observation ---
    corr_score_fraud = df["fraud_probability"].corr(df["isFraud"])
    corr_amount_fraud = df["amount"].corr(df["isFraud"])
    df["is_transfer"] = (df["type"] == "TRANSFER").astype(int)
    corr_transfer_fraud = df["is_transfer"].corr(df["isFraud"])

    ct = pd.crosstab(df["type"], df["isFraud"])
    type_fraud_cramers_v = cramers_v(ct)

    # --- Rung 2: Intervention / confounders (grounded in real numbers) ---
    mean_amount_by_type = df.groupby("type")["amount"].mean().sort_values(ascending=False)
    transfer_mean_amount = mean_amount_by_type.get("TRANSFER", float("nan"))
    other_mean_amount = df.loc[df["type"] != "TRANSFER", "amount"].mean()
    amount_ratio = transfer_mean_amount / other_mean_amount if other_mean_amount else float("nan")

    confounder_1 = (
        f"Transaction type is confounded with amount: in this sample, mean "
        f"TRANSFER amount is {transfer_mean_amount:,.2f} vs {other_mean_amount:,.2f} "
        f"for all other types combined ({amount_ratio:.2f}x larger). Since the "
        "engine's top SHAP feature involves amount, some of the apparent "
        "type -> fraud association could be routing through amount. If we "
        "intervened to equalize transaction amounts across types, the observed "
        "type/fraud correlation would likely shrink, and we cannot tell from "
        "this observational data how much of the correlation is genuinely "
        "about type versus genuinely about size."
    )
    confounder_2 = (
        "The fraud label itself suffers survivorship / labeling bias: isFraud "
        "in PaySim marks transactions the simulation generated as fraud, not "
        "transactions that were investigated and confirmed. In a real "
        "deployment, isFraud=0 would actually mean 'not caught,' not "
        "'confirmed legitimate.' Because only a fraction of flagged "
        "transactions are ever reviewed by a human, any future retraining on "
        "outcomes would only ever see labels for the subset the engine chose "
        "to surface — a feedback loop that reinforces whatever the current "
        "model already believes, and could make the true underlying "
        "correlation (if we could ever observe it) different from what is "
        "measured here."
    )

    # --- Rung 3: Counterfactual ---
    candidate = df[df["recommendation"] == "INVESTIGATE"].sort_values(
        "fraud_probability", ascending=False
    ).iloc[0]
    counterfactual_question = (
        f"Counterfactual question for this transaction (type={candidate['type']}, "
        f"amount={candidate['amount']:,.2f}, fraud_probability="
        f"{candidate['fraud_probability']:.4f}, isFraud={int(candidate['isFraud'])}): "
        "'If an investigator had NOT reviewed this transaction (i.e. it had been "
        "allocated to CLEAR instead of INVESTIGATE), would the fraud outcome "
        "have been different?'"
    )
    counterfactual_answer = (
        "CANNOT COMPUTE. Three concrete reasons: "
        "(1) No outcome data exists for the counterfactual arm — PaySim's "
        "isFraud is a simulated ground-truth label, not an investigation "
        "outcome, so we have no record of what would have happened to THIS "
        "transaction under a different investigator-attention allocation. "
        "(2) SUTVA (Stable Unit Treatment Value Assumption) is violated: "
        "investigator attention is a shared, finite resource, so whether this "
        "transaction is investigated changes how much attention is available "
        "for every other transaction in the queue. The 'treatment' on one unit "
        "is not independent of treatment assignment on other units, which is a "
        "precondition for a valid counterfactual estimate. "
        "(3) No randomization: transactions were not randomly assigned to "
        "investigate/clear, so any comparison between investigated and "
        "uninvestigated transactions here confounds the investigation decision "
        "itself with the very features (amount, type) that were used to make "
        "that decision. We are not going to invent a probability or an "
        "estimated causal effect to fill this gap."
    )

    verdict = (
        "Honest verdict: what this engine produces is correlation dressed as "
        "causation. The fraud_probability score is a strong statistical "
        "association between transaction features and a fraud label "
        f"(corr={corr_score_fraud:.4f} with isFraud in this sample), and "
        "reallocating investigator attention toward higher-scoring "
        "transactions is a reasonable POLICY under that association. But "
        "nothing in this pipeline demonstrates that investigating "
        "higher-scoring transactions CAUSES more fraud to be caught, prevented, "
        "or reduced, because we never observe the counterfactual (what happens "
        "to a transaction that isn't investigated) and never randomize "
        "investigation assignment. Treat the ranking as a triage heuristic "
        "backed by association, not as a validated causal intervention."
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Causal Reasoning Report\n\n")
        f.write("## Rung 1 — Observation\n\n")
        f.write(f"- corr(fraud_probability, isFraud) = {corr_score_fraud:.4f}\n")
        f.write(f"- corr(amount, isFraud) = {corr_amount_fraud:.4f}\n")
        f.write(f"- corr(is_transfer, isFraud) = {corr_transfer_fraud:.4f}\n")
        f.write(f"- Cramer's V(type, isFraud) = {type_fraud_cramers_v:.4f} "
                f"(association across all 5 type categories, not just TRANSFER)\n\n")
        f.write("## Rung 2 — Intervention (named confounders)\n\n")
        f.write(f"**Confounder 1 — type confounded with amount:**\n\n{confounder_1}\n\n")
        f.write(f"**Confounder 2 — survivorship bias in the fraud label:**\n\n{confounder_2}\n\n")
        f.write("## Rung 3 — Counterfactual\n\n")
        f.write(f"{counterfactual_question}\n\n")
        f.write(f"**Answer: {counterfactual_answer}**\n\n")
        f.write("## Verdict\n\n")
        f.write(verdict + "\n")

    print("Rung 1 — Observation:")
    print(f"  corr(fraud_probability, isFraud) = {corr_score_fraud:.4f}")
    print(f"  corr(amount, isFraud) = {corr_amount_fraud:.4f}")
    print(f"  corr(is_transfer, isFraud) = {corr_transfer_fraud:.4f}")
    print(f"  Cramer's V(type, isFraud) = {type_fraud_cramers_v:.4f}")
    print()
    print("Rung 2 — Confounders named:")
    print("  1. type confounded with amount")
    print("  2. survivorship bias in fraud label")
    print()
    print("Rung 3 — Counterfactual:")
    print(f"  {counterfactual_question}")
    print("  Answer: CANNOT COMPUTE (see report for full reasoning)")
    print()
    print(verdict)
    print(f"\nSaved report to: {REPORT_PATH}")

if __name__ == "__main__":
    main()
