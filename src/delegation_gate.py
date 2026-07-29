"""
Script 8: Delegation gate - hard-stop states between the model and action.

Implements APPROVE / FLAG / BLOCK as specified, resolves overlaps between
the three conditions with an explicit, documented precedence (FLAG wins
over APPROVE, APPROVE wins over BLOCK), and enforces the hard stop: a
transaction with amount > 1,000,000 can never land in BLOCK (auto-clear)
no matter what fraud_probability says - it is rerouted to FLAG.

Note: this dataset has no `liveness_check` field (that would come from a
real-time auth signal at transaction time, which PaySim does not simulate).
We do not fabricate one; liveness_check is treated as True for all rows,
and this is stated explicitly rather than hidden.
"""
import pandas as pd

IN_PATH = "data/scored_transactions.csv"
REPORT_PATH = "reports/delegation_gate_report.md"

HARD_STOP_AMOUNT = 1_000_000

def classify(row):
    p = row["fraud_probability"]
    u = row["uncertainty"]
    amt = row["amount"]
    liveness_check = True  # not present in PaySim; documented assumption, not fabricated data

    approve_cond = (p > 0.7) and (u < 0.15)
    flag_cond = (p > 0.3) and (u >= 0.15 or amt > 500_000)
    block_cond = (p <= 0.3) or (not liveness_check)

    matches = []
    if approve_cond:
        matches.append("APPROVE")
    if flag_cond:
        matches.append("FLAG")
    if block_cond:
        matches.append("BLOCK")

    hard_stop_triggered = False
    if block_cond and amt > HARD_STOP_AMOUNT:
        hard_stop_triggered = True
        if "BLOCK" in matches:
            matches.remove("BLOCK")
        if "FLAG" not in matches:
            matches.append("FLAG")

    if len(matches) == 0:
        final_state = "UNDEFINED_GAP"
    elif len(matches) == 1:
        final_state = matches[0]
    else:
        if "FLAG" in matches:
            final_state = "FLAG"
        elif "APPROVE" in matches:
            final_state = "APPROVE"
        else:
            final_state = "BLOCK"

    return pd.Series({
        "approve_cond": approve_cond,
        "flag_cond": flag_cond,
        "block_cond": block_cond,
        "hard_stop_triggered": hard_stop_triggered,
        "n_raw_matches": len(matches) if not hard_stop_triggered else len(matches),
        "final_state": final_state,
    })

def main():
    df = pd.read_csv(IN_PATH)
    results = df.apply(classify, axis=1)
    out = pd.concat([df, results], axis=1)

    state_counts = out["final_state"].value_counts().to_dict()
    raw_condition_counts = {
        "approve_cond_true": int(out["approve_cond"].sum()),
        "flag_cond_true": int(out["flag_cond"].sum()),
        "block_cond_true": int(out["block_cond"].sum()),
    }
    overlap_count = int((out[["approve_cond", "flag_cond", "block_cond"]].sum(axis=1) > 1).sum())
    hard_stop_count = int(out["hard_stop_triggered"].sum())
    undefined_gap_count = int((out["final_state"] == "UNDEFINED_GAP").sum())
    undefined_gap_rows = out[out["final_state"] == "UNDEFINED_GAP"]

    # Verify the hard stop actually blocks: no amount > 1,000,000 row should ever be BLOCK
    violating_hard_stop = out[(out["amount"] > HARD_STOP_AMOUNT) & (out["final_state"] == "BLOCK")]
    hard_stop_holds = len(violating_hard_stop) == 0

    delegation_map = """
| State | What the TOOL decides | What the HUMAN decides |
|---|---|---|
| APPROVE | Routes transaction straight to the investigator queue as high-confidence (fraud_probability > 0.7, uncertainty < 0.15, amount <= 500,000) | Investigator still performs the actual review; tool does not require a second human sign-off before queueing |
| FLAG | Tool declines to make the call - flags for mandatory human review because either the model is uncertain (uncertainty >= 0.15), the amount is large (> 500,000), or the hard stop forced it here | Human must review before any action (clear or escalate) is taken |
| BLOCK | Tool auto-clears the transaction with no investigator involvement at all (fraud_probability <= 0.3 and amount <= 1,000,000) | None - this is the one state where the tool acts alone |
| HARD STOP (amount > 1,000,000) | Tool is structurally forbidden from auto-clearing regardless of fraud_probability; forces FLAG | Human always reviews any transaction over $1,000,000, no exceptions |
"""

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Delegation Gate Report\n\n")
        f.write(f"Total transactions evaluated: {len(out)}\n\n")
        f.write("## Final state distribution\n\n")
        f.write("| State | Count |\n|---|---|\n")
        for state in ["APPROVE", "FLAG", "BLOCK", "UNDEFINED_GAP"]:
            f.write(f"| {state} | {state_counts.get(state, 0)} |\n")

        f.write("\n## Raw condition counts (before overlap resolution)\n\n")
        for k, v in raw_condition_counts.items():
            f.write(f"- {k}: {v}\n")
        f.write(f"- Transactions matching more than one raw condition (overlap): {overlap_count}\n")
        f.write(f"- Overlap resolution precedence used: FLAG > APPROVE > BLOCK\n\n")

        f.write("## Hard stop (amount > $1,000,000)\n\n")
        f.write(f"Rows where the hard stop rerouted BLOCK -> FLAG: {hard_stop_count}\n\n")
        f.write(f"Verification - any amount > $1,000,000 row still landing in BLOCK: "
                f"{len(violating_hard_stop)} (hard stop holds: {hard_stop_holds})\n\n")

        f.write("## Design gap found\n\n")
        if undefined_gap_count > 0:
            f.write(
                f"**{undefined_gap_count} transactions matched none of the three "
                "specified conditions** (APPROVE requires fraud_probability > 0.7; "
                "FLAG requires fraud_probability > 0.3 AND (high uncertainty OR "
                "amount > 500,000); BLOCK requires fraud_probability <= 0.3). This "
                "leaves an uncovered region: fraud_probability in (0.3, 0.7], "
                "uncertainty < 0.15, amount <= 500,000. These rows fell through "
                "every rule as literally specified. We are surfacing this as a "
                "genuine design gap in the three-state spec rather than silently "
                "assigning them to a bucket the spec doesn't put them in.\n\n"
            )
        else:
            f.write("No transactions fell outside the three specified conditions in this run.\n\n")

        f.write("## Delegation map: tool vs human\n")
        f.write(delegation_map + "\n")

    print(f"Total transactions evaluated: {len(out)}")
    print("Final state distribution:")
    for state in ["APPROVE", "FLAG", "BLOCK", "UNDEFINED_GAP"]:
        print(f"  {state}: {state_counts.get(state, 0)}")
    print(f"Raw condition counts: {raw_condition_counts}")
    print(f"Overlap count (matched >1 raw condition): {overlap_count}")
    print(f"Hard stop reroutes (BLOCK->FLAG): {hard_stop_count}")
    print(f"Hard stop holds (no >$1M row left in BLOCK): {hard_stop_holds}")
    print(f"Design gap - UNDEFINED_GAP rows: {undefined_gap_count}")
    print(f"Saved report to: {REPORT_PATH}")

if __name__ == "__main__":
    main()
