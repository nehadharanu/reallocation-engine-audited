# Delegation Gate Report

Total transactions evaluated: 9994

## Final state distribution

| State | Count |
|---|---|
| APPROVE | 501 |
| FLAG | 656 |
| BLOCK | 8837 |
| UNDEFINED_GAP | 0 |

## Raw condition counts (before overlap resolution)

- approve_cond_true: 977
- flag_cond_true: 498
- block_cond_true: 8995
- Transactions matching more than one raw condition (overlap): 476
- Overlap resolution precedence used: FLAG > APPROVE > BLOCK

## Hard stop (amount > $1,000,000)

Rows where the hard stop rerouted BLOCK -> FLAG: 158

Verification - any amount > $1,000,000 row still landing in BLOCK: 0 (hard stop holds: True)

## Design gap found

No transactions fell outside the three specified conditions in this run.

## Delegation map: tool vs human

| State | What the TOOL decides | What the HUMAN decides |
|---|---|---|
| APPROVE | Routes transaction straight to the investigator queue as high-confidence (fraud_probability > 0.7, uncertainty < 0.15, amount <= 500,000) | Investigator still performs the actual review; tool does not require a second human sign-off before queueing |
| FLAG | Tool declines to make the call - flags for mandatory human review because either the model is uncertain (uncertainty >= 0.15), the amount is large (> 500,000), or the hard stop forced it here | Human must review before any action (clear or escalate) is taken |
| BLOCK | Tool auto-clears the transaction with no investigator involvement at all (fraud_probability <= 0.3 and amount <= 1,000,000) | None - this is the one state where the tool acts alone |
| HARD STOP (amount > 1,000,000) | Tool is structurally forbidden from auto-clearing regardless of fraud_probability; forces FLAG | Human always reviews any transaction over $1,000,000, no exceptions |

