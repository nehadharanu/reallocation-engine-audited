# GIGO Gate Report

Input rows: 50000

Rows rejected (any rule): 33

Rows kept: 49967

## Rejection counts per rule

| Rule | Description | Rows rejected |
|---|---|---|
| R1_amount_le_0 | amount <= 0 (invalid transaction amount) | 10 |
| R2_oldbalanceOrg_negative | oldbalanceOrg < 0 (impossible negative balance) | 0 |
| R3_amount_extreme_outlier_gt_10M | amount > 10,000,000 (extreme outlier, flagged as suspicious data) | 23 |
| R4_invalid_type | type not in [CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER] | 0 |
| R5_newbalanceOrig_negative | newbalanceOrig < 0 (impossible negative balance after) | 0 |

## Threshold sensitivity: R3 extreme-outlier cutoff

R3 rejects transactions with `amount` above a cutoff, currently $10,000,000. We re-ran R3 alone at $5M, $10M, and $20M (holding R1/R2/R4/R5 fixed) to see how sensitive the gate is to this choice:

| Threshold | Rows rejected by R3 alone | Total rows rejected (all rules) |
|---|---|---|
| $5,000,000 | 563 | 573 |
| $10,000,000 | 23 | 33 |
| $20,000,000 | 6 | 16 |

**Amount distribution context (this sample):**

| Percentile | Amount |
|---|---|
| p95 | $1,028,488.04 |
| p99 | $5,591,873.94 |
| p99.5 | $9,422,015.99 |
| p99.9 | $10,000,000.00 |
| p99.99 | $25,550,257.08 |
| max | $47,542,583.98 |

Threshold sensitivity: at $5M cutoff, 563 rows rejected; at $10M (chosen), 23 rows rejected; at $20M, 6 rows rejected. This threshold is more load-bearing than we expected going in: moving from $5M to $10M drops the rejection count by 540 rows (1.08% of the 50,000-row sample), and moving from $10M to $20M drops it by a further 17 rows. Even the largest of these (563 rows at $5M) is only 1.13% of the sample, so no choice in this range materially changes the downstream engine's training set size - but the ~94x swing in rejected-row count between $5M and $20M means the specific cutoff does determine which individual transactions are treated as suspicious data versus large legitimate outliers. We chose $10M because it sits above the p99.9 amount ($10,000,000.00) in this sample - high enough that it is not routinely rejecting large-but-plausible legitimate transfers, while still catching amounts an order of magnitude beyond the bulk of the distribution. This is a defensible default, not a validated optimum - a real deployment should set this from domain knowledge of what a legitimate large transfer looks like for its actual customer base, not from a percentile alone.

## Hidden assumptions (not enforceable by this gate)

1. Fraud labels are accurate — uninvestigated transactions are assumed non-fraud (survivorship bias). PaySim's isFraud=0 does not mean 'confirmed legitimate,' it means 'not simulated as fraud in this run.' In a real deployment, isFraud=0 would mean 'not caught,' which is a very different claim. This gate cannot detect or correct that; it only checks internal consistency of the fields it can see.

2. Conversion/exchange rates and currency units are stable across time steps. PaySim's 'step' field spans 744 simulated hours; the gate does not check for regime shifts in amount distributions over time, so a structural change mid-dataset would pass silently.

3. Transaction type is consistently recorded (i.e. the same real-world action is never logged under two different type labels, and type values are not manually miscoded). The gate only checks that the type value is one of the five known strings, not that it was assigned correctly.

## What we did about it

We enforced only checks we can verify from the data itself (field-level consistency), and rejected outright rather than imputing or silently coercing bad rows. We did NOT attempt to fix survivorship bias, temporal drift, or type mislabeling here — those require external validation data the gate does not have, so they are named as open assumptions instead of being papered over. All rejected rows are excluded from data/gated_transactions.csv, and rejection counts are reported per rule so downstream users know exactly how much data (and what kind) was discarded before modeling.
