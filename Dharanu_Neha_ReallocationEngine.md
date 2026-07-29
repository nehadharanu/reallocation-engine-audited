# The Reallocation Engine, Audited
## Fraud Investigation Reallocation Engine — PaySim Synthetic Financial Dataset

**Author:** Neha Dharanu · INFO 7375, Computational Skepticism for AI · 2026-07-28

**Domain anchor:** Chapter 11 of *The Reallocation Engine* — the weighted role
scorer (composite = (Σ vote·weight) × liveness × timeline), adapted here to
reallocate a different scarce resource: finite investigator attention across
flagged financial transactions, ranked and allocated using fraud probability
scores derived from transaction features and historical fraud patterns.

**Repo:** `E:\Assignment7Audit\` · Tool: `src/engine.py` · Orchestrator: `run_all.py`

---

## The objective, stated plainly — and what it leaves out

**Optimizes:** expected fraud caught per investigator slot spent — given a
finite pool of investigator capacity, rank flagged transactions by fraud
probability and allocate the next review slot to the highest-risk,
gate-cleared transaction.

**Leaves out:** the cost of false positives on legitimate customers whose
transactions are delayed or blocked; demographic disparities in who gets
flagged and the business harm of those disparities; feedback-loop bias
(only investigated transactions ever receive confirmed fraud labels, so
any future model trained on outcomes will inherit the current model's
blind spots); and — the honest finding of Component 5 below — whether
investigating a transaction actually causes fraud to be prevented, or
whether the fraud would have occurred regardless of the investigation.

---

## Component 1 — The Working Reallocation Tool

**Run:**
```
python src/engine.py
```

**Real output (from clean run, 2026-07-28):**
```
Train rows: 39973  Test rows: 9994
Accuracy:  0.9991
Precision: 0.9960
Recall:    0.9950
AUC:       0.9979
Recommendation distribution:
  INVESTIGATE: 999
  MONITOR: 9
  CLEAR: 8986

This cycle: move 999 of 9994 transactions' investigator-attention from
routine clearance to priority review (10.0% of queue). Remaining 8986
transactions (89.9%) deprioritized — investigator slots reallocated
away from them.
```

**The quantified move, not just a bucket:** the assignment asks for a move
quantity Q of resource from A to B, not a categorical label. Q here is
concrete: **999 investigator-attention slots** move from the routine-clearance
pool to the priority-review pool this cycle, out of 9,994 transactions
scored (10.0% of the queue). Each scored transaction also carries a new
`attention_reallocated` column (1 if it received a moved slot, 0 if not)
in `data/scored_transactions.csv`, so downstream scripts reference the same
quantified reallocation rather than re-deriving it from the categorical
`recommendation` field.

The tool trains a 200-tree RandomForestClassifier on 39,973 gated transactions
and scores the held-out test set of 9,994 transactions. Each transaction
receives a `fraud_probability` (the fraction of trees voting fraud), an
`uncertainty` estimate (standard deviation of per-tree predictions), and a
90% bootstrap confidence interval derived from 100 resamples of the tree
ensemble. The recommendation is:

- **INVESTIGATE** if `fraud_probability > 0.3` — send to investigator queue
- **MONITOR** if `0.1 < fraud_probability ≤ 0.3` — watch but don't escalate
- **CLEAR** if `fraud_probability ≤ 0.1` — deprioritize, no investigator needed

**Uncertainty attached — and it is not thin.** The 90% bootstrap CI is
derived by resampling the 200 trees 100 times and computing the 5th and
95th percentile of the bootstrap mean distributions. The CI re-bucketing
analysis (Component 9) shows that the CLEAR bucket is stable (8,986 point
estimate, range 8,983–8,988 under re-bucketing), while the MONITOR bucket
is genuinely uncertain (9 point estimate, range 7–12) — which honestly
reflects how few transactions live in that middle band.

**Important caveat on model accuracy:** AUC of 0.9979 and accuracy of
0.9991 are near-perfect. This is a property of PaySim's simulation design,
not evidence the engine would generalize. PaySim generates fraud via
deterministic balance-draining rules, so the balance-change and balance-ratio
features we engineered are nearly sufficient statistics for fraud in this
dataset. A real-world deployment would face far messier patterns, and these
metrics would not hold.

**Reproducibility:** `run_all.py` re-runs the full pipeline in ~25 seconds
with a fixed `random_state=42` throughout. Identical numbers across runs
confirmed.

---

## Component 2 — Data Validation and the GIGO Gate

**Run:**
```
python src/gigo_gate.py
```

**Real output:**
```
Input rows: 50000
Total rejected: 33
Rows kept: 49967
  R1_amount_le_0: 10
  R3_amount_extreme_outlier_gt_10M: 23
```

**Hidden assumptions named:**

1. **Fraud labels are accurate.** PaySim's `isFraud=0` means "not simulated
   as fraud," not "confirmed legitimate." In a real deployment, `isFraud=0`
   would mean "not caught," which is a structurally different claim. The gate
   cannot detect or correct this survivorship bias — it only checks internal
   field consistency.

2. **Amount distributions are stable across time steps.** PaySim spans 744
   simulated hours; the gate does not check for regime shifts in the amount
   distribution over time. A structural change mid-dataset would pass silently.

3. **Transaction type is consistently recorded.** The gate checks that `type`
   is one of the five known strings but cannot verify the label was assigned
   correctly — a TRANSFER mislabeled as CASH_OUT would pass the gate
   undetected.

**The checkable gate:** a transaction is gated only if `amount > 0`,
`oldbalanceOrg ≥ 0`, `amount ≤ 10,000,000`, `type` in the known set, and
`newbalanceOrig ≥ 0`. All five conditions are checkable by a human against
the raw CSV row.

**What we did about it:** rejected rows are excluded from
`data/gated_transactions.csv` entirely — none are imputed, guessed, or
silently coerced. The 23 extreme-amount outliers (>$10M) are the most
consequential rejection: had they been kept, they would have inflated
estimates for that tail of the distribution. Rejection counts are reported
per rule so any downstream user knows exactly what was discarded.

**Threshold sensitivity — why $10M, and does it matter?** The R3 outlier
cutoff was not obviously correct on its own; we re-ran R3 alone at three
thresholds, holding the other four rules fixed:

```
$5,000,000: R3-only=563, total_rejected=573
$10,000,000: R3-only=23, total_rejected=33
$20,000,000: R3-only=6, total_rejected=16
```

This threshold turned out to be more load-bearing than expected: moving
from $5M to $10M drops the rejection count by 540 rows (1.08% of the
50,000-row sample), and moving from $10M to $20M drops it by a further 17
rows — a ~94x swing in rejected-row count between $5M and $20M. Even the
largest of these (563 rows at $5M) is only 1.13% of the sample, so no
choice in this range materially changes the downstream engine's training
set size, but the specific cutoff does determine which individual
transactions are treated as suspicious data versus large legitimate
outliers. We chose $10M because it sits at approximately the p99.9 amount
in this sample ($10,000,000.00) — high enough to not routinely reject
large-but-plausible legitimate transfers, while still catching amounts an
order of magnitude beyond the bulk of the distribution. This is a
defensible default, not a validated optimum: a real deployment should set
this from domain knowledge of what a legitimate large transfer looks like
for its actual customer base, not from a percentile alone.

---

## Component 3 — Bias Audit (data → output)

**Run:**
```
python src/bias_audit.py
```

**Real output:**
```
TRANSFER: n=1204, base_fraud_rate=0.3937, selection_rate=0.3962, tpr=1.0, fpr=0.0041
OTHER: n=8790, base_fraud_rate=0.0596, selection_rate=0.0594, tpr=0.9905, fpr=0.0004
Demographic parity gap (selection rate): 0.3368
Equalized odds gap (TPR): 0.0095
fairlearn available: True
```

**Protected axis:** transaction TYPE, collapsed to TRANSFER vs all other
types. No demographic field (race, gender, income) exists in PaySim — and
that absence is itself a finding: the engine cannot see who initiates a
transaction, only the transaction's features. TRANSFER is the proxy for a
structurally different risk profile.

**Two competing fairness definitions, quantified:**

| | TRANSFER | OTHER | Gap |
|---|---|---|---|
| Base fraud rate | 0.3937 | 0.0596 | 0.3341 |
| Selection rate (DP) | 0.3962 | 0.0594 | **0.3368** |
| True positive rate (EO) | 1.0000 | 0.9905 | **0.0095** |
| False positive rate | 0.0041 | 0.0004 | 0.0037 |

**The tradeoff, stated plainly:** demographic parity requires equal
selection rates (0.3962 vs 0.0594 — a gap of 0.3368, currently violated).
Equalized odds requires equal true positive rates (1.000 vs 0.9905 — a
gap of 0.0095, nearly satisfied). These cannot both hold when base fraud
rates differ by 0.3341: forcing selection-rate equality across groups with
such different true rates would either dramatically lower TRANSFER
investigators' efficiency (many fewer real fraud cases found per slot) or
dramatically raise OTHER false positives. The engine currently optimizes
for equalized-odds performance (which it nearly achieves), at the cost of
a large demographic parity gap.

**Fairlearn cross-check (MetricFrame):** confirmed the manual numbers —
selection_rate OTHER=0.0594, TRANSFER=0.3962; TPR OTHER=0.9905,
TRANSFER=1.0000.

**Highest-leverage intervention point:** the global decision threshold
(0.3). It requires no retraining, is auditable, and could be set
per-group if the organization decides which fairness definition to
prioritize. This is the single cheapest point to change.

**Mechanism:** TRANSFER transactions are flagged more not because the
engine is biased against them, but because TRANSFER is genuinely the
highest-fraud type in this dataset (39.4% base fraud rate). The bias
is in the data structure — TRANSFER is where fraud happens — and the
model learned this correctly. But a legitimate TRANSFER customer faces
a 40% chance of being flagged, which is a real fairness cost even when
the classification is statistically accurate.

---

## Component 4 — Explainability and Its Critique

**Run:**
```
python src/explainability.py
```

**Real output:**
```
Mean |SHAP| by feature:
  amount_to_balance_ratio: 0.132831
  zero_balance_after: 0.083792
  balance_change_orig: 0.044637
  type_encoded: 0.021355
  amount: 0.013276
  balance_change_dest: 0.010117
Dominant feature: amount_to_balance_ratio
```

**What SHAP reveals — and what it corrects about the pre-registered
prediction:** the frictional journal predicted that raw `amount` would
dominate. The actual dominant feature is `amount_to_balance_ratio` (mean
|SHAP| = 0.1328, 10× larger than raw `amount`'s 0.0133). The model did
not learn "flag expensive transactions" — it learned "flag transactions
that drain the sender's account relative to their balance." This is a
more meaningful behavioral signal than raw size, but it creates a specific
failure mode: any transaction where `oldbalanceOrg` is near zero produces
an extremely high ratio regardless of whether the transaction is fraudulent.

**The named misleading case — a $1.93M legitimate TRANSFER:**

| Field | Value |
|---|---|
| type | TRANSFER |
| amount | $1,939,569.24 |
| oldbalanceOrg | $0.00 |
| newbalanceOrig | $0.00 |
| fraud_probability | 0.455 → **INVESTIGATE** |
| isFraud | **0 (legitimate)** |

**SHAP values for this transaction:**

| Feature | SHAP value |
|---|---|
| amount | +0.1763 |
| type_encoded | +0.0958 |
| balance_change_dest | +0.0641 |
| zero_balance_after | +0.0386 |
| amount_to_balance_ratio | −0.0136 |

**Why this is technically accurate but practically misleading:** the SHAP
values are correct — raw `amount` is the largest positive contributor for
THIS specific transaction (because `oldbalanceOrg = 0` makes the ratio
computation degenerate and the model falls back to raw amount). An
investigator given only `fraud_probability = 0.455` would have no way to
see that the model flagged a $1.93M transfer from a zero-balance account
primarily because of its size, not because of any behavioral fraud pattern.
In production, every large legitimate transfer from a new or recently-cleared
account would consume investigator time that a smaller genuinely-suspicious
transaction might have used more productively.

**The gap:** the explanation is accurate about what the model measured;
it is silent about whether what the model measured is the right thing to
measure for this specific case.

**Is this a single anecdote, or a systematic pattern?** To generalize past
the one named case, we filtered every legitimate transaction (isFraud=0)
that received an INVESTIGATE recommendation — all false positives in this
test split — and checked how many share the same degenerate signature:
`zero_balance_after=1` AND `amount_to_balance_ratio > 1000` (an account
that empties to exactly zero, producing a ratio driven to an extreme value
by a near-zero denominator rather than a genuinely large amount relative
to typical activity).

**Result: 6 of 6 false positives (100%) share this exact pattern.** Every
single false positive in this test set is the same failure mode, not a
one-off. Any account whose normal end-state is a full withdrawal, a final
settlement, or an account closure will trigger this signal regardless of
whether fraud actually occurred — this is a systematic blind spot in the
feature set, not a rare edge case.

---

## Component 5 — Causal and Counterfactual Reasoning (Pearl's Three Rungs)

**Run:**
```
python src/causal_analysis.py
```

**Real output:**
```
corr(fraud_probability, isFraud) = 0.9959
corr(amount, isFraud) = 0.4216
corr(is_transfer, isFraud) = 0.3627
Cramer's V(type, isFraud) = 0.4232
```

### Rung 1 — Observation

The fraud_probability score correlates at 0.9959 with isFraud in the test
set. Raw amount correlates at 0.4216. Transaction type association
(Cramér's V) = 0.4232 across all five categories. These are real,
computable correlations in this dataset — not in dispute.

### Rung 2 — Intervention

Would reallocating investigator attention toward high-scoring transactions
*cause* more fraud to be caught? The engine treats Rung 1's correlation
as if it answers this. It does not — and two specific confounders
undermine the interventional claim:

**Confounder 1 — transaction type is confounded with amount:**
Mean TRANSFER amount = $1,099,372 vs $199,308 for all other types combined
(5.52× larger). Since `amount_to_balance_ratio` is the top SHAP feature,
the type → fraud association partially routes through amount. If we
intervened to hold amount constant across types, the observed type/fraud
correlation would attenuate — by how much is unidentifiable from this
observational data.

**Confounder 2 — survivorship bias in the fraud label:**
PaySim's `isFraud` marks transactions the simulator generated as fraud,
not transactions investigated and confirmed. In a real deployment,
`isFraud=0` means "not caught," not "confirmed legitimate." Any future
retraining on real investigation outcomes would only see labels for
transactions the current model chose to surface — a feedback loop that
reinforces the current model's beliefs and makes the true underlying
correlation (if measurable) unknowable from this data.

### Rung 3 — Counterfactual

**Specific case:** type=CASH_OUT, amount=$806,850.06, fraud_probability
= 1.0000, isFraud=1. The engine allocated an investigator slot to this
transaction.

**Counterfactual question:** if this transaction had been allocated to
CLEAR instead of INVESTIGATE, would the fraud outcome have been different?

**Answer: CANNOT COMPUTE.**

Three concrete reasons:

1. **No outcome data for the counterfactual arm.** PaySim's `isFraud` is
   a simulation label, not an investigation outcome. We have no record of
   what would have happened to this specific transaction under a different
   allocation decision.

2. **SUTVA is violated.** Investigator attention is a shared, finite
   resource. Whether this transaction is investigated changes how much
   capacity is available for every other transaction in the queue. The
   treatment on one unit is not independent of treatment assignment on
   others — a precondition for valid counterfactual estimation.

3. **No randomization.** Transactions were not randomly assigned to
   investigate/clear. Any comparison between investigated and
   uninvestigated transactions confounds the investigation decision itself
   with the features (amount, type) used to make it.

We are not going to invent a probability or estimated causal effect to
fill this gap.

**Verdict:** the engine reallocates on correlation dressed as causation.
The fraud_probability ranking is a strong observational association and
a reasonable triage heuristic — but nothing in this pipeline establishes
that investigating high-scoring transactions *causes* more fraud to be
caught. Treat the output as a prioritization heuristic, not a validated
causal intervention.

---

## Component 6 — Adversarial Robustness and Fragility

**Run:**
```
python src/adversarial_test.py
```

**Real output:**
```
Perturbation 1 (threshold 0.30 -> 0.25):
  Recommendations changed: 0 / 9994 (0.00%)

Perturbation 2 (+10% amount on TRANSFER):
  TRANSFER rows affected: 1204
  TRANSFER recommendations changed: 269
  TRANSFER before: INVESTIGATE=477, MONITOR=3, CLEAR=724
  TRANSFER after:  INVESTIGATE=232, MONITOR=268, CLEAR=704
```

**Perturbation 1 — threshold shift (0.30 → 0.25):**
Zero recommendations changed. This initially appears to show robustness,
but it is actually a finding about PaySim's near-deterministic structure:
there are essentially no transactions with `fraud_probability` in the
(0.25, 0.30] band because the model assigns near-0 or near-1 probabilities
for almost every transaction. The model learned that fraud in PaySim is
close to perfectly separable by balance-change features. A real-world
deployment would have many transactions in that band and would be far more
fragile to threshold shifts than this result implies. The clean number
is itself misleading about real-world robustness.

**Perturbation 2 — 10% amount inflation on TRANSFER transactions:**
269 of 1,204 TRANSFER recommendations changed — and counterintuitively,
INVESTIGATE *decreased* (477 → 232) while MONITOR *increased* (3 → 268).
This was not predicted before the build. Inflating amount also inflated
`amount_to_balance_ratio`, but for transactions already at very high ratio
values (already in the INVESTIGATE band), the model's response was
non-linear: marginal INVESTIGATE cases shifted down to MONITOR rather than
higher-confidence cases shifting up. A plausible 10% data entry error on
amounts moves 22% of TRANSFER recommendations — not in the direction an
attacker would necessarily want, but demonstrably fragile.

**Failure condition:** the engine's recommendations are fragile for
transactions near a decision boundary. A data perturbation (10% amount
inflation) that a human reviewer might not notice in an audit changed
22% of TRANSFER recommendations. The failure is not in the model's
average behavior — it is at the margin, exactly where the decisions
are least certain and most consequential.

---

## Cross-component finding: does adversarial fragility compound the fairness gap?

Component 3 found a demographic parity gap of **0.3368** between TRANSFER
and OTHER transaction types. Component 6's amount-inflation perturbation
only touches TRANSFER transactions. The natural next question — the one a
component-by-component report misses if it never connects them — is
whether that same plausible data error also makes the fairness problem
worse.

**Run:**
```
python src/adversarial_test.py
```

**Real output (added to the perturbation-2 analysis):**
```
Cross-component: fairness gap before/after inflation:
  TRANSFER selection rate: 0.3962 -> 0.1927
  OTHER selection rate (unchanged): 0.0594
  Demographic parity gap: 0.3368 -> 0.1333 (NARROWS by 0.2035)
```

Because only TRANSFER amounts were inflated, OTHER's selection rate is
mechanically unchanged (0.0594 before and after). All movement in the gap
comes from TRANSFER's selection rate collapsing from 0.3962 to 0.1927 —
the same non-linear INVESTIGATE→MONITOR shift documented in Component 6
(477 → 232 INVESTIGATE among TRANSFER rows).

**The gap narrows, not widens — but this is not good news.** A shrinking
demographic parity gap sounds like an improvement, but it isn't one: it is
the side effect of the model becoming systematically *less* willing to
flag TRANSFER fraud once amounts are inflated, not evidence of any
fairness-aware correction. The mechanism is identical to the fragility
finding in Component 6 — marginal INVESTIGATE cases shift down into
MONITOR as the ratio feature saturates — applied to a subset (TRANSFER)
that already carries nearly all of the fraud base rate in this dataset. In
other words, the same 10% amount error that "narrows" the fairness gap
does so by making the engine miss more TRANSFER fraud, which is a
recall failure wearing a fairness improvement's clothes. If the underlying
mechanism had pushed the other direction (as intuition would predict —
larger amounts should mean *more* flagging, not less), the gap would have
widened instead, and the tool would have been doubly unfair under a
plausible data error: worse for legitimate TRANSFER customers on top of
an already-large baseline gap. Either direction is a failure; this run
happened to produce the less obviously alarming one, which is itself worth
flagging — a report that only checked "did the gap widen?" would have
missed that the gap narrowed for the wrong reason.

---

## Component 7 — Delegation Map and the Hard-Stop Gate

**Run:**
```
python src/delegation_gate.py
```

**Real output:**
```
Final state distribution:
  APPROVE: 501
  FLAG: 656
  BLOCK: 8837
  UNDEFINED_GAP: 0
Hard stop reroutes (BLOCK->FLAG): 158
Hard stop holds (no >$1M row left in BLOCK): True
```

**Three hard-stop states:**

| State | Trigger | Response |
|---|---|---|
| **APPROVE** | fraud_probability > 0.7 AND uncertainty < 0.15 | Routes to investigator queue — confident recommendation, investigator still reviews |
| **FLAG** | fraud_probability > 0.3 AND (uncertainty ≥ 0.15 OR amount > $500K), OR hard stop | Human must review before any action — tool declines to make the call |
| **BLOCK** | fraud_probability ≤ 0.3 AND amount ≤ $1M | Auto-cleared — the one state where the tool acts without human involvement |

**The hard stop:** no transaction with amount > $1,000,000 can land in
BLOCK (auto-clear) regardless of fraud_probability. In this run, 158
transactions were rerouted from BLOCK to FLAG by this rule. Post-hoc
verification confirmed zero transactions over $1M remain in BLOCK
(`hard_stop holds: True`). The hard stop is not decorative — it fired
on 158 real transactions.

**Why non-negotiable here:** auto-clearing a $1M+ transaction is a
committed, non-recoverable action (the transaction clears, funds move).
The cost of a false negative at that scale — missing real fraud on a
million-dollar transfer — is not recoverable from a downstream audit.
Running this unattended is exactly the "it ran unattended and moved the
resource" failure the assignment warns against.

**Delegation map:**

| Decision step | Tool decides | Human decides |
|---|---|---|
| GIGO gate | Whether transaction passes field-level consistency rules | Whether the gate's five rules are the right rules for this organization |
| Fraud scoring | fraud_probability, uncertainty, CI bounds | Whether the features used reflect the actual fraud patterns relevant to this org |
| Recommendation | INVESTIGATE / MONITOR / CLEAR | Whether to act on the recommendation before checking the hard-stop state |
| Hard stop | Structurally blocks auto-clear for >$1M transactions | Reviews all flagged high-amount transactions before clearance |
| Final action | None — tool never clears or blocks autonomously above $1M | All high-stakes clearances require explicit human approval |

---

## Uncertainty Communication

**Chart:** `reports/uncertainty_chart.png` — four fraud-probability buckets,
colored by recommendation, with error bars derived from 90% CI re-bucketing
(not invented — computed from each transaction's real `ci_lower_90` and
`ci_upper_90` from `engine.py`).

| Bucket | Recommendation | Count | CI lower rebucket | CI upper rebucket |
|---|---|---|---|---|
| 0–0.1 | CLEAR | 8,986 | 8,988 | 8,983 |
| 0.1–0.3 | MONITOR | 9 | 7 | 12 |
| 0.3–0.7 | INVESTIGATE | 7 | 7 | 7 |
| 0.7–1.0 | INVESTIGATE | 992 | 992 | 992 |

**Plain-language sentence a non-specialist would trust:**
"This tool is good at telling you 'this transaction looks nothing like
the fraud patterns we've seen before' — the CLEAR bucket is large, stable,
and barely moves under uncertainty, so trusting it to deprioritize the
bulk of obviously-routine transactions is reasonable. But any
recommendation near the decision boundaries (0.1, 0.3, or 0.7) should
be treated as uncertain and reviewed by a human before acting."

**Where I would NOT trust this tool:**
1. For any transaction near a decision boundary — the adversarial test
   showed a 10% amount change alone flipped 22% of TRANSFER recommendations.
2. As evidence that investigating a transaction prevents fraud — the causal
   analysis explicitly could not establish this.
3. As a fairness-neutral tool — the bias audit found a demographic parity
   gap of 0.3368 by transaction type.
4. On real-world fraud patterns — PaySim's near-deterministic structure
   produces AUC of 0.9979; real fraud is messier and this accuracy would
   not transfer.

---

## Frictional Journal Summary

**Pre-registered predictions (2026-07-28, 11:05 AM, before any code):**

| Prediction | Confidence | Result |
|---|---|---|
| Amount dominates SHAP | 75% | PARTIALLY WRONG — amount_to_balance_ratio dominated (10× larger than raw amount) |
| TRANSFER flagged disproportionately | 60% | CORRECT — DP gap 0.3368 |
| Rung 3 cannot compute | 90% | CORRECT |
| Hard stop reveals design failure | 50% | CORRECT — zero threshold-shift flips reveals near-deterministic behavior, not robustness |

**The finding not on the list:** the 10% TRANSFER amount inflation
decreased INVESTIGATE recommendations (477 → 232) rather than
increasing them. I expected inflation to push more transactions over
the threshold. The actual response was non-linear — the model's
marginal INVESTIGATE cases shifted to MONITOR rather than
higher-confidence cases shifting up. This was the most important
adversarial finding and the one I did not predict.

Full frictional journal: `reports/frictional_journal.md`

---

## Reproducing this report

```bash
cd E:\Assignment7Audit
python run_all.py
```

Runs all 9 scripts in order in ~25 seconds. Output is deterministic
(`random_state=42` throughout). All numbers in this report were pulled
directly from the terminal output of `run_all.py` run on 2026-07-28 —
nothing was invented.

Full pipeline summary from the latest run (after Improvements 1–5):
```
[OK] Sample data (10.9s)
[OK] GIGO gate (0.9s)
[OK] Engine (train + score) (3.3s)
[OK] Bias audit (1.7s)
[OK] Explainability (SHAP) (3.8s)
[OK] Causal analysis (0.6s)
[OK] Adversarial test (1.7s)
[OK] Delegation gate (1.6s)
[OK] Uncertainty communication (1.0s)
All scripts completed successfully.
```

All core numbers (accuracy, AUC, bias gaps, SHAP ranking, hard-stop counts)
are unchanged from the pre-improvement run — Improvements 1–5 added new
reported quantities and analyses on top of the existing pipeline; they did
not change the model, the data, or any existing metric.
