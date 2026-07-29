# Causal Reasoning Report

## Rung 1 — Observation

- corr(fraud_probability, isFraud) = 0.9959
- corr(amount, isFraud) = 0.4216
- corr(is_transfer, isFraud) = 0.3627
- Cramer's V(type, isFraud) = 0.4232 (association across all 5 type categories, not just TRANSFER)

## Rung 2 — Intervention (named confounders)

**Confounder 1 — type confounded with amount:**

Transaction type is confounded with amount: in this sample, mean TRANSFER amount is 1,099,372.73 vs 199,307.85 for all other types combined (5.52x larger). Since the engine's top SHAP feature involves amount, some of the apparent type -> fraud association could be routing through amount. If we intervened to equalize transaction amounts across types, the observed type/fraud correlation would likely shrink, and we cannot tell from this observational data how much of the correlation is genuinely about type versus genuinely about size.

**Confounder 2 — survivorship bias in the fraud label:**

The fraud label itself suffers survivorship / labeling bias: isFraud in PaySim marks transactions the simulation generated as fraud, not transactions that were investigated and confirmed. In a real deployment, isFraud=0 would actually mean 'not caught,' not 'confirmed legitimate.' Because only a fraction of flagged transactions are ever reviewed by a human, any future retraining on outcomes would only ever see labels for the subset the engine chose to surface — a feedback loop that reinforces whatever the current model already believes, and could make the true underlying correlation (if we could ever observe it) different from what is measured here.

## Rung 3 — Counterfactual

Counterfactual question for this transaction (type=CASH_OUT, amount=806,850.06, fraud_probability=1.0000, isFraud=1): 'If an investigator had NOT reviewed this transaction (i.e. it had been allocated to CLEAR instead of INVESTIGATE), would the fraud outcome have been different?'

**Answer: CANNOT COMPUTE. Three concrete reasons: (1) No outcome data exists for the counterfactual arm — PaySim's isFraud is a simulated ground-truth label, not an investigation outcome, so we have no record of what would have happened to THIS transaction under a different investigator-attention allocation. (2) SUTVA (Stable Unit Treatment Value Assumption) is violated: investigator attention is a shared, finite resource, so whether this transaction is investigated changes how much attention is available for every other transaction in the queue. The 'treatment' on one unit is not independent of treatment assignment on other units, which is a precondition for a valid counterfactual estimate. (3) No randomization: transactions were not randomly assigned to investigate/clear, so any comparison between investigated and uninvestigated transactions here confounds the investigation decision itself with the very features (amount, type) that were used to make that decision. We are not going to invent a probability or an estimated causal effect to fill this gap.**

## Verdict

Honest verdict: what this engine produces is correlation dressed as causation. The fraud_probability score is a strong statistical association between transaction features and a fraud label (corr=0.9959 with isFraud in this sample), and reallocating investigator attention toward higher-scoring transactions is a reasonable POLICY under that association. But nothing in this pipeline demonstrates that investigating higher-scoring transactions CAUSES more fraud to be caught, prevented, or reduced, because we never observe the counterfactual (what happens to a transaction that isn't investigated) and never randomize investigation assignment. Treat the ranking as a triage heuristic backed by association, not as a validated causal intervention.
