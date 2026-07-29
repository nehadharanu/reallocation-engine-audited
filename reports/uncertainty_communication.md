# Uncertainty Communication Report

Chart saved to: reports/uncertainty_chart.png

## Bucket counts (point estimate, and 90% CI re-bucketing range)

| Bucket | Recommendation | Count | Count if all at CI lower bound | Count if all at CI upper bound |
|---|---|---|---|---|
| 0-0.1 | CLEAR | 8986 | 8988 | 8983 |
| 0.1-0.3 | MONITOR | 9 | 7 | 12 |
| 0.3-0.7 | INVESTIGATE | 7 | 7 | 7 |
| 0.7-1.0 | INVESTIGATE | 992 | 992 | 992 |

## Plain language

**What a non-specialist would trust: the tool is good at telling you "this transaction looks nothing like the fraud patterns we've seen before" (the CLEAR bucket is large, stable, and has a narrow CI re-bucketing range), so trusting it to deprioritize the bulk of obviously-routine transactions is reasonable.**

**Where I would NOT trust this tool: (1) for any single transaction near a decision boundary (fraud_probability just above or below 0.1, 0.3, or 0.7) - the adversarial test showed a 10% amount change alone can flip the recommendation; (2) as evidence that investigating a transaction will actually prevent fraud - the causal analysis could not establish that; (3) as a fairness-neutral tool - the bias audit found a large demographic-parity gap by transaction type; (4) on any transaction type or amount range not well represented in the training sample, since the model has never seen it.**

