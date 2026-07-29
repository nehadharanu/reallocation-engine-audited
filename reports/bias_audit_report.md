# Bias Audit Report

Protected axis: transaction TYPE, collapsed to TRANSFER vs OTHER.

## Group summary

| Group | n | Base fraud rate | Selection rate (INVESTIGATE) | TPR | FPR |
|---|---|---|---|---|---|
| TRANSFER | 1204 | 0.3937 | 0.3962 | 1.0000 | 0.0041 |
| OTHER | 8790 | 0.0596 | 0.0594 | 0.9905 | 0.0004 |

Base fraud rate gap (TRANSFER - OTHER): 0.3341

## Fairness definition 1: Demographic parity

Selection rate gap between groups: **0.3368** (TRANSFER: 0.3962, OTHER: 0.0594)

## Fairness definition 2: Equalized odds (TPR component)

True positive rate gap between groups: **0.0095** (TRANSFER: 1.0000, OTHER: 0.9905)

False positive rate gap between groups (for completeness): **0.0037**

## The tradeoff, quantified

The two groups have different base fraud rates (0.3937 vs 0.0596, a gap of 0.3341). This is the textbook condition under which demographic parity and equalized odds are mathematically incompatible for a non-trivial classifier: if selection rates are forced equal across groups with different true fraud rates, the group with the lower true rate will necessarily have a higher false positive rate (or the higher-rate group will have a lower true positive rate) than a threshold optimized per-group would produce. Our engine currently uses one global threshold (0.3), which produces a demographic parity gap of 0.3368 and an equalized-odds (TPR) gap of 0.0095.

## Highest-leverage intervention point

Highest-leverage intervention point: the DECISION THRESHOLD applied to fraud_probability, not the underlying model. Because base fraud rates differ sharply between TRANSFER and OTHER transaction types in this sample (0.3937 vs 0.0596), a single global threshold (0.3) cannot equalize both selection rate and true positive rate across groups simultaneously. The threshold is also the cheapest point to intervene on: it requires no retraining, is auditable, and can be set per-group if the organization decides which fairness definition it is willing to prioritize.

## Fairlearn cross-check

fairlearn was available; MetricFrame results:

| is_transfer   |   selection_rate |      tpr |
|:--------------|-----------------:|---------:|
| OTHER         |        0.0593857 | 0.990458 |
| TRANSFER      |        0.396179  | 1        |
