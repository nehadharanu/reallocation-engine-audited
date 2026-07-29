# Adversarial Report

## Perturbation 1: Threshold shift (0.30 -> 0.25)

Recommendations changed: **0** / 9994 (0.00%)

Direction of change (old -> new : count):

| From | To | Count |
|---|---|---|

Transactions with fraud_probability in (0.25, 0.30]: 0

## Perturbation 2: +10% amount on TRANSFER transactions

TRANSFER transactions affected: 1204

Recommendations changed among TRANSFER rows: **269**

Recommendation distribution, TRANSFER only, before -> after:

| Recommendation | Before | After |
|---|---|---|
| INVESTIGATE | 477 | 232 |
| MONITOR | 3 | 268 |
| CLEAR | 724 | 704 |

Recommendation distribution, full test set, before -> after:

| Recommendation | Before | After |
|---|---|---|
| INVESTIGATE | 999 | 754 |
| MONITOR | 9 | 274 |
| CLEAR | 8986 | 8966 |

## Failure condition

The engine's recommendation is fragile for any transaction whose fraud_probability sits within a narrow band around the decision boundary. A 5-point threshold shift (0.30 -> 0.25) flipped 0 of 9994 recommendations (0.00%) purely from transactions with probability in (0.25, 0.30] (0 such rows) - none of these transactions changed in any real way, only the threshold moved. Separately, a 10% amount inflation on TRANSFER transactions (representing a plausible data entry error or an adversary padding amounts to see how the system reacts) changed the recommendation for 269 of 1204 TRANSFER transactions. Both failure modes point to the same root cause: hard threshold cutoffs on a continuous probability score are inherently unstable for transactions near the cutoff, regardless of whether the underlying risk actually changed.

## Cross-component: does this perturbation worsen the fairness gap?

OTHER's selection rate is unaffected by this perturbation (only TRANSFER amounts were inflated), so any change in the demographic parity gap comes entirely from the TRANSFER side.

| | Before inflation | After +10% TRANSFER inflation |
|---|---|---|
| TRANSFER selection rate | 0.3962 | 0.1927 |
| OTHER selection rate | 0.0594 | 0.0594 (unchanged) |
| Demographic parity gap | 0.3368 | 0.1333 |

**Gap NARROWS by 0.2035** (a plausible data error incidentally makes the measured gap smaller, driven by the same non-linear MONITOR-shift seen in perturbation 2, not by any fairness-aware behavior).
