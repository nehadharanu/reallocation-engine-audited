# Explainability Report (SHAP)

SHAP computed on a sample of 1000 test transactions using TreeExplainer.

## Mean absolute SHAP value per feature (ranked)

| Feature | Mean |SHAP| |
|---|---|
| amount_to_balance_ratio | 0.132831 |
| zero_balance_after | 0.083792 |
| balance_change_orig | 0.044637 |
| type_encoded | 0.021355 |
| amount | 0.013276 |
| balance_change_dest | 0.010117 |

**Dominant feature: `amount_to_balance_ratio`.**

Summary plot saved to: reports/shap_summary.png

## The misleading case

Transaction (test-set row, isFraud=0, recommendation=INVESTIGATE):

| Field | Value |
|---|---|
| type | TRANSFER |
| amount | 1939569.24 |
| oldbalanceOrg | 0.0 |
| newbalanceOrig | 0.0 |
| oldbalanceDest | 4303564.8 |
| newbalanceDest | 6243134.04 |
| balance_change_orig | 0.0 |
| balance_change_dest | 1939569.24 |
| amount_to_balance_ratio | 1939569.24 |
| zero_balance_after | 1 |
| fraud_probability | 0.455 |
| uncertainty | 0.4979708826829138 |

**SHAP values for this transaction (positive/fraud class):**

| Feature | SHAP value |
|---|---|
| amount | +0.1763 |
| type_encoded | +0.0958 |
| balance_change_dest | +0.0641 |
| zero_balance_after | +0.0386 |
| amount_to_balance_ratio | -0.0136 |
| balance_change_orig | -0.0061 |

Base value (explainer.expected_value, positive class): 0.0998

**Why this is technically accurate but practically misleading:** the model is not wrong about what it learned — `amount` (and amount-derived features) genuinely correlate with fraud in this sample, and the SHAP values above show that correctly. But for *this specific transaction*, the large positive SHAP contribution from amount pushed the score into the INVESTIGATE band even though the transaction is confirmed legitimate (isFraud=0). An investigator handed only the fraud_probability number, without this SHAP breakdown, would have no way to see that the recommendation is being driven almost entirely by transaction size rather than any behavioral fraud signal. In production this means large, legitimate transactions systematically consume investigator attention that a smaller genuinely-suspicious transaction may have used more productively.

## Is this a single anecdote, or a systematic pattern?

Among all 6 legitimate transactions (isFraud=0) that received an INVESTIGATE recommendation (all false positives in this test set), **6 of 6 (100.0%) share the same degenerate pattern as the named case above: `zero_balance_after=1` AND `amount_to_balance_ratio > 1000`** (i.e. the sender's account goes to exactly zero and the amount-to-balance ratio is driven to an extreme value by a near-zero denominator, not by a genuinely large amount relative to typical account activity).

Every single false positive in this test set follows this pattern - it is not a single anecdote, it is the systematic failure mode of this feature set: any account that empties to zero as its normal end-state (a full withdrawal, a final settlement, an account closure) will trigger this signal regardless of whether fraud actually occurred.
