# Frictional Journal

\_Written before any code was run or data was explored.
Timestamp is real — 2026-07-28, 11:05 AM

---

## Prediction (before the build)

**Timestamp:** 2026-07-28, 11:05 AM

I am building a fraud investigation reallocation engine on the
PaySim synthetic financial dataset. The engine reallocates finite
investigator attention across flagged transactions — given limited
capacity, which flagged transactions get human review first.

**Prediction 1 — Transaction amount dominates the fraud score.**
I expect SHAP to show that transaction amount is the single
largest driver of the fraud prediction, more than transaction
type or balance changes. The tool will essentially be a
"flag expensive transactions first" engine dressed up as
sophisticated fraud detection.
Confidence: 75%

**Prediction 2 — TRANSFER transactions get flagged at higher rates.**
I expect the bias audit to show that TRANSFER type transactions
are flagged disproportionately relative to their actual fraud
base rate, creating disparate impact on that transaction category.
Confidence: 60%

**Prediction 3 — Rung 3 will be structurally blocked.**
I expect the causal counterfactual to be uncomputable — we have
no outcome data for uninvestigated transactions. The honest
answer will be "cannot compute" not a number.
Confidence: 90%

**Prediction 4 — The hard stop gate will reveal something
embarrassing about the tool's own design.**
I expect at least one hard stop to fire too broadly or not
broadly enough — revealing a design failure I didn't anticipate
before building it.
Confidence: 50%

**Overall causal validity expectation:** Low. The tool optimizes
an observational correlation between transaction features and
fraud labels. I cannot claim reallocating investigator attention
to high-scoring transactions causes more fraud to be caught.

**What I am least confident about:** whether the surprising
finding will be in the bias audit, the adversarial test, or
somewhere I haven't thought of yet. In my experience the most
important failure is usually the one not on this list.

---

## Reflection (written immediately after the build)

**Timestamp:** 2026-07-28, after all scripts ran successfully

### What actually happened, against each prediction

**Prediction 1 — Transaction amount dominates the fraud score (75%):**
PARTIALLY WRONG — and wrong in an interesting way. I predicted raw
`amount` would be the dominant SHAP feature. The actual dominant feature
was `amount_to_balance_ratio` (mean |SHAP| = 0.1328 vs amount's 0.0133,
a 10x difference). The model did not learn "flag expensive transactions"
— it learned "flag transactions that drain the sender's account." When
`oldbalanceOrg` is near zero and `amount` is large, the ratio explodes,
and that behavioral signal (account drainage) is what the model weights
most heavily. My prediction was right about the direction (amount-related
features dominate over pure behavioral patterns) but wrong about which
amount feature and why. This matters: the tool is less naive than I
predicted, but the misleading case (a $1.93M legitimate TRANSFER from a
zero-balance account flagged INVESTIGATE) still holds for a different
reason than I expected.

**Prediction 2 — TRANSFER transactions get flagged at higher rates (60%):**
CORRECT — but the magnitude was larger than I expected.
Demographic parity gap: 0.3368 (TRANSFER selection rate 0.3962 vs OTHER
0.0594). I predicted the direction; I did not predict it would be this
large. The equalized-odds (TPR) gap was tiny (0.0095), which is the
textbook proof that demographic parity and equalized odds cannot both
hold when base fraud rates differ this sharply (0.3937 for TRANSFER vs
0.0596 for OTHER). I was right about the finding, wrong about its size.

**Prediction 3 — Rung 3 will be structurally blocked (90%):**
CORRECT. The causal analysis correctly returned CANNOT COMPUTE with
three concrete reasons: no outcome data for uninvestigated transactions,
SUTVA violation (investigator attention is shared and finite), and no
randomization of investigation assignment. No counterfactual number was
invented.

**Prediction 4 — Hard stop gate will reveal a design failure (50%):**
CORRECT, but not in the way I expected. I anticipated a gate firing too
broadly (like Saloni's Hard Stop 2 firing on 100% of the population).
What actually happened was subtler: the threshold-shift perturbation
(0.30 -> 0.25) flipped ZERO recommendations. This initially looks like
the tool is robust, but it is actually a finding about the model's
near-deterministic behavior — there are essentially no transactions in
the (0.25, 0.30] probability band because PaySim's fraud patterns are
so cleanly separable that the model assigns near-0 or near-1
probabilities with almost nothing in between. A real-world deployment
would have many transactions in that band and the tool would be far
more fragile than this clean result implies. The robust-looking number
is itself misleading.

### The thing I did not predict at all

The amount inflation perturbation (10% more on TRANSFER amounts) flipped
269 of 1,204 TRANSFER recommendations — and counterintuitively DECREASED
the INVESTIGATE count (from 477 to 232) rather than increasing it. I
expected inflation to push more transactions over the threshold. What
actually happened: inflating amount also inflated `amount_to_balance_ratio`,
which the model uses as the primary fraud signal. But for transactions
that were already at high ratio values, the model's response was not
linear — the already-high-confidence fraud cases stayed INVESTIGATE while
the marginal cases shifted in unexpected directions. I had not anticipated
that inflating an input could decrease the INVESTIGATE count for the very
group (TRANSFER) that is already most flagged. This is the most valuable
finding I did not put on my list.

### Where my prediction was wrong

I was right about the big-picture story (correlation dressed as causation,
TRANSFER disproportionately flagged, Rung 3 uncomputable) but wrong about
the mechanism. I predicted "amount dominates" and got "account-drainage
ratio dominates." I predicted "threshold shift would show fragility" and
got "threshold shift showed near-zero fragility while amount inflation
showed real fragility." The tool surprised me in the adversarial test,
not the explainability — which is the opposite of where I was looking.

### What this says about my calibration

I was well-calibrated on the causal-validity prediction (correlation not
causation) and on the TRANSFER bias prediction. I was over-confident
about which specific feature would dominate SHAP, and I entirely missed
that the adversarial finding would come from amount inflation rather than
threshold instability. The calibration lesson: I was right about where a
problem would live (amount-related features, TRANSFER disparity) but wrong
about the specific mechanism within those areas. This matches the pattern
in other examples — the most important failure is usually the one you
named but mis-described, not the one you named precisely.
