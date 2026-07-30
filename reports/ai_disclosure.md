# AI Use Disclosure

**Tool(s) used:** Claude (Sonnet, claude.ai chat interface, with code execution).

**Portions assisted:** All source code (`src/*.py`), all generated reports
(`reports/*.md`), the README, the frictional journal reflection section,
and the drafting of this disclosure were produced through conversation with
Claude. The frictional journal prediction was written by me before any
code was written or any data was explored. The reflection was drafted by
Claude from my real terminal output and then corrected by me before
finalizing. The domain choice, the decision to use transaction type as
the protected axis for the bias audit, the choice of adversarial
perturbations to run, and the decision to connect the adversarial and
fairness findings into a cross-component analysis were all made by me.

**How used:** Claude wrote all the Python code and report text in this
repository, one component at a time, in dependency order. Each script was
run immediately after being written, and its actual terminal output was
inspected and pasted back before moving to the next one. The validation
report was drafted from that real output after the full pipeline ran clean.
No numbers were described in advance or invented before the scripts ran.

**What I changed:** The most consequential correction was in the frictional
journal reflection. Claude's first draft framed the adversarial finding
(threshold shift flipping zero recommendations) as evidence of robustness.
I corrected this. Zero flips is not robustness. It is evidence that PaySim's
fraud patterns are near-deterministic, leaving essentially no transactions
in the boundary band to flip. A real deployment would have many transactions
there and would be far more fragile. Claude produced the correct number.
I produced the correct reading of what that number means for real-world
trust in the tool.

I also directed the cross-component analysis. After Claude treated the
adversarial test and the bias audit separately, I asked whether the
adversarial perturbation also changed the fairness gap. Claude had not
proposed connecting them. I asked for it because I wanted to know whether
the two failures compounded each other. The finding (gap narrows for the
wrong reason, a recall failure disguised as a fairness improvement) came
from that question.

**What the AI could not do:**

When the adversarial test showed that a 10% amount inflation on TRANSFER
transactions decreased INVESTIGATE recommendations from 477 to 232 rather
than increasing them, Claude flagged the result as counterintuitive and
reported the numbers correctly. I understood why it happened. Inflating
amount also inflates amount-to-balance-ratio, but for transactions already
sitting deep inside the INVESTIGATE band at near-saturated probability
values, the model response is non-linear near saturation. The marginal
cases near the 0.3 threshold shifted down to MONITOR rather than
higher-confidence cases shifting further up.

Claude could compute the output and notice something unexpected happened.
It could not explain the mechanism without knowing how RandomForest
probability estimates behave near decision boundaries at high-confidence
regions. That required understanding both what the adversarial test was
measuring and how tree ensemble probabilities saturate in that regime.
The finding existed in the numbers. The interpretation that makes it
useful rather than just surprising was mine to provide.
