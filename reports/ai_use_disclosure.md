# AI Use Disclosure — The Reallocation Engine, Audited

**Tool(s) used:** Claude (Anthropic, Sonnet 4.6), via claude.ai chat interface.

**Portions assisted:** All source code (`src/*.py`), the main validation
report (`Dharanu_Neha_ReallocationEngine.md`), the README, the frictional
journal reflection section, and this disclosure were produced through
conversation with Claude. The frictional journal prediction was written
by me before any code existed.

**How used:** I provided the assignment rubric, the PaySim dataset location,
and the repo structure I wanted. Claude wrote each script in order, one
rubric component at a time. I ran each script immediately after it was
written and pasted the real terminal output back. Claude then drafted
reports and the main validation document from that actual output — never
from invented or expected numbers. The pipeline ran clean end-to-end
(`python run_all.py`, all 9 scripts `[OK]`) before the main report was
drafted.

**What I changed:** I reviewed every script and report for accuracy against
the actual terminal output. The most consequential correction was in the
frictional journal reflection: Claude's first draft of the reflection
framed the adversarial finding (threshold shift flipping zero
recommendations) as straightforward robustness. I corrected this —
zero flips on a threshold shift is not evidence of robustness, it is
evidence that PaySim's fraud patterns are near-deterministic and the
model assigns near-0 or near-1 probabilities for almost every transaction.
A real-world deployment would have many transactions in the boundary band
and would be far more fragile. Claude produced the technically correct
number (0 flips); I provided the domain judgment about what that number
actually means for real-world trust in this tool.

**What the AI could not do:** When the adversarial test showed that a 10%
amount inflation on TRANSFER transactions decreased INVESTIGATE
recommendations (from 477 to 232) rather than increasing them, Claude
reported the numbers correctly but described the finding as "counterintuitive."
I recognized why it happened: inflating `amount` inflates
`amount_to_balance_ratio`, but for transactions already at very high ratio
values (already deep in the INVESTIGATE band), the model's probability
surface is non-linear near saturation — the marginal cases near the 0.3
threshold shifted to MONITOR rather than the high-confidence cases shifting
further up. Claude could compute the output; it could not explain the
non-linear mechanism from the model's own probability surface without
domain knowledge about how tree ensembles behave near decision boundaries
at high-confidence regions. That explanation required me to understand both
what the adversarial test was measuring and how the RandomForest probability
estimates behave in that regime — the AI produced the finding, I produced
the interpretation that makes it useful rather than just surprising.
