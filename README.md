# The Reallocation Engine

A fraud-investigation reallocation engine built on the PaySim synthetic
financial fraud dataset. The resource being reallocated is **investigator
attention slots**: given limited human review capacity, which flagged
transactions get reviewed first?

**Objective (plain sentence):** Rank flagged transactions by estimated
fraud probability to allocate the next investigator slot to the
highest-risk, gate-cleared transaction.

**What the objective leaves out:** the cost of false positives on
legitimate customers, demographic disparities in who gets flagged,
feedback-loop bias (only investigated transactions ever get confirmed
labels, so the model trains on a non-representative sample), and whether
blocking a transaction actually prevents fraud or just delays it.

## Setup

1. Clone or copy this repo anywhere on your machine.
2. Download the [PaySim synthetic fraud dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)
   and place the CSV in the project root as `PS_20174392719_1491204439457_log.csv`
   (same folder as `run_all.py`).
3. Install dependencies: `pip install pandas numpy scikit-learn matplotlib shap fairlearn joblib tabulate`

## Pipeline

Run the whole thing from the project root with:

```
python run_all.py
```

Or run each stage individually, in order:

| # | Script | Produces |
|---|---|---|
| 1 | `src/sample_data.py` | `data/raw/paysim_sample.csv` - 50,000-row stratified sample (5,000 fraud) with engineered features |
| 2 | `src/gigo_gate.py` | `data/gated_transactions.csv`, `reports/gigo_gate_report.md` - data quality gate |
| 3 | `src/engine.py` | `data/scored_transactions.csv`, `models/` - trained RandomForest, fraud_probability, uncertainty, 90% bootstrap CI, recommendation |
| 4 | `src/bias_audit.py` | `reports/bias_audit_report.md` - demographic parity vs equalized odds by transaction type |
| 5 | `src/explainability.py` | `reports/explainability_report.md`, `reports/shap_summary.png` - SHAP feature attribution and a named misleading case |
| 6 | `src/causal_analysis.py` | `reports/causal_reasoning_report.md` - Pearl's ladder of causation (observation / intervention / counterfactual) |
| 7 | `src/adversarial_test.py` | `reports/adversarial_report.md` - threshold-shift and amount-inflation stress tests |
| 8 | `src/delegation_gate.py` | `reports/delegation_gate_report.md` - APPROVE / FLAG / BLOCK hard-stop states |
| 9 | `src/uncertainty_communication.py` | `reports/uncertainty_communication.md`, `reports/uncertainty_chart.png` - uncertainty-aware bucket chart |

## Repo structure

All paths below are relative to the project root (wherever you cloned this repo):

```
./
├── README.md
├── run_all.py
├── PS_20174392719_1491204439457_log.csv   (full raw PaySim dataset - place here yourself, not sampled)
├── data/
│   ├── raw/paysim_sample.csv
│   ├── gated_transactions.csv
│   └── scored_transactions.csv
├── models/
│   ├── rf_model.joblib
│   ├── type_encoder.joblib
│   └── feature_cols.joblib
├── reports/
│   ├── gigo_gate_report.md
│   ├── bias_audit_report.md
│   ├── explainability_report.md
│   ├── shap_summary.png
│   ├── causal_reasoning_report.md
│   ├── adversarial_report.md
│   ├── delegation_gate_report.md
│   ├── uncertainty_communication.md
│   ├── uncertainty_chart.png
│   └── frictional_journal.md   (pre-registered predictions, written before this build)
└── src/
    ├── sample_data.py
    ├── gigo_gate.py
    ├── engine.py
    ├── bias_audit.py
    ├── explainability.py
    ├── causal_analysis.py
    ├── adversarial_test.py
    ├── delegation_gate.py
    └── uncertainty_communication.py
```

All scripts in `src/` read and write using relative paths (`data/...`,
`reports/...`, `models/...`) and assume they are run from the project root -
no absolute paths are hardcoded anywhere in the pipeline.

## Headline findings from this run

- **Model is near-perfect on held-out data** (accuracy 0.9991, AUC 0.9979) -
  in PaySim, fraud is close to deterministic from balance changes, which is
  a property of the simulator, not evidence the engine would generalize to
  real-world fraud patterns.
- **SHAP dominant feature is `amount_to_balance_ratio`**, not raw `amount` -
  the pre-registered prediction (`reports/frictional_journal.md`) expected
  raw amount to dominate; the actual top feature by mean |SHAP| is the
  ratio, though raw `amount` was still the single largest driver for the
  specific misleading high-amount false-positive case found.
- **Large demographic parity gap by transaction type** (0.3368) despite a
  small equalized-odds/TPR gap (0.0095) - a concrete, quantified
  demonstration that both fairness definitions cannot hold at once when
  base fraud rates differ this much between TRANSFER and other types.
- **Threshold shift (0.30 -> 0.25) flipped zero recommendations**, but a
  10% amount inflation on TRANSFER transactions flipped 269 of 1,204
  (22%) - and counterintuitively *decreased* INVESTIGATE volume. The
  engine is far more sensitive to a plausible data/amount perturbation
  than to a routine threshold tweak.
- **Hard stop verified working**: 158 transactions over $1,000,000 were
  rerouted from auto-clear (BLOCK) to mandatory human review (FLAG); a
  post-hoc check confirmed zero transactions over $1,000,000 ever landed
  in BLOCK.
- **Rung 3 (counterfactual) is explicitly "CANNOT COMPUTE"** - no invented
  causal number, per the assignment's core requirement.

All numbers above are pulled directly from the terminal output of
`run_all.py`; see the individual reports in `reports/` for full detail and
methodology.
