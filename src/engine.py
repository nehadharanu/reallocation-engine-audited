"""
Script 3: The reallocation engine.

Trains a RandomForestClassifier on gated transactions to predict isFraud,
scores the held-out test set with fraud_probability, attaches an uncertainty
estimate (std of per-tree predictions) and a 90% bootstrap confidence
interval (100 resamples of the trees), and buckets each transaction into a
recommendation: INVESTIGATE / MONITOR / CLEAR.

OBJECTIVE (stated plainly): Rank flagged transactions by estimated fraud
probability to allocate the next investigator slot to the highest-risk,
gate-cleared transaction.

WHAT THIS OBJECTIVE LEAVES OUT: the cost of false positives on legitimate
customers, demographic disparities in who gets flagged, feedback-loop bias
(only investigated transactions ever get confirmed labels, so the model is
trained on a sample that is not representative of all transactions), and
whether blocking a transaction actually prevents fraud or just delays it.
None of this is measured by accuracy/precision/recall/AUC below.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

IN_PATH = "data/gated_transactions.csv"
OUT_PATH = "data/scored_transactions.csv"
MODEL_DIR = "models"
RANDOM_STATE = 42
N_ESTIMATORS = 200
N_BOOTSTRAP = 100

FEATURE_COLS = [
    "amount",
    "balance_change_orig",
    "balance_change_dest",
    "amount_to_balance_ratio",
    "zero_balance_after",
    "type_encoded",
]

def recommend(p):
    if p > 0.3:
        return "INVESTIGATE"
    elif p > 0.1:
        return "MONITOR"
    else:
        return "CLEAR"

def main():
    df = pd.read_csv(IN_PATH)

    encoder = LabelEncoder()
    df["type_encoded"] = encoder.fit_transform(df["type"])

    X = df[FEATURE_COLS]
    y = df["isFraud"]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train, y_train)

    proba_test = model.predict_proba(X_test)[:, 1]
    pred_test = (proba_test > 0.5).astype(int)

    accuracy = accuracy_score(y_test, pred_test)
    precision = precision_score(y_test, pred_test, zero_division=0)
    recall = recall_score(y_test, pred_test, zero_division=0)
    auc = roc_auc_score(y_test, proba_test)

    # Per-tree predictions on the test set: shape (n_trees, n_test)
    X_test_arr = X_test.to_numpy()
    tree_probs = np.array(
        [tree.predict_proba(X_test_arr)[:, 1] for tree in model.estimators_]
    )
    uncertainty = tree_probs.std(axis=0)

    # 90% CI via bootstrap resampling of the trees (100 resamples)
    rng = np.random.default_rng(RANDOM_STATE)
    n_trees = tree_probs.shape[0]
    boot_means = np.empty((N_BOOTSTRAP, tree_probs.shape[1]))
    for b in range(N_BOOTSTRAP):
        sample_idx = rng.integers(0, n_trees, size=n_trees)
        boot_means[b] = tree_probs[sample_idx].mean(axis=0)
    ci_lower = np.percentile(boot_means, 5, axis=0)
    ci_upper = np.percentile(boot_means, 95, axis=0)

    scored = df.loc[idx_test].copy().reset_index(drop=True)
    scored["fraud_probability"] = proba_test
    scored["uncertainty"] = uncertainty
    scored["ci_lower_90"] = ci_lower
    scored["ci_upper_90"] = ci_upper
    scored["recommendation"] = [recommend(p) for p in proba_test]
    # attention_reallocated: 1 if this transaction is the recipient of a moved
    # investigator-attention slot this cycle (INVESTIGATE), 0 otherwise. This is
    # the quantified move, not just a categorical bucket - downstream scripts
    # can read this column directly instead of re-deriving it from recommendation.
    scored["attention_reallocated"] = (scored["recommendation"] == "INVESTIGATE").astype(int)

    os.makedirs("data", exist_ok=True)
    scored.to_csv(OUT_PATH, index=False)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "rf_model.joblib"))
    joblib.dump(encoder, os.path.join(MODEL_DIR, "type_encoder.joblib"))
    joblib.dump(FEATURE_COLS, os.path.join(MODEL_DIR, "feature_cols.joblib"))

    rec_counts = scored["recommendation"].value_counts().to_dict()

    total_n = len(scored)
    move_n = int(scored["attention_reallocated"].sum())
    move_pct = move_n / total_n * 100
    clear_n = rec_counts.get("CLEAR", 0)
    clear_pct = clear_n / total_n * 100
    reallocation_line = (
        f"This cycle: move {move_n} of {total_n} transactions' investigator-attention "
        f"from routine clearance to priority review ({move_pct:.1f}% of queue). "
        f"Remaining {clear_n} transactions ({clear_pct:.1f}%) deprioritized - "
        f"investigator slots reallocated away from them."
    )

    print(f"Train rows: {len(X_train)}  Test rows: {len(X_test)}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"AUC:       {auc:.4f}")
    print("Recommendation distribution:")
    for rec in ["INVESTIGATE", "MONITOR", "CLEAR"]:
        print(f"  {rec}: {rec_counts.get(rec, 0)}")
    print(f"Saved scored transactions to: {OUT_PATH}")
    print(f"Saved model + encoder to: {MODEL_DIR}/")
    print()
    print("OBJECTIVE: Rank flagged transactions by estimated fraud probability")
    print("to allocate the next investigator slot to the highest-risk,")
    print("gate-cleared transaction.")
    print()
    print("WHAT THIS LEAVES OUT: cost of false positives on legitimate")
    print("customers, demographic disparities in who gets flagged, feedback-loop")
    print("bias (only investigated transactions get confirmed labels), and")
    print("whether blocking a transaction prevents fraud or just delays it.")
    print()
    print(reallocation_line)

if __name__ == "__main__":
    main()
