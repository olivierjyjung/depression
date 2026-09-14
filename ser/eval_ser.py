#!/usr/bin/env python3
"""Turn SER embeddings into per-speaker scores and check the two success criteria.

Criteria (DECISIONS.md 2026-09-09, fixed before looking at any result):
  Gate 1    - does test AUC beat 0.626, the existing whisper (ASR) embedding?
  Adoption  - does it improve on the current best combination
              (text + latency, test AUC 0.881 / F1 0.797)?
              (requires a per-speaker score file for text and latency)

Overfitting control: the L2 logistic regularization strength is chosen by
cross-validation inside train only. dev/test never enter the selection.
"""
import argparse, json, os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score

BASELINE_EMB_TEST_AUC = 0.626      # whisper (ASR) encoder emb from the 08-21 table
CS = np.logspace(-4, 2, 13)


def load_labels(meta):
    tr = pd.read_csv(f"{meta}/train_split_Depression_AVEC2017.csv")
    dv = pd.read_csv(f"{meta}/dev_split_Depression_AVEC2017.csv")
    te = pd.read_csv(f"{meta}/full_test_split.csv").rename(
        columns={"PHQ_Binary": "PHQ8_Binary", "PHQ_Score": "PHQ8_Score"})
    rows = []
    for name, d in [("train", tr), ("dev", dv), ("test", te)]:
        for _, r in d.iterrows():
            rows.append(dict(pid=int(r.Participant_ID), split=name, y=int(r.PHQ8_Binary)))
    return pd.DataFrame(rows)


def best_threshold(y, s):
    """Threshold maximizing F1 on train OOF scores, then applied as-is to dev/test."""
    cand = np.unique(np.round(s, 4))
    return max(cand, key=lambda t: f1_score(y, (s >= t).astype(int), zero_division=0))


def score_embedding(npz_path, lab, seed=0):
    Z = np.load(npz_path)
    lab = lab[lab.pid.astype(str).isin(Z.files)].copy()
    X = np.stack([Z[str(p)] for p in lab.pid])
    m = {k: (lab.split == k).values for k in ["train", "dev", "test"]}
    Xtr, ytr = X[m["train"]], lab.y.values[m["train"]]

    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    gs = GridSearchCV(pipe, {"logisticregression__C": CS}, scoring="roc_auc", cv=cv, n_jobs=-1)
    gs.fit(Xtr, ytr)                       # C is selected inside train only
    C = gs.best_params_["logisticregression__C"]

    final = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
    oof = cross_val_predict(final, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
    final.fit(Xtr, ytr)
    s = np.empty(len(lab)); s[m["train"]] = oof
    for k in ["dev", "test"]:
        s[m[k]] = final.predict_proba(X[m[k]])[:, 1]
    thr = best_threshold(ytr, oof)

    out = {"C": float(C), "dim": int(X.shape[1]), "cv_auc_train": float(gs.best_score_),
           "threshold": float(thr)}
    for k in ["train", "dev", "test"]:
        y, sc = lab.y.values[m[k]], s[m[k]]
        out[f"auc_{k}"] = float(roc_auc_score(y, sc))
        out[f"f1_{k}"] = float(f1_score(y, (sc >= thr).astype(int), zero_division=0))
    return out, pd.DataFrame({"pid": lab.pid.values, "split": lab.split.values, "ser": s})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", nargs="+", required=True)
    ap.add_argument("--data", default="/data/daic")
    ap.add_argument("--fusion", default="", help="per-speaker text/latency score csv (pid,text,lat)")
    ap.add_argument("--out", default="/work/eval")
    a = ap.parse_args()

    lab = load_labels(f"{a.data}/meta")
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for p in a.emb:
        tag = os.path.basename(p).replace(".npz", "")
        r, s = score_embedding(p, lab)
        r["name"] = tag
        r["gate_pass"] = r["auc_test"] > BASELINE_EMB_TEST_AUC
        s.to_csv(f"{a.out}/{tag}_scores.csv", index=False)
        rows.append(r)

    df = pd.DataFrame(rows)[["name", "dim", "C", "cv_auc_train", "auc_dev", "auc_test",
                             "f1_dev", "f1_test", "gate_pass"]]
    print("\n=== SER alone (gate 1: test AUC > %.3f) ===" % BASELINE_EMB_TEST_AUC)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if a.fusion and os.path.exists(a.fusion):
        print("\n=== Stage 2: stacking on the existing combination ===")
        print("(not implemented yet - pending the score file format)")
    else:
        print("\nStage 2 (fusion gain) deferred - no per-speaker text/latency score file")
    df.to_csv(f"{a.out}/summary.csv", index=False)
    print(f"\nsaved: {a.out}/summary.csv")


if __name__ == "__main__":
    main()
