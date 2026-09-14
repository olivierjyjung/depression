#!/usr/bin/env python3
"""Was the 1024-dim embedding's failure about dimensionality?

neg_mean -- one scalar read off the same model's 9-class head -- reaches AUC 0.658 over
all 189 speakers, while the 1024-dim speaker-mean embedding from the same model managed
0.671 on the official test and collapsed on the higher-powered instrument. This puts both
on the same instrument (20-repeat stratified CV over 189 speakers) so the comparison is
like for like.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_features import (load_labels, repeated_cv, paired_bootstrap, fast_auc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="/work/eval/turn_features.csv")
    ap.add_argument("--emb", default="/work/emb/emotion2vec_plus_turn.npz")
    ap.add_argument("--data", default="/data/daic")
    ap.add_argument("--out", default="/work/eval/dim_comparison.json")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--boot", type=int, default=40000)
    a = ap.parse_args()

    f = pd.read_csv(a.features)
    df = load_labels(f"{a.data}/meta").merge(f, on="pid")

    Z = np.load(a.emb)
    keep = df.pid.astype(str).isin(Z.files).values
    df = df[keep].reset_index(drop=True)
    E = np.stack([Z[str(p)] for p in df.pid])
    for i in range(E.shape[1]):
        df[f"e{i}"] = E[:, i]
    ecols = [f"e{i}" for i in range(E.shape[1])]

    sets = {"emb1024": ecols, "neg_mean": ["neg_mean"], "lat_median": ["lat_median"]}
    rows, scores = [], {}
    for name, cols in sets.items():
        auc, fold_auc, s = repeated_cv(df, cols, a.repeats)
        scores[name] = s
        rows.append(dict(name=name, n_feat=len(cols), auc_cv189=float(auc),
                         auc_perfold=float(fold_auc)))
        print(f"  {name:12s} n_feat {len(cols):5d}  cv189 {auc:.3f} perfold {fold_auc:.3f}", flush=True)

    y = df.y.values
    b = paired_bootstrap(y, scores["emb1024"], scores["neg_mean"], a.boot)
    v = "better" if b["lo"] > 0 else "worse" if b["hi"] < 0 else "unresolved"
    print(f"\nneg_mean (1 feature) vs emb1024 (1024 features), same instrument:")
    print(f"  dAUC {b['observed']:+.3f} 95% [{b['lo']:+.3f},{b['hi']:+.3f}] -> {v}")

    with open(a.out, "w") as fh:
        json.dump(dict(rows=rows, neg_mean_vs_emb1024=b), fh, indent=2)
    print(f"\nsaved: {a.out}")


if __name__ == "__main__":
    main()
