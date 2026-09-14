#!/usr/bin/env python3
"""Why does a single feature lose AUC when it goes through the CV pipeline?

neg_mean scores 0.658 read straight off the feature values but 0.537 as out-of-fold
predictions; lat_median goes 0.632 -> 0.557. A one-feature logistic is a monotone
transform of that feature, so within a fold the ranking cannot change -- unless the
fitted coefficient changes sign between folds, in which case averaging predictions
across repeats destroys the ordering. If that is what happens, the lat_only baseline in
the 09-14 comparison is understated and the reported gain over it is inflated.

Checks, for each single feature: raw AUC, out-of-fold AUC, the rank correlation between
the two scores, the sign of the fitted coefficients, and the AUC per repeat before any
averaging.
"""
import argparse, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_features import load_labels, prep, fast_auc, CS


def diagnose(df, col, n_repeats=20, seed=0):
    y = df.y.values
    acc, cnt = np.zeros(len(df)), np.zeros(len(df))
    coefs, per_repeat = [], []
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=seed)
    fold_scores = np.zeros(len(df))
    for k, (tr, te) in enumerate(rskf.split(np.zeros(len(df)), y)):
        fit = np.zeros(len(df), bool); fit[tr] = True
        Z = prep(df, [col], fit)
        inner = StratifiedKFold(5, shuffle=True, random_state=seed)
        gs = GridSearchCV(make_pipeline(LogisticRegression(max_iter=5000)),
                          {"logisticregression__C": CS}, scoring="roc_auc",
                          cv=inner, n_jobs=-1)
        gs.fit(Z[tr], y[tr])
        clf = LogisticRegression(C=gs.best_params_["logisticregression__C"],
                                 max_iter=5000).fit(Z[tr], y[tr])
        coefs.append(float(clf.coef_[0][0]))
        p = clf.predict_proba(Z[te])[:, 1]
        acc[te] += p; cnt[te] += 1
        fold_scores[te] = p
        if (k + 1) % 5 == 0:                       # one full pass over the data
            per_repeat.append(fast_auc(y, fold_scores))
            fold_scores = np.zeros(len(df))
    s = acc / np.maximum(cnt, 1)

    v = df[col].values.astype(float)
    ok = np.isfinite(v)
    raw = fast_auc(y[ok], v[ok])
    coefs = np.asarray(coefs)
    return dict(col=col, raw_auc=raw, oof_auc=fast_auc(y, s),
                per_repeat_mean=float(np.mean(per_repeat)),
                per_repeat_min=float(np.min(per_repeat)),
                per_repeat_max=float(np.max(per_repeat)),
                spearman_score_vs_feature=float(spearmanr(s[ok], v[ok]).statistic),
                coef_positive=int((coefs > 0).sum()), coef_negative=int((coefs < 0).sum()),
                coef_zero=int((np.abs(coefs) < 1e-8).sum()),
                coef_median=float(np.median(coefs)),
                coef_absmedian=float(np.median(np.abs(coefs))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="/work/eval/turn_features.csv")
    ap.add_argument("--data", default="/data/daic")
    ap.add_argument("--cols", nargs="+",
                    default=["lat_median", "neg_mean", "lat_symptom_vs_neutral"])
    a = ap.parse_args()

    f = pd.read_csv(a.features)
    df = load_labels(f"{a.data}/meta").merge(f, on="pid")
    for c in a.cols:
        d = diagnose(df, c)
        print(f"\n--- {c} ---")
        print(f"  raw AUC (feature values)      {d['raw_auc']:.3f}")
        print(f"  out-of-fold AUC (averaged)    {d['oof_auc']:.3f}")
        print(f"  AUC per repeat, before averaging: mean {d['per_repeat_mean']:.3f} "
              f"(min {d['per_repeat_min']:.3f}, max {d['per_repeat_max']:.3f})")
        print(f"  spearman(oof score, feature)  {d['spearman_score_vs_feature']:+.3f}")
        print(f"  fitted coefficient sign: +{d['coef_positive']} / -{d['coef_negative']} "
              f"/ ~0 {d['coef_zero']}  | median {d['coef_median']:+.4f} "
              f"(|median| {d['coef_absmedian']:.4f})")


if __name__ == "__main__":
    main()
