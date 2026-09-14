#!/usr/bin/env python3
"""Evaluate the turn-level features two ways, because one way cannot answer the question.

Official split (train 107 / dev 35 / test 47) keeps the numbers comparable to the 08-21
table and to published work. But the 09-14 bootstrap showed a 47-speaker test set cannot
resolve an AUC difference below roughly 0.15, so a second, higher-powered estimate runs
alongside it: repeated stratified cross-validation over all 189 speakers, with the
regularization strength chosen inside each training fold only, then a paired bootstrap
over the 189 out-of-fold scores.

The cross-validated number is not comparable to the official-split numbers -- it uses
every speaker and a different protocol -- so it is reported as a secondary instrument for
comparing feature sets against each other, not as a headline result.
"""
import argparse, json, os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

CS = np.logspace(-4, 2, 13)

CONTROLS = ["lat_median", "neg_mean", "neu_mean"]
CONTRAST = ["neg_symptom_vs_neutral", "neg_negative_vs_neutral", "pos_positive_vs_neutral"]
LATEMO = ["lat_symptom_vs_neutral", "lat_neg_corr"]
TAILS = ["neg_p90_minus_p50", "lat_p90_minus_p50"]
NEW = CONTRAST + LATEMO + TAILS

SETS = {
    "lat_only": ["lat_median"],
    "controls": CONTROLS,
    "new_only": NEW,
    "lat+contrast": ["lat_median"] + CONTRAST,
    "lat+latemo": ["lat_median"] + LATEMO,
    "lat+tails": ["lat_median"] + TAILS,
    "lat+new": ["lat_median"] + NEW,
    "all10": CONTROLS + NEW,
}


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


def prep(df, cols, fit_mask):
    """Impute with the fitting subset's median and z-score with its statistics."""
    X = df[cols].values.astype(float)
    med = np.nanmedian(X[fit_mask], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    X = np.where(np.isnan(X), med, X)
    mu, sd = X[fit_mask].mean(0), X[fit_mask].std(0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def official(df, cols, seed=0):
    m = {k: (df.split == k).values for k in ["train", "dev", "test"]}
    Z, y = prep(df, cols, m["train"]), df.y.values
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    gs = GridSearchCV(make_pipeline(LogisticRegression(max_iter=5000)),
                      {"logisticregression__C": CS}, scoring="roc_auc", cv=cv, n_jobs=-1)
    gs.fit(Z[m["train"]], y[m["train"]])
    clf = LogisticRegression(C=gs.best_params_["logisticregression__C"],
                             max_iter=5000).fit(Z[m["train"]], y[m["train"]])
    s = clf.predict_proba(Z)[:, 1]
    return ({f"auc_{k}": float(roc_auc_score(y[m[k]], s[m[k]])) for k in ["dev", "test"]}
            | {"C": float(gs.best_params_["logisticregression__C"])}), s


def repeated_cv(df, cols, n_repeats=20, seed=0):
    """Out-of-fold scores over all 189 speakers, averaged across repeats."""
    y = df.y.values
    acc, cnt = np.zeros(len(df)), np.zeros(len(df))
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=seed)
    for tr, te in rskf.split(np.zeros(len(df)), y):
        fit = np.zeros(len(df), bool); fit[tr] = True
        Z = prep(df, cols, fit)                      # imputation/scaling from train fold only
        inner = StratifiedKFold(5, shuffle=True, random_state=seed)
        gs = GridSearchCV(make_pipeline(LogisticRegression(max_iter=5000)),
                          {"logisticregression__C": CS}, scoring="roc_auc",
                          cv=inner, n_jobs=-1)
        gs.fit(Z[tr], y[tr])
        clf = LogisticRegression(C=gs.best_params_["logisticregression__C"],
                                 max_iter=5000).fit(Z[tr], y[tr])
        acc[te] += clf.predict_proba(Z[te])[:, 1]
        cnt[te] += 1
    s = acc / np.maximum(cnt, 1)
    return float(roc_auc_score(y, s)), s


def paired_bootstrap(y, sa, sb, n_boot=10000, seed=0):
    obs = roc_auc_score(y, sb) - roc_auc_score(y, sa)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], sb[i]) - roc_auc_score(y[i], sa[i]))
    d = np.asarray(d)
    return dict(observed=float(obs), lo=float(np.percentile(d, 2.5)),
                hi=float(np.percentile(d, 97.5)), p_worse=float((d < 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="/work/eval/turn_features.csv")
    ap.add_argument("--data", default="/data/daic")
    ap.add_argument("--out", default="/work/eval")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()

    f = pd.read_csv(a.features)
    df = load_labels(f"{a.data}/meta").merge(f, on="pid")
    print(f"speakers {len(df)} | positives {int(df.y.sum())} "
          f"(train {int(df[df.split=='train'].y.sum())}/{(df.split=='train').sum()}, "
          f"dev {int(df[df.split=='dev'].y.sum())}/{(df.split=='dev').sum()}, "
          f"test {int(df[df.split=='test'].y.sum())}/{(df.split=='test').sum()})")

    rows, cv_scores, off_scores = [], {}, {}
    for name, cols in SETS.items():
        o, so = official(df, cols)
        cv_auc, scv = repeated_cv(df, cols, a.repeats)
        cv_scores[name], off_scores[name] = scv, so
        rows.append(dict(name=name, n_feat=len(cols), **o, auc_cv189=cv_auc))
        print(f"  {name:14s} n={len(cols):2d} dev {o['auc_dev']:.3f} "
              f"test {o['auc_test']:.3f} | cv189 {cv_auc:.3f}", flush=True)

    res = pd.DataFrame(rows)[["name", "n_feat", "C", "auc_dev", "auc_test", "auc_cv189"]]
    print("\n=== feature sets ===")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    y = df.y.values
    yt = y[(df.split == "test").values]
    print(f"\n=== gain over lat_median, paired bootstrap ===")
    boots = []
    for name in SETS:
        if name == "lat_only":
            continue
        b189 = paired_bootstrap(y, cv_scores["lat_only"], cv_scores[name], a.boot)
        mt = (df.split == "test").values
        b47 = paired_bootstrap(yt, off_scores["lat_only"][mt], off_scores[name][mt], a.boot)
        boots.append(dict(name=name, cv189=b189, test47=b47))
        v189 = "worse" if b189["hi"] < 0 else "better" if b189["lo"] > 0 else "unresolved"
        print(f"  {name:14s} cv189 {b189['observed']:+.3f} "
              f"[{b189['lo']:+.3f},{b189['hi']:+.3f}] {v189:10s} | "
              f"test47 {b47['observed']:+.3f} [{b47['lo']:+.3f},{b47['hi']:+.3f}]")

    os.makedirs(a.out, exist_ok=True)
    res.to_csv(f"{a.out}/turn_feature_eval.csv", index=False)
    with open(f"{a.out}/turn_feature_eval.json", "w") as fh:
        json.dump(dict(sets={k: v for k, v in SETS.items()}, rows=rows, bootstrap=boots),
                  fh, indent=2)
    print(f"\nsaved: {a.out}/turn_feature_eval.csv")


if __name__ == "__main__":
    main()
