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


def fast_auc(y, s):
    """Rank-based AUC. Same value as roc_auc_score, fast enough for 40k resamples."""
    order = np.argsort(s, kind="mergesort")
    ys = y[order]
    ss = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(ss):                       # average ranks within ties
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    npos = ys.sum()
    nneg = len(ys) - npos
    if npos == 0 or nneg == 0:
        return np.nan
    return float((ranks[ys == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))

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


def paired_bootstrap(y, sa, sb, n_boot=40000, seed=0, n_compare=1):
    """Paired resampling of the evaluated speakers, models held fixed.

    Reports the plain 95% interval and a Bonferroni-widened one, because several feature
    sets are compared against the same baseline here; with 7 comparisons, one interval
    excluding zero by chance is more likely than not.
    """
    obs = fast_auc(y, sb) - fast_auc(y, sa)
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    k = 0
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        yi = y[i]
        if yi.sum() == 0 or yi.sum() == len(yi):
            continue
        d[k] = fast_auc(yi, sb[i]) - fast_auc(yi, sa[i])
        k += 1
    d = d[:k]
    alpha = 0.05 / max(n_compare, 1)
    return dict(observed=float(obs), n_boot=int(k),
                lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
                lo_adj=float(np.percentile(d, 100 * alpha / 2)),
                hi_adj=float(np.percentile(d, 100 * (1 - alpha / 2))),
                p_worse=float((d < 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="/work/eval/turn_features.csv")
    ap.add_argument("--data", default="/data/daic")
    ap.add_argument("--out", default="/work/eval")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--boot", type=int, default=40000)
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
    ncmp = len(SETS) - 1
    print(f"\n=== gain over lat_median, paired bootstrap "
          f"(95% CI, then Bonferroni for {ncmp} comparisons) ===")
    boots = []
    mt = (df.split == "test").values
    for name in SETS:
        if name == "lat_only":
            continue
        b189 = paired_bootstrap(y, cv_scores["lat_only"], cv_scores[name], a.boot,
                                n_compare=ncmp)
        b47 = paired_bootstrap(yt, off_scores["lat_only"][mt], off_scores[name][mt],
                               a.boot, n_compare=ncmp)
        boots.append(dict(name=name, cv189=b189, test47=b47))
        for tag, b in (("cv189 ", b189), ("test47", b47)):
            v = "better" if b["lo"] > 0 else "worse" if b["hi"] < 0 else "unresolved"
            vadj = ("better" if b["lo_adj"] > 0 else
                    "worse" if b["hi_adj"] < 0 else "unresolved")
            print(f"  {name:14s} {tag} {b['observed']:+.3f} "
                  f"95% [{b['lo']:+.3f},{b['hi']:+.3f}] {v:10s} | "
                  f"adj [{b['lo_adj']:+.3f},{b['hi_adj']:+.3f}] {vadj}")

    print(f"\n=== diagnostic: single features, raw AUC over all 189 ===")
    for c in CONTROLS + NEW:
        v = df[c].values.astype(float)
        ok = np.isfinite(v)
        print(f"  {c:26s} {fast_auc(y[ok], v[ok]):.3f}  "
              f"(n={int(ok.sum())})")

    os.makedirs(a.out, exist_ok=True)
    res.to_csv(f"{a.out}/turn_feature_eval.csv", index=False)
    with open(f"{a.out}/turn_feature_eval.json", "w") as fh:
        json.dump(dict(sets={k: v for k, v in SETS.items()}, rows=rows, bootstrap=boots),
                  fh, indent=2)
    print(f"\nsaved: {a.out}/turn_feature_eval.csv")


if __name__ == "__main__":
    main()
