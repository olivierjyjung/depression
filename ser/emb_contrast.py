#!/usr/bin/env python3
"""Question-bucket contrasts on the turn-level embeddings, not on the 9-class head.

The 09-14 features were built from emotion2vec+'s 9 class probabilities, which turned out
to be almost entirely the "sad" probability. The 1024-dim per-turn vectors were saved and
never used -- even though the speaker-mean embedding was the strongest single thing
measured (cv189 0.671). This applies the same baseline-cancelling idea to the embedding:
the symptom-question mean vector minus the neutral-question mean vector, and so on.

Emotion2Vec-S has no classifier head at all, so for that model this is the only route.
Pass a different --npz/--meta to run the same analysis on it.

1024-dim contrasts against 107 training speakers need reduction, so PCA sits inside the
pipeline and its component count is chosen within the training folds only, never on the
data it is scored against. Response latency is passed through unreduced.

Pre-registered before looking at any number (DECISIONS.md 2026-09-15): the single judged
comparison is `lat+contrast` against `lat_only` on the 189-speaker instrument. Everything
else in the table is descriptive.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_features import load_labels, fast_auc, paired_bootstrap
from daic_turns import all_turns

CS = np.logspace(-3, 2, 11)
NCOMP = [2, 4, 8, 16]


def wmean(V, w):
    w = np.asarray(w, float)
    return np.average(V, axis=0, weights=w) if len(V) and w.sum() > 0 else None


def build(npz_path, meta_csv, data_dir):
    """Per-speaker blocks: the overall mean vector and three bucket contrasts."""
    Z = np.load(npz_path)
    m = pd.read_csv(meta_csv)
    turns, _ = all_turns(f"{data_dir}/meta", sorted(m.pid.unique()))
    m = m.merge(turns[["pid", "turn", "lat"]], on=["pid", "turn"], how="left")

    rows = {}
    D = None
    for pid, g in m.groupby("pid", sort=True):
        k = str(pid)
        if k not in Z.files:
            continue
        V = Z[k]
        if len(V) != len(g):
            print(f"  skip {pid}: {len(V)} vectors vs {len(g)} meta rows")
            continue
        D = V.shape[1]
        g = g.reset_index(drop=True)
        base = wmean(V[(g.bucket == "neutral").values], g.dur[g.bucket == "neutral"])
        rec = {"pid": int(pid),
               "lat_median": float(np.nanmedian(g.lat)) if g.lat.notna().any() else np.nan,
               "mean": wmean(V, g.dur)}
        for b in ["symptom", "negative", "positive"]:
            mb = wmean(V[(g.bucket == b).values], g.dur[g.bucket == b])
            rec[f"c_{b}"] = (mb - base) if (mb is not None and base is not None) else None
        rows[int(pid)] = rec
    return rows, D


def matrix(rows, pids, blocks, D):
    """Stack the requested blocks into one matrix; missing blocks become NaN."""
    out = []
    for p in pids:
        r, v = rows[p], []
        for b in blocks:
            if b == "lat":
                v.append([r["lat_median"]])
            else:
                x = r.get(b)
                v.append(np.full(D, np.nan) if x is None else x)
        out.append(np.concatenate(v))
    return np.asarray(out, float)


def make_pipe(n_lat, n_emb):
    """PCA on the embedding columns, latency passed through; both fitted per fold."""
    emb = list(range(n_lat, n_lat + n_emb))
    lat = list(range(n_lat))
    steps = []
    if n_emb:
        steps.append(("ct", ColumnTransformer(
            [("emb", Pipeline([("sc", StandardScaler()), ("pca", PCA(random_state=0))]), emb)]
            + ([("lat", StandardScaler(), lat)] if lat else []))))
    else:
        steps.append(("sc", StandardScaler()))
    steps.append(("lr", LogisticRegression(max_iter=5000)))
    return Pipeline(steps)


def grid(n_emb):
    g = {"lr__C": CS}
    if n_emb:
        g["ct__emb__pca__n_components"] = [c for c in NCOMP if c < n_emb]
    return g


def impute(X, fit):
    med = np.nanmedian(X[fit], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return np.where(np.isnan(X), med, X)


def run_cv(X, y, n_lat, n_emb, n_repeats=10, seed=0):
    acc, cnt, fold_aucs = np.zeros(len(y)), np.zeros(len(y)), []
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=seed)
    for tr, te in rskf.split(X, y):
        fit = np.zeros(len(y), bool); fit[tr] = True
        Xi = impute(X, fit)
        gs = GridSearchCV(make_pipe(n_lat, n_emb), grid(n_emb), scoring="roc_auc",
                          cv=StratifiedKFold(5, shuffle=True, random_state=seed), n_jobs=-1)
        gs.fit(Xi[tr], y[tr])
        p = gs.best_estimator_.predict_proba(Xi[te])[:, 1]
        fold_aucs.append(fast_auc(y[te], p))
        acc[te] += rankdata(p) / (len(p) + 1.0)     # rank within fold before pooling
        cnt[te] += 1
    s = acc / np.maximum(cnt, 1)
    return float(fast_auc(y, s)), float(np.mean(fold_aucs)), s


def run_official(X, y, split, n_lat, n_emb, seed=0):
    m = {k: (split == k).values for k in ["train", "dev", "test"]}
    Xi = impute(X, m["train"])
    gs = GridSearchCV(make_pipe(n_lat, n_emb), grid(n_emb), scoring="roc_auc",
                      cv=StratifiedKFold(5, shuffle=True, random_state=seed), n_jobs=-1)
    gs.fit(Xi[m["train"]], y[m["train"]])
    s = gs.best_estimator_.predict_proba(Xi)[:, 1]
    return ({f"auc_{k}": float(fast_auc(y[m[k]], s[m[k]])) for k in ["dev", "test"]},
            gs.best_params_, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/work/emb/emotion2vec_turnwise.npz")
    ap.add_argument("--meta", default="/work/emb/emotion2vec_turnwise_meta.csv")
    ap.add_argument("--data", default="/data/daic")
    ap.add_argument("--tag", default="emotion2vec_plus")
    ap.add_argument("--out", default="/work/eval")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--boot", type=int, default=40000)
    ap.add_argument("--prereg", default="lat+contrast",
                    help="the one judged feature set; everything else is descriptive")
    a = ap.parse_args()

    rows, D = build(a.npz, a.meta, a.data)
    lab = load_labels(f"{a.data}/meta")
    lab = lab[lab.pid.isin(rows)].reset_index(drop=True)
    y, split, pids = lab.y.values, lab.split, list(lab.pid)
    print(f"{a.tag}: {len(pids)} speakers | embedding dim {D} | positives {int(y.sum())}")

    SETS = {
        "lat_only":     (["lat"], 1, 0),
        "mean":         (["mean"], 0, D),
        "contrast":     (["c_symptom", "c_negative", "c_positive"], 0, 3 * D),
        "lat+contrast": (["lat", "c_symptom", "c_negative", "c_positive"], 1, 3 * D),
        "lat+mean":     (["lat", "mean"], 1, D),
    }

    res, cvs, offs = [], {}, {}
    for name, (blocks, n_lat, n_emb) in SETS.items():
        X = matrix(rows, pids, blocks, D)
        cv_auc, fold_auc, scv = run_cv(X, y, n_lat, n_emb, a.repeats)
        off, best, soff = run_official(X, y, split, n_lat, n_emb)
        cvs[name], offs[name] = scv, soff
        res.append(dict(name=name, n_cols=X.shape[1], **off,
                        auc_cv189=cv_auc, auc_perfold=fold_auc,
                        best=str(best)))
        print(f"  {name:13s} cols {X.shape[1]:5d} dev {off['auc_dev']:.3f} "
              f"test {off['auc_test']:.3f} | cv189 {cv_auc:.3f} perfold {fold_auc:.3f}",
              flush=True)

    df = pd.DataFrame(res)[["name", "n_cols", "auc_dev", "auc_test", "auc_cv189",
                            "auc_perfold", "best"]]
    print(f"\n=== {a.tag}: bucket contrasts on turn embeddings ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print(f"\n=== pre-registered comparison: {a.prereg} vs lat_only ===")
    b = paired_bootstrap(y, cvs["lat_only"], cvs[a.prereg], a.boot)
    v = "better" if b["lo"] > 0 else "worse" if b["hi"] < 0 else "unresolved"
    print(f"  cv189  dAUC {b['observed']:+.3f} 95% [{b['lo']:+.3f},{b['hi']:+.3f}] -> {v}")
    mt = (split == "test").values
    b47 = paired_bootstrap(y[mt], offs["lat_only"][mt], offs[a.prereg][mt], a.boot)
    v47 = "better" if b47["lo"] > 0 else "worse" if b47["hi"] < 0 else "unresolved"
    print(f"  test47 dAUC {b47['observed']:+.3f} 95% [{b47['lo']:+.3f},{b47['hi']:+.3f}] "
          f"-> {v47}")
    print("  (single pre-registered comparison, so no multiplicity correction applies)")

    print("\n=== descriptive: other sets vs lat_only on cv189 ===")
    for name in SETS:
        if name in ("lat_only", a.prereg):
            continue
        bb = paired_bootstrap(y, cvs["lat_only"], cvs[name], a.boot)
        print(f"  {name:13s} {bb['observed']:+.3f} [{bb['lo']:+.3f},{bb['hi']:+.3f}]")

    os.makedirs(a.out, exist_ok=True)
    df.to_csv(f"{a.out}/{a.tag}_contrast.csv", index=False)
    with open(f"{a.out}/{a.tag}_contrast.json", "w") as fh:
        json.dump(dict(dim=D, prereg=a.prereg, rows=res,
                       prereg_cv189=b, prereg_test47=b47), fh, indent=2)
    print(f"\nsaved: {a.out}/{a.tag}_contrast.csv")


if __name__ == "__main__":
    main()
