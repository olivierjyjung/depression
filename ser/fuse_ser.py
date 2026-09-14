#!/usr/bin/env python3
"""Stage 2: does an SER embedding add anything on top of the timing features?

Criteria (DECISIONS.md 2026-09-09): stage 2 asks whether the SER score improves the
current best combination. The 08-21 per-speaker *text* scores no longer exist on disk,
so the reference here is the strongest feature rebuildable from transcripts alone --
response latency (08-21 table: dev 0.678 / test 0.776). Reproducing that pair is the
sanity check that this reimplementation matches what 08-21 actually did; a mismatch is
reported, not silently accepted.

Fusion follows the 08-21 recipe unchanged: per-speaker scalars -> z-scored with train
statistics -> logistic weights fitted on train (107) -> dev/test pass through once,
with the decision threshold fixed at logit 0.

No GPU and no LLM calls: transcripts + the SER score csv written by eval_ser.py.
"""
import argparse, json, os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score

# 08-21 table, single-feature test AUC. Used only to flag a definition mismatch.
REF_LAT = {"dev": 0.678, "test": 0.776}
PAUSE_CAP = 2.0          # 08-21 pause definition: gaps within 0..2s only


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


def timing_features(pid, meta):
    """Per-speaker latency and pause from transcript timestamps only.

    lat   -- Ellie turn ends -> next participant line starts, speaker median.
    pause -- gaps between consecutive participant lines inside one response,
             capped at PAUSE_CAP, speaker mean (the 08-21 definition).
    """
    t = pd.read_csv(f"{meta}/{pid}_TRANSCRIPT.csv", sep="\t").dropna(subset=["speaker"])
    t["speaker"] = t["speaker"].astype(str).str.strip().str.lower()
    t = t.sort_values("start_time").reset_index(drop=True)
    is_p = t["speaker"].str.startswith("participant").values
    start = t["start_time"].astype(float).values
    stop = t["stop_time"].astype(float).values

    lats, pauses, n_neg = [], [], 0
    for i in range(1, len(t)):
        if is_p[i] and not is_p[i - 1]:          # Ellie -> participant: a real turn boundary
            d = start[i] - stop[i - 1]
            if d < 0:
                n_neg += 1
            lats.append(d)
        elif is_p[i] and is_p[i - 1]:            # inside one response
            g = start[i] - stop[i - 1]
            if 0 <= g <= PAUSE_CAP:
                pauses.append(g)
    return dict(pid=pid,
                lat=float(np.median(lats)) if lats else np.nan,
                pause=float(np.mean(pauses)) if pauses else np.nan,
                n_lat=len(lats), n_pause=len(pauses), n_lat_negative=n_neg)


def evaluate(df, cols, name, missing="impute"):
    """Fit logistic on train with z-scores from train statistics; dev/test pass once.

    3 speakers (451, 458, 480) have transcripts with no Ellie lines at all, so latency
    is undefined for them. `missing="impute"` fills the train median -- this keeps
    dev at 35 and test at 47, matching the 08-21 table. `missing="drop"` removes them,
    which is cleaner but evaluates on a different set, so both are reported.
    """
    df = df.copy()
    if missing == "drop":
        df = df.dropna(subset=cols)
    m = {k: (df.split == k).values for k in ["train", "dev", "test"]}
    X = df[cols].values.astype(float)
    if missing == "impute":
        med = np.nanmedian(X[m["train"]], axis=0)
        X = np.where(np.isnan(X), med, X)
    mu, sd = X[m["train"]].mean(0), X[m["train"]].std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    y = df.y.values

    clf = LogisticRegression(max_iter=5000).fit(Z[m["train"]], y[m["train"]])
    s = clf.predict_proba(Z)[:, 1]
    out = {"name": name, "n_feat": len(cols), "missing": missing,
           "n_dev": int(m["dev"].sum()), "n_test": int(m["test"].sum())}
    for k in ["dev", "test"]:
        out[f"auc_{k}"] = float(roc_auc_score(y[m[k]], s[m[k]]))
        out[f"f1_{k}"] = float(f1_score(y[m[k]], (s[m[k]] >= 0.5).astype(int),
                                        zero_division=0))   # threshold = logit 0
    return out, df.assign(score=s)[["pid", "split", "y", "score"]]


def paired_bootstrap(sa, sb, split, n_boot=10000, seed=0):
    """CI on AUC(b) - AUC(a) from resampling the eval speakers, models held fixed.

    Both models are already fitted on train; the question here is only how much the
    difference moves with the eval sample, so the fits are not redone. Resampling is
    paired -- the same resampled speakers go through both models.
    """
    j = sa.merge(sb, on=["pid", "split", "y"], suffixes=("_a", "_b"))
    j = j[j.split == split]
    y, a, b = j.y.values, j.score_a.values, j.score_b.values
    obs = roc_auc_score(y, b) - roc_auc_score(y, a)

    rng = np.random.default_rng(seed)
    d, skipped = [], 0
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:          # AUC undefined on a one-class resample
            skipped += 1
            continue
        d.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    d = np.asarray(d)
    return dict(split=split, n=int(len(y)), n_pos=int(y.sum()), observed=float(obs),
                lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
                p_worse=float((d < 0).mean()), n_boot=int(len(d)), skipped=int(skipped))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ser", nargs="+", required=True, help="eval_ser.py *_scores.csv files")
    ap.add_argument("--data", default="/data/daic")
    ap.add_argument("--out", default="/work/eval")
    ap.add_argument("--boot", type=int, default=10000,
                    help="paired bootstrap resamples; 0 to skip")
    a = ap.parse_args()

    meta = f"{a.data}/meta"
    lab = load_labels(meta)
    tf = pd.DataFrame([timing_features(p, meta) for p in lab.pid])
    df = lab.merge(tf, on="pid")
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(f"{a.out}/timing_features.csv", index=False)

    print(f"timing features: {len(df)} speakers | "
          f"latency values/speaker median {df.n_lat.median():.0f} | "
          f"negative latencies {int(df.n_lat_negative.sum())} total")
    print(f"missing: lat {int(df.lat.isna().sum())} | pause {int(df.pause.isna().sum())}")

    sers = []
    for p in a.ser:
        tag = os.path.basename(p).replace("_scores.csv", "")
        s = pd.read_csv(p)[["pid", "ser"]].rename(columns={"ser": tag})
        if len(df.merge(s, on="pid")) != len(df):
            print(f"  warning: {tag} covers {len(df.merge(s, on='pid'))}/{len(df)} speakers")
        sers.append((tag, s))

    all_rows, checks, scores = [], {}, {}
    for mode in ["impute", "drop"]:
        rows = []

        def add(d, cols, name):
            r, sc = evaluate(d, cols, name, mode)
            rows.append(r)
            scores[(mode, name)] = sc

        add(df, ["lat"], "lat")
        add(df, ["pause"], "pause")
        add(df, ["lat", "pause"], "lat+pause")
        for tag, s in sers:
            d2 = df.merge(s, on="pid")
            add(d2, [tag], tag)
            add(d2, ["lat", tag], f"lat+{tag}")
            add(d2, ["lat", "pause", tag], f"lat+pause+{tag}")

        lat = rows[0]
        delta = {k: lat[f"auc_{k}"] - REF_LAT[k] for k in REF_LAT}
        ok = all(abs(v) <= 0.03 for v in delta.values())
        checks[mode] = dict(lat_auc={k: lat[f"auc_{k}"] for k in REF_LAT},
                            delta=delta, reproduced=bool(ok))

        res = pd.DataFrame(rows)[["name", "n_feat", "n_dev", "n_test",
                                  "auc_dev", "auc_test", "f1_dev", "f1_test"]]
        print(f"\n=== missing latency handling: {mode} "
              f"(threshold = logit 0) ===")
        print(f"sanity check vs 08-21 lat ({REF_LAT['dev']:.3f}/{REF_LAT['test']:.3f}): "
              f"got {lat['auc_dev']:.3f}/{lat['auc_test']:.3f} "
              f"(diff {delta['dev']:+.3f}/{delta['test']:+.3f}) -> "
              f"{'reproduced' if ok else 'MISMATCH - definition differs from 08-21'}")
        print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print("  -- gain over lat alone --")
        for r in rows:
            if r["name"].startswith("lat+") and r["name"] != "lat+pause":
                print(f"  {r['name']:32s} dAUC dev {r['auc_dev']-lat['auc_dev']:+.3f} "
                      f"test {r['auc_test']-lat['auc_test']:+.3f} | "
                      f"dF1 dev {r['f1_dev']-lat['f1_dev']:+.3f} "
                      f"test {r['f1_test']-lat['f1_test']:+.3f}")
        all_rows += rows

    # Is the drop in test AUC bigger than eval-sample noise? Paired bootstrap on the
    # impute tables (the primary ones, dev 35 / test 47 as in the 08-21 table).
    boots = []
    if a.boot:
        print(f"\n=== paired bootstrap, {a.boot} resamples "
              f"(eval speakers resampled; models held fixed) ===")
        pairs = [(base, f"{base}+{tag}") for tag, _ in sers for base in ["lat", "lat+pause"]]
        for base, fused in pairs:
            for split in ["dev", "test"]:
                r = paired_bootstrap(scores[("impute", base)], scores[("impute", fused)],
                                     split, n_boot=a.boot)
                r["base"], r["fused"] = base, fused
                boots.append(r)
                verdict = ("worse" if r["hi"] < 0 else
                           "better" if r["lo"] > 0 else "indistinguishable")
                print(f"  {split:4s} n={r['n']:2d} pos={r['n_pos']:2d} | {fused:32s} "
                      f"dAUC {r['observed']:+.3f} [95% CI {r['lo']:+.3f}, {r['hi']:+.3f}] "
                      f"P(worse)={r['p_worse']:.2f} -> {verdict}")

    res = pd.DataFrame(all_rows)[["missing", "name", "n_feat", "n_dev", "n_test",
                                  "auc_dev", "auc_test", "f1_dev", "f1_test"]]
    res.to_csv(f"{a.out}/stage2_fusion.csv", index=False)
    with open(f"{a.out}/stage2_fusion.json", "w") as f:
        json.dump(dict(reference=REF_LAT, sanity=checks, pause_cap=PAUSE_CAP,
                       no_ellie_pids=[int(p) for p in df[df.lat.isna()].pid],
                       rows=all_rows, bootstrap=boots), f, indent=2)
    print(f"\nsaved: {a.out}/stage2_fusion.csv, {a.out}/timing_features.csv")


if __name__ == "__main__":
    main()
