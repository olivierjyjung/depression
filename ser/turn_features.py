#!/usr/bin/env python3
"""Per-speaker features from turn-level emotion, keeping the count under the 107-speaker budget.

The design (DECISIONS.md 2026-09-14): the speaker mean failed most likely because each
person's habitual voice and affect dominate it. So most features here are *contrasts*
measured inside one speaker -- how their emotion or latency shifts between question
buckets, or between their own typical and extreme turns -- which cancels that baseline.

10 features, including two deliberate controls: lat_median (the only feature that has
worked so far, so gains must be shown on top of it) and neg_mean (the low-dimensional
version of the pooling that already failed).

emotion2vec class order: 0 angry, 1 disgusted, 2 fearful, 3 happy, 4 neutral, 5 other,
6 sad, 7 surprised, 8 unknown.
"""
import argparse, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from daic_turns import all_turns

NEG = ["p0", "p1", "p2", "p6"]      # angry, disgusted, fearful, sad
POS, NEU = "p3", "p4"


def wmean(v, w):
    v, w = np.asarray(v, float), np.asarray(w, float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return float(np.average(v[ok], weights=w[ok])) if ok.sum() else np.nan


def bucket_wmean(g, col, bucket):
    s = g[g.bucket == bucket]
    return wmean(s[col], s.dur) if len(s) else np.nan


def speaker_features(g):
    """g: one speaker's turns, with emotion probabilities, duration, latency, bucket."""
    f = {}
    # controls -- what already works, and the pooling that already failed
    f["lat_median"] = float(np.nanmedian(g.lat)) if g.lat.notna().any() else np.nan
    f["neg_mean"] = wmean(g.neg, g.dur)
    f["neu_mean"] = wmean(g[NEU], g.dur)

    # (1) question-aligned contrasts: how emotion shifts away from neutral questions
    base_neg = bucket_wmean(g, "neg", "neutral")
    base_pos = bucket_wmean(g, POS, "neutral")
    f["neg_symptom_vs_neutral"] = bucket_wmean(g, "neg", "symptom") - base_neg
    f["neg_negative_vs_neutral"] = bucket_wmean(g, "neg", "negative") - base_neg
    f["pos_positive_vs_neutral"] = bucket_wmean(g, POS, "positive") - base_pos

    # (2) emotion x latency -- latency is the one signal that works, so modulate it
    ls = g[g.bucket == "symptom"].lat.median()
    ln = g[g.bucket == "neutral"].lat.median()
    f["lat_symptom_vs_neutral"] = float(ls - ln) if np.isfinite(ls) and np.isfinite(ln) else np.nan
    ok = g.lat.notna() & g.neg.notna()
    f["lat_neg_corr"] = (float(np.corrcoef(g.lat[ok], g.neg[ok])[0, 1])
                         if ok.sum() >= 5 and g.lat[ok].std() > 0 and g.neg[ok].std() > 0
                         else np.nan)

    # (3) tails -- the worst moments rather than the average, each minus the speaker's own median
    f["neg_p90_minus_p50"] = float(np.nanpercentile(g.neg, 90) - np.nanpercentile(g.neg, 50))
    f["lat_p90_minus_p50"] = (float(np.nanpercentile(g.lat.dropna(), 90) -
                                    np.nanpercentile(g.lat.dropna(), 50))
                              if g.lat.notna().sum() >= 5 else np.nan)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-csv", default="/work/emb/emotion2vec_turnwise_meta.csv")
    ap.add_argument("--data", default="/data/daic")
    ap.add_argument("--out", default="/work/eval/turn_features.csv")
    a = ap.parse_args()

    t = pd.read_csv(a.meta_csv)
    meta = f"{a.data}/meta"
    turns, _ = all_turns(meta, sorted(t.pid.unique()))
    t = t.merge(turns[["pid", "turn", "lat"]], on=["pid", "turn"], how="left")

    pcols = [c for c in t.columns if c.startswith("p") and c[1:].isdigit()]
    if not pcols:
        raise SystemExit("no class probabilities in the meta csv")
    t["neg"] = t[NEG].sum(axis=1)

    rows = []
    for pid, g in t.groupby("pid", sort=True):
        r = speaker_features(g)
        r["pid"] = int(pid)
        r["n_turns"] = len(g)
        rows.append(r)
    f = pd.DataFrame(rows).set_index("pid").reset_index()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    f.to_csv(a.out, index=False)

    feat = [c for c in f.columns if c not in ("pid", "n_turns")]
    print(f"speakers {len(f)} | turns {len(t)} | features {len(feat)}")
    print("\nmissing values per feature:")
    for c in feat:
        print(f"  {c:26s} {int(f[c].isna().sum()):3d} missing | "
              f"median {f[c].median():+.4f} | sd {f[c].std():.4f}")
    print(f"\nemotion mass check (duration-weighted over all turns):")
    for c in pcols:
        print(f"  {c} {wmean(t[c], t.dur):.3f}", end="")
    print(f"\nsaved: {a.out}")


if __name__ == "__main__":
    main()
