#!/usr/bin/env python3
"""Which SER model is the better feature extractor for this task?

This is the question the supervisor actually asked on 09-02 ("find out which speech
emotion recognition model is suitable and check its performance"), so it is reported as a
number with an interval rather than judged against a threshold.

Both models are compared on identical inputs -- the same turn boundaries, the same 30s
chunking, the same duration-weighted pooling -- and on the same instrument, the
189-speaker repeated CV with the regularization and PCA chosen inside training folds.
emotion2vec+ gives 1024 dimensions through funasr's official path; Emotion2Vec-S gives
768 through C2SER's fairseq path, averaged over its 10 utterance tokens.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emb_contrast import build, matrix, run_cv, run_official
from eval_features import load_labels, fast_auc, paired_bootstrap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/daic")
    ap.add_argument("--out", default="/work/eval/model_comparison.json")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--boot", type=int, default=40000)
    ap.add_argument("--models", nargs="+", default=[
        "emotion2vec_plus:/work/emb/emotion2vec_turnwise.npz:/work/emb/emotion2vec_turnwise_meta.csv",
        "emotion2vec_s:/work/emb/emotion2vec_s_turnwise.npz:/work/emb/emotion2vec_s_turnwise_meta.csv"])
    a = ap.parse_args()

    built = {}
    for spec in a.models:
        tag, npz, meta = spec.split(":")
        rows, D = build(npz, meta, a.data)
        built[tag] = (rows, D)
        print(f"{tag}: {len(rows)} speakers, dim {D}", flush=True)

    # evaluate every model on the speakers all of them cover, so the comparison is paired
    common = set.intersection(*[set(r) for r, _ in built.values()])
    lab = load_labels(f"{a.data}/meta")
    lab = lab[lab.pid.isin(common)].reset_index(drop=True)
    y, split, pids = lab.y.values, lab.split, list(lab.pid)
    print(f"common speakers: {len(pids)} | positives {int(y.sum())}")

    res, scores = [], {}
    for tag, (rows, D) in built.items():
        for block, n_lat, n_emb in [("mean", 0, D), ("lat+mean", 1, D)]:
            blocks = ["mean"] if block == "mean" else ["lat", "mean"]
            X = matrix(rows, pids, blocks, D)
            cv_auc, fold_auc, s = run_cv(X, y, n_lat, n_emb, a.repeats)
            off, _, soff = run_official(X, y, split, n_lat, n_emb)
            scores[f"{tag}:{block}"] = s
            res.append(dict(model=tag, block=block, dim=D, **off,
                            auc_cv189=cv_auc, auc_perfold=fold_auc))
            print(f"  {tag:16s} {block:9s} dev {off['auc_dev']:.3f} "
                  f"test {off['auc_test']:.3f} | cv189 {cv_auc:.3f} "
                  f"perfold {fold_auc:.3f}", flush=True)

    # latency alone, once, as the shared reference point
    rows0, D0 = next(iter(built.values()))
    Xl = matrix(rows0, pids, ["lat"], D0)
    cv_lat, fold_lat, s_lat = run_cv(Xl, y, 1, 0, a.repeats)
    scores["lat_only"] = s_lat
    print(f"  {'lat_only':16s} {'-':9s} cv189 {cv_lat:.3f} perfold {fold_lat:.3f}")

    df = pd.DataFrame(res)[["model", "block", "dim", "auc_dev", "auc_test",
                            "auc_cv189", "auc_perfold"]]
    print("\n=== SER model comparison, identical inputs and instrument ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    tags = list(built)
    out = {"rows": res, "lat_only_cv189": cv_lat, "pairs": []}
    if len(tags) == 2:
        print("\n=== paired difference, cv189 (second minus first) ===")
        for block in ["mean", "lat+mean"]:
            b = paired_bootstrap(y, scores[f"{tags[0]}:{block}"],
                                 scores[f"{tags[1]}:{block}"], a.boot)
            v = ("second better" if b["lo"] > 0 else
                 "first better" if b["hi"] < 0 else "indistinguishable")
            out["pairs"].append(dict(block=block, a=tags[0], b=tags[1], **b))
            print(f"  {block:9s} {tags[1]} − {tags[0]}: {b['observed']:+.3f} "
                  f"95% [{b['lo']:+.3f},{b['hi']:+.3f}] -> {v}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved: {a.out}")


if __name__ == "__main__":
    main()
