#!/usr/bin/env python3
"""Per-turn SER extraction, keeping what extract_ser.py threw away.

extract_ser.py averaged the turn vectors into one vector per speaker and saved only
that, so nothing about *when* an emotion appeared survived. This keeps every turn, plus
the 9-class emotion probabilities (interpretable and low-dimensional, which matters with
107 training speakers), and tags each turn with the Ellie question it answers.

Turn boundaries, the 30s chunking and the duration-weighted pooling are unchanged from
extract_ser.py, so a speaker mean computed from this output reproduces the earlier run.
"""
import argparse, json, os, time
import numpy as np, pandas as pd, soundfile as sf

from daic_turns import all_turns

SR = 16000
MAX_CHUNK = 30.0


def load_labels(meta):
    tr = pd.read_csv(f"{meta}/train_split_Depression_AVEC2017.csv")
    dv = pd.read_csv(f"{meta}/dev_split_Depression_AVEC2017.csv")
    te = pd.read_csv(f"{meta}/full_test_split.csv").rename(
        columns={"PHQ_Binary": "PHQ8_Binary", "PHQ_Score": "PHQ8_Score"})
    rows = []
    for name, d in [("train", tr), ("dev", dv), ("test", te)]:
        for _, r in d.iterrows():
            rows.append(dict(pid=int(r.Participant_ID), split=name,
                             y=int(r.PHQ8_Binary), score=int(r.PHQ8_Score)))
    return pd.DataFrame(rows)


def chunks(s, e, maxlen=MAX_CHUNK):
    d = e - s
    if d <= maxlen:
        return [(s, e)]
    n = int(np.ceil(d / maxlen))
    step = d / n
    return [(s + i * step, s + (i + 1) * step) for i in range(n)]


class Emotion2Vec:
    """funasr, the official extraction path. Returns the embedding and the class scores."""
    name = "emotion2vec_plus"

    def __init__(self, device):
        from funasr import AutoModel
        self.m = AutoModel(model="iic/emotion2vec_plus_large", device=device,
                           disable_update=True)
        self.labels = None

    def embed(self, wav):
        r = self.m.generate(wav, granularity="utterance", extract_embedding=True,
                            disable_pbar=True)[0]
        if self.labels is None and "labels" in r:
            self.labels = [str(x) for x in r["labels"]]
        sc = np.asarray(r.get("scores", []), dtype=np.float32)
        return np.asarray(r["feats"], dtype=np.float32), sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/daic")
    ap.add_argument("--out", default="/work/emb")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    meta, audio = f"{a.data}/meta", f"{a.data}/audio"
    os.makedirs(a.out, exist_ok=True)
    lab = load_labels(meta)
    pids = list(lab.pid)[: a.limit] if a.limit else list(lab.pid)

    turns, _ = all_turns(meta, pids)
    print(f"turns: {len(turns)} over {turns.pid.nunique()} speakers "
          f"| median {turns.dur.median():.2f}s", flush=True)

    model = Emotion2Vec(a.device)
    vecs, rows, failed, t0 = {}, [], [], time.time()

    for n, (pid, g) in enumerate(turns.groupby("pid", sort=False), 1):
        wav_path = f"{audio}/{pid}_AUDIO.wav"
        embs, keep = [], []
        for r in g.itertuples():
            ce, cs, cd = [], [], []
            for s, e in chunks(r.start, r.stop):
                st, sp = int(s * SR), int(e * SR)
                if sp <= st:
                    continue
                try:
                    w, _ = sf.read(wav_path, start=st, stop=sp, dtype="float32")
                except Exception as ex:
                    failed.append((pid, r.turn, f"{type(ex).__name__}: {ex}"))
                    continue
                if len(w) < SR * 0.05:
                    continue
                try:
                    v, sc = model.embed(w)
                except Exception as ex:
                    failed.append((pid, r.turn, f"{type(ex).__name__}: {ex}"))
                    continue
                ce.append(v); cs.append(sc); cd.append(e - s)
            if not ce:
                continue
            W = np.asarray(cd, dtype=np.float64)
            embs.append(np.average(np.stack(ce), axis=0, weights=W).astype(np.float32))
            row = dict(pid=pid, turn=r.turn, start=r.start, stop=r.stop, dur=r.dur,
                       qid=r.qid, bucket=r.bucket)
            if cs and cs[0].size:
                p = np.average(np.stack(cs), axis=0, weights=W)
                for i, v in enumerate(p):
                    row[f"p{i}"] = float(v)
            rows.append(row)
            keep.append(r.turn)
        if embs:
            vecs[str(pid)] = np.stack(embs)
        print(f"[{n}/{turns.pid.nunique()}] {pid} turns {len(embs)}/{len(g)} "
              f"| {time.time()-t0:6.0f}s", flush=True)

    np.savez_compressed(f"{a.out}/emotion2vec_turnwise.npz", **vecs)
    df = pd.DataFrame(rows)
    df.to_csv(f"{a.out}/emotion2vec_turnwise_meta.csv", index=False)
    info = dict(model="emotion2vec_plus", unit="turn", max_chunk=MAX_CHUNK,
                labels=model.labels, n_speakers=len(vecs), n_turns=int(len(df)),
                dim=int(next(iter(vecs.values())).shape[1]) if vecs else 0,
                elapsed_sec=round(time.time() - t0, 1), n_failed=len(failed),
                failed=failed[:50])
    with open(f"{a.out}/emotion2vec_turnwise.json", "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"\nsaved {len(vecs)} speakers / {len(df)} turns | labels={model.labels} "
          f"| failures {len(failed)} | {info['elapsed_sec']}s")


if __name__ == "__main__":
    main()
