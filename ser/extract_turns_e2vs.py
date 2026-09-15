#!/usr/bin/env python3
"""Per-turn Emotion2Vec-S embeddings, matching the emotion2vec+ extraction exactly.

Same turn boundaries, same 30s chunking, same duration-weighted pooling as
extract_turns_ser.py, so the two models can be compared on identical inputs. The only
difference is the loader: Emotion2Vec-S ships as a fairseq checkpoint from the C2SER
project, and has no classifier head, so there are no class probabilities to save -- only
the embedding. That is why the feature recipe for this model has to be the bucket
contrasts in emb_contrast.py rather than the 9-class features from 09-14.

Runs in the py310 conda env: fairseq 0.12.2 cannot be imported on Python 3.11
(see DECISIONS.md 2026-09-15).
"""
import argparse, json, os, sys, time
from dataclasses import dataclass
import numpy as np, pandas as pd, soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from daic_turns import all_turns

SR = 16000
MAX_CHUNK = 30.0


@dataclass
class UserDirModule:
    user_dir: str


def load_labels(meta):
    tr = pd.read_csv(f"{meta}/train_split_Depression_AVEC2017.csv")
    dv = pd.read_csv(f"{meta}/dev_split_Depression_AVEC2017.csv")
    te = pd.read_csv(f"{meta}/full_test_split.csv").rename(
        columns={"PHQ_Binary": "PHQ8_Binary", "PHQ_Score": "PHQ8_Score"})
    rows = []
    for name, d in [("train", tr), ("dev", dv), ("test", te)]:
        for _, r in d.iterrows():
            rows.append(dict(pid=int(r.Participant_ID), split=name,
                             y=int(r.PHQ8_Binary)))
    return pd.DataFrame(rows)


def chunks(s, e, maxlen=MAX_CHUNK):
    d = e - s
    if d <= maxlen:
        return [(s, e)]
    n = int(np.ceil(d / maxlen))
    step = d / n
    return [(s + i * step, s + (i + 1) * step) for i in range(n)]


class Emotion2VecS:
    name = "emotion2vec_s"

    def __init__(self, ckpt, model_dir, device):
        import torch, fairseq
        self.torch = torch
        self.device = device
        fairseq.utils.import_user_module(UserDirModule(model_dir))
        models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt])
        self.m = models[0].to(device).eval()

    def embed(self, wav):
        """One 768-dim vector per segment.

        extract_features returns x (1, frames, 768) and utt_x (1, 10, 768) -- the model
        carries 10 extra tokens, and C2SER's own script treats utt_x as the utterance
        representation. Those 10 tokens are averaged here to get a single vector, chosen
        before looking at any result, the same way funasr's official path was taken for
        emotion2vec+. The frame mean is a plausible alternative and is deliberately not
        also tried, to keep this a single pre-specified comparison.
        """
        x = self.torch.from_numpy(np.ascontiguousarray(wav)).float().view(1, -1).to(self.device)
        with self.torch.no_grad():
            out = self.m.extract_features(x)
        v = out["utt_x"].mean(dim=1) if "utt_x" in out else out["x"].mean(dim=1)
        return v.squeeze(0).float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/work/C2SER/Emotion2Vec-S/ckpt/checkpoint.pt")
    ap.add_argument("--model-dir", default="/work/C2SER/Emotion2Vec-S/examples/data2vec/")
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
    print(f"turns: {len(turns)} over {turns.pid.nunique()} speakers", flush=True)

    model = Emotion2VecS(a.ckpt, a.model_dir, a.device)
    vecs, rows, failed, t0 = {}, [], [], time.time()

    for n, (pid, g) in enumerate(turns.groupby("pid", sort=False), 1):
        wav_path = f"{audio}/{pid}_AUDIO.wav"
        embs = []
        for r in g.itertuples():
            ce, cd = [], []
            for s, e in chunks(r.start, r.stop):
                st, sp = int(s * SR), int(e * SR)
                if sp <= st:
                    continue
                try:
                    w, _ = sf.read(wav_path, start=st, stop=sp, dtype="float32")
                    if len(w) < SR * 0.05:
                        continue
                    ce.append(model.embed(w)); cd.append(e - s)
                except Exception as ex:
                    failed.append((pid, int(r.turn), f"{type(ex).__name__}: {ex}"))
            if not ce:
                continue
            W = np.asarray(cd, dtype=np.float64)
            embs.append(np.average(np.stack(ce), axis=0, weights=W).astype(np.float32))
            rows.append(dict(pid=pid, turn=int(r.turn), start=r.start, stop=r.stop,
                             dur=r.dur, qid=r.qid, bucket=r.bucket))
        if embs:
            vecs[str(pid)] = np.stack(embs)
        print(f"[{n}/{turns.pid.nunique()}] {pid} turns {len(embs)}/{len(g)} "
              f"| {time.time()-t0:6.0f}s", flush=True)

    np.savez_compressed(f"{a.out}/emotion2vec_s_turnwise.npz", **vecs)
    df = pd.DataFrame(rows)
    df.to_csv(f"{a.out}/emotion2vec_s_turnwise_meta.csv", index=False)
    info = dict(model="emotion2vec_s", unit="turn", max_chunk=MAX_CHUNK,
                n_speakers=len(vecs), n_turns=int(len(df)),
                dim=int(next(iter(vecs.values())).shape[1]) if vecs else 0,
                elapsed_sec=round(time.time() - t0, 1), n_failed=len(failed),
                failed=failed[:50])
    with open(f"{a.out}/emotion2vec_s_turnwise.json", "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"\nsaved {len(vecs)} speakers / {len(df)} turns | dim {info['dim']} "
          f"| failures {len(failed)} | {info['elapsed_sec']}s")


if __name__ == "__main__":
    main()
