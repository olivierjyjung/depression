#!/usr/bin/env python3
"""Extract SER embeddings from DAIC-WoZ participant speech, one vector per speaker.

Design decisions (see DECISIONS.md, entries 2026-09-09 / 2026-09-11):
  - Unit is the turn (consecutive Participant lines merged into one)
  - Short utterances are kept, not dropped; pooled by duration-weighted mean
  - Turns longer than 30s are split into 30s chunks and pooled the same way
"""
import argparse, glob, json, os, time
import numpy as np, pandas as pd, soundfile as sf

SR = 16000
MAX_CHUNK = 30.0


def load_labels(meta):
    tr = pd.read_csv(f"{meta}/train_split_Depression_AVEC2017.csv")
    dv = pd.read_csv(f"{meta}/dev_split_Depression_AVEC2017.csv")
    # test labels live only in full_test_split.csv and use different column names
    te = pd.read_csv(f"{meta}/full_test_split.csv").rename(
        columns={"PHQ_Binary": "PHQ8_Binary", "PHQ_Score": "PHQ8_Score"})
    rows = []
    for name, d in [("train", tr), ("dev", dv), ("test", te)]:
        for _, r in d.iterrows():
            rows.append(dict(pid=int(r.Participant_ID), split=name,
                             binary=int(r.PHQ8_Binary), score=int(r.PHQ8_Score)))
    return pd.DataFrame(rows)


def segments_for(pid, meta, unit):
    """Participant segments as a list of (start, stop). For session unit, [(None, None)]."""
    if unit == "session":
        return [(None, None)]
    t = pd.read_csv(f"{meta}/{pid}_TRANSCRIPT.csv", sep="\t").dropna(subset=["speaker"])
    t["speaker"] = t["speaker"].astype(str).str.strip()
    isp = t["speaker"].str.lower().str.startswith("participant")
    if isp.sum() == 0:
        return []
    if unit == "line":
        return [(float(r.start_time), float(r.stop_time)) for _, r in t[isp].iterrows()]
    grp = (isp != isp.shift()).cumsum()          # consecutive Participant lines = one turn
    return [(float(g.start_time.min()), float(g.stop_time.max()))
            for _, g in t[isp].groupby(grp)]


def chunk(segs, maxlen=MAX_CHUNK):
    """Split long segments evenly. Nothing is discarded."""
    out = []
    for s, e in segs:
        if s is None:
            out.append((s, e)); continue
        d = e - s
        if d <= maxlen:
            out.append((s, e))
        else:
            n = int(np.ceil(d / maxlen)); step = d / n
            out += [(s + i * step, s + (i + 1) * step) for i in range(n)]
    return out


class Emotion2Vec:
    """Via funasr, the official extraction path."""
    name = "emotion2vec_plus"

    def __init__(self, device):
        from funasr import AutoModel
        self.m = AutoModel(model="iic/emotion2vec_plus_large",
                           device=device, disable_update=True)

    def embed(self, wav):
        r = self.m.generate(wav, granularity="utterance",
                            extract_embedding=True, disable_pbar=True)
        return np.asarray(r[0]["feats"], dtype=np.float32)


class WhisperSER:
    """Via transformers. Uses the representation just before the classification layer
    (projector + mean pool)."""
    name = "whisper_ser"
    REPO = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"

    def __init__(self, device):
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        self.torch = torch
        self.device = device
        self.fe = AutoFeatureExtractor.from_pretrained(self.REPO)
        self.m = AutoModelForAudioClassification.from_pretrained(self.REPO).to(device).eval()

    def embed(self, wav):
        x = self.fe(wav, sampling_rate=SR, return_tensors="pt")
        with self.torch.no_grad():
            h = self.m.encoder(x.input_features.to(self.device))[0]
            h = self.m.projector(h)
            pooled = h.mean(dim=1)
        return pooled.squeeze(0).float().cpu().numpy()


MODELS = {"emotion2vec_plus": Emotion2Vec, "whisper_ser": WhisperSER}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--unit", default="turn", choices=["turn", "line", "session"])
    ap.add_argument("--data", default="/data/daic")
    ap.add_argument("--out", default="/work/emb")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="number of participants, for a smoke test")
    a = ap.parse_args()

    meta, audio = f"{a.data}/meta", f"{a.data}/audio"
    os.makedirs(a.out, exist_ok=True)
    labels = load_labels(meta)
    if a.limit:
        labels = labels.groupby("split").head(max(1, a.limit // 3))

    model = MODELS[a.model](a.device)
    vecs, keep, failed, t0 = {}, [], [], time.time()

    for i, r in enumerate(labels.itertuples(), 1):
        wav_path = f"{audio}/{r.pid}_AUDIO.wav"
        try:
            segs = chunk(segments_for(r.pid, meta, a.unit))
            if not segs:
                failed.append((r.pid, "no segments")); continue
            embs, durs = [], []
            for s, e in segs:
                if s is None:
                    w, _ = sf.read(wav_path, dtype="float32")
                    d = len(w) / SR
                else:
                    st, sp = int(s * SR), int(e * SR)
                    if sp <= st:
                        continue
                    w, _ = sf.read(wav_path, start=st, stop=sp, dtype="float32")
                    d = e - s
                if len(w) < SR * 0.05:       # models choke on anything under 50ms
                    continue
                embs.append(model.embed(w)); durs.append(d)
            if not embs:
                failed.append((r.pid, "0 valid segments")); continue
            W = np.asarray(durs, dtype=np.float64)
            vecs[str(r.pid)] = np.average(np.stack(embs), axis=0, weights=W).astype(np.float32)
            keep.append(r.pid)
            print(f"[{i}/{len(labels)}] {r.pid} {r.split:5s} "
                  f"{len(embs):4d} segments | {time.time()-t0:6.0f}s elapsed", flush=True)
        except Exception as ex:
            failed.append((r.pid, f"{type(ex).__name__}: {ex}"))
            print(f"[{i}/{len(labels)}] {r.pid} failed - {type(ex).__name__}", flush=True)

    tag = f"{a.model}_{a.unit}"
    np.savez_compressed(f"{a.out}/{tag}.npz", **vecs)
    meta_out = dict(model=a.model, unit=a.unit, max_chunk=MAX_CHUNK,
                    pooling="duration_weighted_mean", n_ok=len(keep),
                    failed=failed, elapsed_sec=round(time.time() - t0, 1),
                    dim=int(next(iter(vecs.values())).shape[0]) if vecs else 0)
    with open(f"{a.out}/{tag}.json", "w") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)
    print(f"\nsaved: {a.out}/{tag}.npz | {len(keep)} ok / {len(failed)} failed "
          f"| {meta_out['elapsed_sec']}s")
    if failed:
        print("failures:", failed[:10])


if __name__ == "__main__":
    main()
