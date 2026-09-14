#!/usr/bin/env python3
"""Recover a question id for every Ellie prompt, so participant turns can be aligned.

Some DAIC sessions write Ellie's prompts as `question_id (spoken text)`; most write only
the spoken text. This builds the id -> canonical-text dictionary from the tagged sessions
and matches the untagged ones against it, then reports how much of each session's
participant speech ends up under a known question id.

Output: ser/question_map.csv (id, canonical text, how many sessions) and
ser/turn_question.csv (one row per participant turn, with the question id it answers).
"""
import argparse, glob, os, re
from collections import Counter, defaultdict
import numpy as np, pandas as pd

TAGGED = re.compile(r"^\s*([a-z0-9_]+)\s*\((.+)\)\s*$", re.I)
BACKCHANNEL = re.compile(
    r"^(mhm|mm|uh huh|um|hmm|okay|ok|yeah|yes|right|sure|alright|awesome|nice|cool|"
    r"good|great|wow|i see|that's good|that's great|thank you|thanks|really|"
    r"interesting|sorry|oh)[\s.,!?]*$")


def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"<[^>]*>", " ", s)
    s = re.sub(r"[^a-z' ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def rows_of(path):
    t = pd.read_csv(path, sep="\t").dropna(subset=["speaker"])
    t["speaker"] = t["speaker"].astype(str).str.strip().str.lower()
    return t.sort_values("start_time").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/Desktop/depression/wwwdaicwoz"))
    ap.add_argument("--outdir", default="ser")
    a = ap.parse_args()

    files = [f for f in sorted(glob.glob(f"{a.data}/*_TRANSCRIPT.csv"))
             if os.path.basename(f).split("_")[0].isdigit()]

    # pass 1 -- learn id -> canonical text from the tagged sessions
    text2id, id_sessions, tagged_sessions = {}, Counter(), set()
    for f in files:
        pid = os.path.basename(f).split("_")[0]
        t = rows_of(f)
        for v in t[~t.speaker.str.startswith("participant")].value.fillna(""):
            m = TAGGED.match(str(v))
            if m:
                qid, spoken = m.group(1).lower(), norm(m.group(2))
                if spoken:
                    text2id.setdefault(spoken, qid)
                    tagged_sessions.add(pid)
    print(f"sessions: {len(files)} | sessions with id-tagged prompts: {len(tagged_sessions)}")
    print(f"distinct question ids learned: {len(set(text2id.values()))} "
          f"from {len(text2id)} canonical texts")

    # pass 2 -- assign an id to every prompt, tagged or not, and attribute turns
    turn_rows, cover = [], []
    for f in files:
        pid = int(os.path.basename(f).split("_")[0])
        t = rows_of(f)
        is_p = t.speaker.str.startswith("participant").values
        start = t.start_time.astype(float).values
        stop = t.stop_time.astype(float).values
        vals = t.value.fillna("").astype(str).values

        cur_id, cur_src = None, None
        seg = []                                  # (start, stop, qid, src)
        for i in range(len(t)):
            if not is_p[i]:
                m = TAGGED.match(vals[i])
                if m:
                    cur_id, cur_src = m.group(1).lower(), "tagged"
                else:
                    n = norm(vals[i])
                    if BACKCHANNEL.match(n) or not n:
                        continue                  # acknowledgement: topic unchanged
                    cur_id = text2id.get(n)
                    cur_src = "matched" if cur_id else None
                    if cur_id is None:
                        cur_id, cur_src = f"UNK:{n[:40]}", "unmatched"
            elif is_p[i]:
                seg.append((start[i], stop[i], cur_id, cur_src))

        d = pd.DataFrame(seg, columns=["start", "stop", "qid", "src"])
        d["dur"] = d.stop - d.start
        d["pid"] = pid
        turn_rows.append(d)
        known = d.src.isin(["tagged", "matched"])
        cover.append(dict(pid=pid, n_turns=len(d), speech=d.dur.sum(),
                          share_known=float(d.dur[known].sum() / max(d.dur.sum(), 1e-9)),
                          n_ids=int(d.qid[known].nunique())))

    turns = pd.concat(turn_rows, ignore_index=True)
    cov = pd.DataFrame(cover)
    turns.to_csv(f"{a.outdir}/turn_question.csv", index=False)

    print(f"\nparticipant speech attributed to a known question id: "
          f"{cov.share_known.median():.1%} median per speaker "
          f"(min {cov.share_known.min():.1%}, "
          f"{(cov.share_known >= 0.5).sum()}/{len(cov)} speakers above 50%)")
    print(f"distinct known ids per speaker: median {cov.n_ids.median():.0f}")

    known = turns[turns.src.isin(["tagged", "matched"])]
    g = known.groupby("qid").agg(n_speakers=("pid", "nunique"),
                                 speech_h=("dur", lambda s: s.sum() / 3600),
                                 median_turn_s=("dur", "median")).reset_index()
    g["share_speakers"] = g.n_speakers / len(cov)
    g = g.sort_values("n_speakers", ascending=False).reset_index(drop=True)
    g.to_csv(f"{a.outdir}/question_map.csv", index=False)

    print(f"\ndistinct question ids appearing in the data: {len(g)}")
    for thr in [0.9, 0.75, 0.5, 0.25]:
        print(f"  ids reaching >={thr:.0%} of speakers: {(g.share_speakers >= thr).sum()}")
    print(f"\n--- top 30 question ids by speaker coverage ---")
    print(g.head(30).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nsaved: {a.outdir}/question_map.csv, {a.outdir}/turn_question.csv")


if __name__ == "__main__":
    main()
