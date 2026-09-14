#!/usr/bin/env python3
"""Participant turns with the Ellie question each one answers.

A turn is a run of consecutive Participant lines (the unit the supervisor fixed on
2026-09-11), so each turn is preceded by exactly one Ellie prompt. The prompt's question
id comes from the transcript when it is tagged as `question_id (spoken text)`, and
otherwise from a dictionary of canonical texts learned from the 124 tagged sessions.
Follow-ups and acknowledgements keep the topic of the question before them.

Shared by the extraction and feature scripts so both use the same turn boundaries.
"""
import glob, os, re
import numpy as np, pandas as pd

from question_buckets import bucket_of

TAGGED = re.compile(r"^\s*([a-z0-9_]+)\s*\((.+)\)\s*$", re.I)


def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"<[^>]*>", " ", s)
    s = re.sub(r"[^a-z' ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def read_transcript(path):
    t = pd.read_csv(path, sep="\t").dropna(subset=["speaker"])
    t["speaker"] = t["speaker"].astype(str).str.strip().str.lower()
    return t.sort_values("start_time").reset_index(drop=True)


def learn_text2id(meta):
    """id -> canonical text dictionary, from the sessions that tag their prompts."""
    text2id = {}
    for f in sorted(glob.glob(f"{meta}/*_TRANSCRIPT.csv")):
        t = read_transcript(f)
        for v in t[~t.speaker.str.startswith("participant")].value.fillna("").astype(str):
            m = TAGGED.match(v)
            if m:
                spoken = norm(m.group(2))
                if spoken:
                    text2id.setdefault(spoken, m.group(1).lower())
    return text2id


def turns_with_questions(pid, meta, text2id):
    """DataFrame of one row per participant turn: start, stop, dur, qid, bucket."""
    t = read_transcript(f"{meta}/{pid}_TRANSCRIPT.csv")
    is_p = t.speaker.str.startswith("participant").values
    start = t.start_time.astype(float).values
    stop = t.stop_time.astype(float).values
    vals = t.value.fillna("").astype(str).values

    cur_qid, cur_bucket, rows = None, None, []
    i = 0
    while i < len(t):
        if not is_p[i]:
            m = TAGGED.match(vals[i])
            qid = m.group(1).lower() if m else text2id.get(norm(vals[i]))
            if qid:
                b = bucket_of(qid)
                if b in ("followup", "backchannel"):
                    pass                       # topic unchanged
                elif b == "other":
                    cur_qid = qid              # keep the id, bucket stays as it was
                else:
                    cur_qid, cur_bucket = qid, b
            i += 1
            continue
        j = i
        while j < len(t) and is_p[j]:          # a run of Participant lines = one turn
            j += 1
        # response latency: this turn starts this long after the previous speaker stopped
        lat = float(start[i] - stop[i - 1]) if i > 0 else np.nan
        rows.append(dict(pid=pid, start=float(start[i]), stop=float(stop[j - 1]),
                         lat=lat, qid=cur_qid, bucket=cur_bucket or "other"))
        i = j

    d = pd.DataFrame(rows)
    if len(d):
        d["dur"] = d.stop - d.start
        d["turn"] = np.arange(len(d))
    return d


def all_turns(meta, pids):
    text2id = learn_text2id(meta)
    return pd.concat([turns_with_questions(p, meta, text2id) for p in pids],
                     ignore_index=True), text2id
