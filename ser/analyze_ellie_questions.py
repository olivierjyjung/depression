#!/usr/bin/env python3
"""Can participant turns be aligned across speakers by the Ellie question they answer?

Feature idea (1) in DECISIONS.md 2026-09-14 rests on a claim from the 09-02 supervisor
brief: Ellie asks ~100 fixed questions in a fixed order. That claim decides whether
question-aligned pooling is even possible, so it is measured here before anything is
built: how many distinct Ellie prompts there are, how many recur across sessions, how
stable their order is, and what share of participant speech sits under a recurring
prompt.
"""
import argparse, glob, os, re
from collections import Counter, defaultdict
import numpy as np, pandas as pd

# Ellie's short acknowledgements are not questions; they carry no topic to align on.
BACKCHANNEL = re.compile(
    r"^(mhm|mm|uh huh|um|hmm|okay|ok|yeah|yes|right|sure|alright|awesome|nice|cool|"
    r"good|great|wow|i see|that's good|that's great|thank you|thanks|really|"
    r"interesting|sorry|oh)[\s.,!?]*$")


def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"<[^>]*>", " ", s)          # <sync>, <laughter> etc.
    s = re.sub(r"[^a-z' ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/Desktop/depression/wwwdaicwoz"))
    ap.add_argument("--out", default="ser/ellie_questions.csv")
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()

    files = sorted(glob.glob(f"{a.data}/*_TRANSCRIPT.csv"))
    per_session, order, p_turn_rows = Counter(), defaultdict(list), []
    n_sess = 0

    for f in files:
        pid = os.path.basename(f).split("_")[0]
        if not pid.isdigit():
            continue
        t = pd.read_csv(f, sep="\t").dropna(subset=["speaker"])
        t["speaker"] = t["speaker"].astype(str).str.strip().str.lower()
        t = t.sort_values("start_time").reset_index(drop=True)
        is_p = t["speaker"].str.startswith("participant").values
        txt = [norm(v) for v in t["value"].fillna("")]
        n_sess += 1

        seen = set()
        # one Ellie prompt = the concatenation of consecutive Ellie lines
        cur, prompts = [], []            # prompts: (question_text, row index where it ends)
        for i in range(len(t)):
            if not is_p[i]:
                cur.append(txt[i])
            elif cur:
                prompts.append((" ".join(cur).strip(), i))
                cur = []
        for q, _ in prompts:
            if not q or BACKCHANNEL.match(q):
                continue
            if q not in seen:
                per_session[q] += 1      # count each prompt once per session
                seen.add(q)
        for rank, (q, _) in enumerate(prompts):
            if q and not BACKCHANNEL.match(q):
                order[q].append(rank / max(1, len(prompts) - 1))

        # participant speech time, and how much of it sits under a non-backchannel prompt
        dur = (t.stop_time.astype(float) - t.start_time.astype(float)).values
        last_q = None
        for i in range(len(t)):
            if not is_p[i]:
                last_q = txt[i] if not BACKCHANNEL.match(txt[i]) else last_q
            else:
                p_turn_rows.append((pid, dur[i], last_q))

    q = pd.DataFrame({"question": list(per_session), "n_sessions": list(per_session.values())})
    q["share_sessions"] = q.n_sessions / n_sess
    q["median_position"] = [float(np.median(order[x])) for x in q.question]
    q["position_iqr"] = [float(np.percentile(order[x], 75) - np.percentile(order[x], 25))
                         for x in q.question]
    q = q.sort_values("n_sessions", ascending=False).reset_index(drop=True)
    q.to_csv(a.out, index=False)

    print(f"sessions: {n_sess}")
    print(f"distinct Ellie prompts (backchannels removed): {len(q)}")
    for thr in [0.9, 0.5, 0.25, 0.1]:
        print(f"  prompts present in >={thr:.0%} of sessions: {(q.share_sessions >= thr).sum()}")

    pt = pd.DataFrame(p_turn_rows, columns=["pid", "dur", "q"])
    tot = pt.dur.sum()
    common = set(q[q.share_sessions >= 0.5].question)
    print(f"\nparticipant speech: {tot/3600:.1f} h total")
    print(f"  under a prompt seen in >=50% of sessions: "
          f"{pt[pt.q.isin(common)].dur.sum()/tot:.1%} of speech time")
    print(f"  under any non-backchannel prompt:         "
          f"{pt[pt.q.notna()].dur.sum()/tot:.1%}")

    freq = q[q.share_sessions >= 0.5]
    if len(freq):
        print(f"\norder stability among the {len(freq)} common prompts: "
              f"median position IQR {freq.position_iqr.median():.3f} "
              f"(0 = same slot in every session, 1 = anywhere)")
    print(f"\n--- top {a.top} recurring prompts ---")
    show = q.head(a.top).copy()
    show["question"] = show.question.str.slice(0, 62)
    print(show[["n_sessions", "share_sessions", "median_position",
                "position_iqr", "question"]].to_string(index=False,
                                                       float_format=lambda v: f"{v:.2f}"))
    print(f"\nsaved: {a.out}")


if __name__ == "__main__":
    main()
