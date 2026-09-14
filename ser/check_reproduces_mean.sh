#!/bin/sh
# The per-turn extraction should collapse back to the 09-12 speaker-mean vectors.
# If it does not, the new turn segmentation differs from extract_ser.py's.
docker exec ojjung_ser python -c "
import numpy as np, pandas as pd
old = np.load('/work/emb/emotion2vec_plus_turn.npz')
new = np.load('/work/emb/emotion2vec_turnwise.npz')
m = pd.read_csv('/work/emb/emotion2vec_turnwise_meta.csv')
d = []
for pid, g in m.groupby('pid'):
    k = str(pid)
    if k not in old.files or k not in new.files: continue
    V = new[k]
    if len(V) != len(g): print('turn count mismatch', pid, len(V), len(g)); continue
    mean = np.average(V, axis=0, weights=g.dur.values)
    o = old[k]
    d.append(np.abs(mean - o).max() / (np.abs(o).max() + 1e-12))
d = np.array(d)
print('speakers compared:', len(d))
print('max relative difference: %.2e | median %.2e' % (d.max(), np.median(d)))
print('verdict:', 'reproduces the 09-12 vectors' if d.max() < 1e-3 else 'DIFFERS - segmentation changed')
"
