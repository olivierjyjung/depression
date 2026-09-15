#!/bin/sh
# The two extractions saved 10792 and 10795 turns. If they are not the same turn set,
# the "identical inputs" claim in the model comparison is wrong.
docker exec ojjung_ser python -c "
import pandas as pd
a = pd.read_csv('/work/emb/emotion2vec_turnwise_meta.csv')
b = pd.read_csv('/work/emb/emotion2vec_s_turnwise_meta.csv')
print('emotion2vec+ turns', len(a), '| emotion2vec-S turns', len(b))
ka = set(zip(a.pid, a.turn)); kb = set(zip(b.pid, b.turn))
print('shared', len(ka & kb), '| only in plus', len(ka - kb), '| only in S', len(kb - ka))
print('only in plus:', sorted(ka - kb)[:10])
print('only in S:   ', sorted(kb - ka)[:10])
for pid, t in sorted(kb - ka)[:5]:
    r = b[(b.pid==pid) & (b.turn==t)].iloc[0]
    print(f'  S-only pid {pid} turn {t}: dur {r.dur:.3f}s bucket {r.bucket}')
"
