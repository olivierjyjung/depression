#!/bin/sh
# Diagnose the 3 speakers with no latency value and the negative-latency share.
set -e
docker exec ojjung_ser python -c "
import pandas as pd, numpy as np
d = pd.read_csv('/work/eval/timing_features.csv')
print('--- speakers with no latency ---')
print(d[d.lat.isna()][['pid','split','y','n_lat','n_pause']].to_string(index=False))
for p in d[d.lat.isna()].pid:
    t = pd.read_csv(f'/data/daic/meta/{p}_TRANSCRIPT.csv', sep='\t')
    print(f'pid {p}: rows {len(t)}, speaker values {sorted(set(t.speaker.dropna().astype(str).str.strip()))[:6]}')
print()
print('--- latency distribution over speakers ---')
print(d.lat.describe().to_string())
print()
print('--- negative latency share per speaker ---')
print((d.n_lat_negative / d.n_lat).describe().to_string())
print('speakers where >50% of latencies are negative:', int(((d.n_lat_negative/d.n_lat)>0.5).sum()))
"
