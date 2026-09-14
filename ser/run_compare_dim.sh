#!/bin/sh
# 1024-dim speaker-mean embedding vs a single scalar from the same model, same instrument.
set -e
docker exec ojjung_ser python /work/compare_dim.py \
  --features /work/eval/turn_features.csv \
  --emb /work/emb/emotion2vec_plus_turn.npz \
  --data /data/daic --out /work/eval/dim_comparison.json
