#!/bin/sh
# Build the per-speaker turn features from the per-turn extraction.
set -e
docker exec ojjung_ser python /work/turn_features.py \
  --meta-csv /work/emb/emotion2vec_turnwise_meta.csv \
  --data /data/daic --out /work/eval/turn_features.csv
