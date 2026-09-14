#!/bin/sh
# Official split + 189-speaker repeated CV + paired bootstrap over the feature sets.
set -e
docker exec ojjung_ser python /work/eval_features.py \
  --features /work/eval/turn_features.csv \
  --data /data/daic --out /work/eval --repeats 20 --boot 10000
