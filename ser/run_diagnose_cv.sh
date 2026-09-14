#!/bin/sh
set -e
docker exec ojjung_ser python /work/diagnose_cv.py \
  --features /work/eval/turn_features.csv --data /data/daic \
  --cols lat_median neg_mean lat_symptom_vs_neutral
