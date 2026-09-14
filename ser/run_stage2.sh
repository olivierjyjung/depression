#!/bin/sh
# Run stage 2 (timing + SER fusion) inside the ojjung_ser container on allen.
# Copied to /home/ojjung/ser_work/ (= container /work) and invoked as: sh run_stage2.sh
set -e
docker exec ojjung_ser python /work/fuse_ser.py \
  --ser /work/eval/emotion2vec_plus_turn_scores.csv /work/eval/whisper_ser_turn_scores.csv \
  --data /data/daic --out /work/eval
