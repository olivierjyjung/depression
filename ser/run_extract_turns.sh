#!/bin/sh
# Full per-turn extraction, all 189 speakers. Container sees host GPUs 1,2 as cuda:0,1.
set -e
docker exec ojjung_ser python /work/extract_turns_ser.py \
  --data /data/daic --out /work/emb --device cuda:1
