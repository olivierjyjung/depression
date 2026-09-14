#!/bin/sh
# Smoke test: 3 speakers, per-turn extraction, to check GPU fit and output shape.
set -e
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
docker exec ojjung_ser python /work/extract_turns_ser.py \
  --data /data/daic --out /work/emb --device cuda:1 --limit 3
