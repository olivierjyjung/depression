#!/bin/sh
# Emotion2Vec-S smoke test: 3 speakers, in the py310 env where fairseq works.
set -e
docker exec ojjung_ser sh -c '
  ls -d /work/C2SER/Emotion2Vec-S/examples/data2vec/ 2>/dev/null || echo "MISSING model_dir"
  . /opt/conda/etc/profile.d/conda.sh && conda activate py310
  cd /work && python /work/extract_turns_e2vs.py \
    --ckpt /work/C2SER/Emotion2Vec-S/ckpt/checkpoint.pt \
    --model-dir /work/C2SER/Emotion2Vec-S/examples/data2vec/ \
    --data /data/daic --out /work/emb --device cuda:1 --limit 3
'
