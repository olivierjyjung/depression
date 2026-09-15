#!/bin/sh
# Bucket contrasts on the emotion2vec+ turn embeddings. No fairseq, no GPU.
set -e
docker exec ojjung_ser python /work/emb_contrast.py \
  --npz /work/emb/emotion2vec_turnwise.npz \
  --meta /work/emb/emotion2vec_turnwise_meta.csv \
  --data /data/daic --tag emotion2vec_plus --out /work/eval \
  --repeats 10 --boot 40000
