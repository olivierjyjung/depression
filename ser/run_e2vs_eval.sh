#!/bin/sh
# 1) the single pre-registered judgement for Emotion2Vec-S: lat+mean vs lat_only, cv189
# 2) the descriptive model comparison the supervisor asked for
set -e
docker exec ojjung_ser sh -c '
  cd /work
  echo "############ pre-registered: Emotion2Vec-S lat+mean vs lat_only ############"
  python /work/emb_contrast.py \
    --npz /work/emb/emotion2vec_s_turnwise.npz \
    --meta /work/emb/emotion2vec_s_turnwise_meta.csv \
    --data /data/daic --tag emotion2vec_s --out /work/eval \
    --repeats 10 --boot 40000 --prereg "lat+mean"
  echo
  echo "############ model comparison ############"
  python /work/compare_models.py --data /data/daic --out /work/eval/model_comparison.json \
    --repeats 10 --boot 40000
'
