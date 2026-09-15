#!/bin/sh
# Download the Emotion2Vec-S checkpoint (Apache-2.0) to where C2SER's loader expects it.
set -e
docker exec ojjung_ser sh -c '
  mkdir -p /work/C2SER/Emotion2Vec-S/ckpt
  cd /work/C2SER/Emotion2Vec-S/ckpt
  if [ -s checkpoint.pt ]; then echo "already present"; else
    python - <<PY
import urllib.request, shutil
u = "https://huggingface.co/ASLP-lab/Emotion2Vec-S/resolve/main/checkpoint.pt"
with urllib.request.urlopen(u) as r, open("checkpoint.pt", "wb") as f:
    shutil.copyfileobj(r, f)
PY
  fi
  ls -la checkpoint.pt
'
