#!/bin/sh
# Fetch the Emotion2Vec-S checkpoint (Apache-2.0, ASLP-lab) and the C2SER extraction code.
set -e
echo "--- HF repo file list ---"
docker exec ojjung_ser python -c "
import json, urllib.request
u='https://huggingface.co/api/models/ASLP-lab/Emotion2Vec-S'
d=json.load(urllib.request.urlopen(u))
print('license:', (d.get('cardData') or {}).get('license'))
for s in d.get('siblings', []): print(' ', s['rfilename'])
"
echo "--- clone C2SER ---"
docker exec ojjung_ser sh -c '
  rm -rf /work/C2SER && git clone -q --depth 1 https://github.com/zxzhao0/C2SER /work/C2SER
  ls /work/C2SER
  echo "-- Emotion2Vec-S dir --"
  ls /work/C2SER/Emotion2Vec-S 2>/dev/null || echo "(no such dir)"
'
