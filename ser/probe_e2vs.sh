#!/bin/sh
# What shape does extract_features actually return, and how does C2SER's own script use it?
set -e
docker exec ojjung_ser sh -c '
  echo "--- how C2SER handles utterance level ---"
  grep -n -A6 "utterance" /work/C2SER/Emotion2Vec-S/speech_feature_extraction.py | head -25
  echo "--- actual output shapes ---"
  . /opt/conda/etc/profile.d/conda.sh && conda activate py310
  cd /work && python - <<PY 2>&1 | tail -12
import sys, numpy as np, soundfile as sf, torch, fairseq
from dataclasses import dataclass
sys.path.insert(0, "/work")
@dataclass
class U: user_dir: str
fairseq.utils.import_user_module(U("/work/C2SER/Emotion2Vec-S/examples/data2vec/"))
m, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
    ["/work/C2SER/Emotion2Vec-S/ckpt/checkpoint.pt"])
m = m[0].to("cuda:1").eval()
w, _ = sf.read("/data/daic/audio/303_AUDIO.wav", start=0, stop=16000*5, dtype="float32")
x = torch.from_numpy(w).float().view(1, -1).to("cuda:1")
with torch.no_grad():
    out = m.extract_features(x)
for k, v in out.items():
    print(k, tuple(v.shape) if hasattr(v, "shape") else type(v))
PY
'
