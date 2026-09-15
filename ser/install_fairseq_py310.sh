#!/bin/sh
# fairseq into the py310 env, where its hydra/omegaconf pins are actually supported.
# pip stays at 24.0 so the old omegaconf metadata is still accepted, and the build needs
# g++ (already installed) plus Cython under --no-build-isolation.
set -e
docker exec ojjung_ser sh -c '
  . /opt/conda/etc/profile.d/conda.sh && conda activate py310
  [ -d /tmp/fairseq ] || git clone -q --depth 1 https://github.com/facebookresearch/fairseq /tmp/fairseq
  cd /tmp/fairseq && rm -rf build
  pip install --root-user-action=ignore --no-build-isolation . > /tmp/fs310.log 2>&1 || true
  grep -nE "^ERROR|error:|Successfully installed" /tmp/fs310.log | head -8
  echo "--- import check ---"
  python -c "
import fairseq, fairseq.checkpoint_utils, fairseq.utils, torch, torchaudio
print(\"OK fairseq\", fairseq.__version__, \"| torch\", torch.__version__)
" 2>&1 | tail -6
'
