#!/bin/sh
# Install a compiler and fairseq for Emotion2Vec-S, inside a venv so the container's
# working environment (funasr, the emotion2vec+ path) stays untouched.
# fairseq pins omegaconf<2.1, whose metadata pip>=24.1 rejects, so pip is held at 24.0
# inside the venv -- the resolver error message itself says to use pip<24.1.
set -e
echo "--- compiler ---"
docker exec ojjung_ser sh -c 'apt-get update -qq && apt-get install -y -qq g++ git >/dev/null 2>&1; g++ --version | head -1'
echo "--- fairseq from git main into /tmp/fsvenv ---"
docker exec ojjung_ser sh -c '
  rm -rf /tmp/fsvenv /tmp/fairseq
  python -m venv --system-site-packages /tmp/fsvenv
  /tmp/fsvenv/bin/pip install -q "pip==24.0" setuptools wheel
  git clone -q --depth 1 https://github.com/facebookresearch/fairseq /tmp/fairseq
  # not editable: fairseq setup.py does not support PEP 660 editable builds, and we are
  # not modifying fairseq anyway
  cd /tmp/fairseq && timeout 900 /tmp/fsvenv/bin/pip install -q --no-build-isolation . 2>&1 | tail -15
'
echo "--- import check ---"
docker exec ojjung_ser sh -c '/tmp/fsvenv/bin/python -c "
import fairseq, torch, numpy
print(\"fairseq\", fairseq.__version__, \"| torch\", torch.__version__, \"| numpy\", numpy.__version__)
"' 2>&1 | tail -5
