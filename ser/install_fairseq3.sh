#!/bin/sh
# fairseq generates C++ from Cython .pyx sources at build time, and --no-build-isolation
# means Cython has to be present in the venv itself.
set -e
docker exec ojjung_ser sh -c '
  /tmp/fsvenv/bin/pip install -q "cython" numpy
  cd /tmp/fairseq
  /tmp/fsvenv/bin/pip install --no-build-isolation . > /tmp/fairseq_build.log 2>&1 || true
  grep -nE "^ERROR|error:|fatal|Successfully installed" /tmp/fairseq_build.log | head -15
  echo "--- import check ---"
  /tmp/fsvenv/bin/python -c "import fairseq, torch; print(\"fairseq\", fairseq.__version__, \"torch\", torch.__version__)" 2>&1 | tail -3
'
