#!/bin/sh
# Feasibility gate for Emotion2Vec-S: it loads through fairseq, which often fails to
# build on Python 3.11 / torch 2.x. Tested in a throwaway venv so the container's
# working environment is left alone.
echo "--- container python / torch ---"
docker exec ojjung_ser python -c "import sys, torch; print(sys.version); print('torch', torch.__version__)"
echo "--- is fairseq already present? ---"
docker exec ojjung_ser python -c "import fairseq; print('fairseq', fairseq.__version__)" 2>&1 | tail -2
echo "--- try installing into a throwaway venv (5 min cap) ---"
docker exec ojjung_ser sh -c '
  rm -rf /tmp/fsvenv && python -m venv --system-site-packages /tmp/fsvenv
  timeout 300 /tmp/fsvenv/bin/pip install --no-input -q fairseq 2>&1 | tail -20
  /tmp/fsvenv/bin/python -c "import fairseq; print(\"OK fairseq\", fairseq.__version__)" 2>&1 | tail -3
'
