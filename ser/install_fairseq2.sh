#!/bin/sh
# Full-output fairseq build attempt, so the real compile error is visible.
set -e
docker exec ojjung_ser sh -c '
  cd /tmp/fairseq
  /tmp/fsvenv/bin/pip install --no-build-isolation . > /tmp/fairseq_build.log 2>&1 || true
  echo "--- lines mentioning error ---"
  grep -nE "error|Error|ERROR|fatal" /tmp/fairseq_build.log | grep -v "^.*copying" | head -25
'
