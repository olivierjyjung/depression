#!/bin/sh
# hydra-core 1.0.7 (pinned by fairseq) has the same Python 3.11 dataclass problem.
# Same mechanical fix. This is the last third-party patch attempt -- if another library
# breaks after this, Emotion2Vec-S is reported as blocked by the environment instead.
set -e
docker exec ojjung_ser sh -c '
  for F in /tmp/fsvenv/lib/python3.11/site-packages/hydra/conf/__init__.py \
           /tmp/fsvenv/lib/python3.11/site-packages/hydra/core/*.py; do
    [ -f "$F" ] || continue
    cp -n "$F" "$F.orig" 2>/dev/null || true
    sed -i -E "s/^([[:space:]]+)([a-z_]+): ([A-Za-z_.]+) = \\3\\(\\)[[:space:]]*\$/\\1\\2: \\3 = field(default_factory=\\3)/" "$F"
  done
  H=/tmp/fsvenv/lib/python3.11/site-packages/hydra/conf/__init__.py
  grep -n "from dataclasses" "$H" | head -2
  grep -nE "default_factory|= [A-Za-z_.]+\(\)[[:space:]]*\$" "$H" | head -20
  echo "--- import check ---"
  /tmp/fsvenv/bin/python -c "
import fairseq, fairseq.checkpoint_utils, fairseq.utils, torch
print(\"OK fairseq\", fairseq.__version__, \"torch\", torch.__version__)
" 2>&1 | tail -6
'
