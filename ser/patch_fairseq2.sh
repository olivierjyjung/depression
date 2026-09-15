#!/bin/sh
# Apply the Python 3.11 dataclass fix to fairseq's config module, then check the import.
# 11 fields in FairseqConfig use another dataclass instance as a default; 3.11 requires
# default_factory. `field` is already imported in that module.
set -e
docker exec ojjung_ser sh -c '
  F=/tmp/fsvenv/lib/python3.11/site-packages/fairseq/dataclass/configs.py
  cp -n "$F" "$F.orig"
  sed -i -E "s/^([[:space:]]+)([a-z_]+): ([A-Za-z_]+) = \\3\\(\\)[[:space:]]*\$/\\1\\2: \\3 = field(default_factory=\\3)/" "$F"
  echo "--- after patch ---"
  grep -nE "default_factory" "$F" | head -15
  echo "--- remaining mutable defaults ---"
  grep -cE "^[[:space:]]+[a-z_]+: [A-Za-z_]+ = [A-Za-z_]+\(\)[[:space:]]*\$" "$F" || true
  echo "--- import check ---"
  /tmp/fsvenv/bin/python -c "
import fairseq, torch
print(\"fairseq\", fairseq.__version__, \"| torch\", torch.__version__)
import fairseq.checkpoint_utils, fairseq.utils
print(\"checkpoint_utils OK\")
" 2>&1 | tail -6
'
