#!/bin/sh
# fairseq 0.12.2 cannot be imported on Python 3.11: its config dataclasses use other
# dataclass instances as defaults, which 3.11 rejects. The fix is mechanical -- turn
# `x: XConfig = XConfig()` into `x: XConfig = field(default_factory=XConfig)`.
set -e
docker exec ojjung_ser sh -c '
  F=$(/tmp/fsvenv/bin/python -c "import fairseq, os; print(os.path.dirname(fairseq.__file__))" 2>/dev/null \
      || ls -d /tmp/fsvenv/lib/python3.11/site-packages/fairseq)
  echo "fairseq dir: $F"
  C="$F/dataclass/configs.py"
  ls -la "$C"
  echo "--- imports ---"
  grep -n "from dataclasses" "$C" | head -3
  echo "--- candidate lines ---"
  grep -nE "^[[:space:]]+[a-z_]+:[[:space:]]*[A-Za-z_]+[[:space:]]*=[[:space:]]*[A-Za-z_]+\(\)" "$C" | head -30
'
