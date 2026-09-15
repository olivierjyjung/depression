#!/bin/sh
# data2vec2's AltBlock does `from timm.models.vision_transformer import DropPath, Mlp`,
# which is the pre-0.9 timm layout (0.9 moved those into timm.layers). Pin 0.6.x.
set -e
docker exec ojjung_ser sh -c '
  . /opt/conda/etc/profile.d/conda.sh && conda activate py310
  pip install -q --root-user-action=ignore "timm==0.6.13" 2>&1 | tail -3
  python -c "
import timm
from timm.models.vision_transformer import DropPath, Mlp
print(\"timm\", timm.__version__, \"| DropPath/Mlp importable\")
"
'
