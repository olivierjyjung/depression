#!/bin/sh
# timm pulled a PyPI torchvision that does not match torch 2.14.0+cu126, so its C++ ops
# fail to register. Reinstall torchvision from the same wheel index as torch.
set -e
docker exec ojjung_ser sh -c '
  . /opt/conda/etc/profile.d/conda.sh && conda activate py310
  pip install -q --root-user-action=ignore --force-reinstall --no-deps torchvision \
      --index-url https://download.pytorch.org/whl/cu126 2>&1 | tail -3
  python -c "
import torch, torchvision, timm
from timm.models.vision_transformer import DropPath, Mlp
print(\"torch\", torch.__version__, \"| torchvision\", torchvision.__version__, \"| timm\", timm.__version__)
print(\"DropPath/Mlp importable\")
" 2>&1 | tail -5
'
