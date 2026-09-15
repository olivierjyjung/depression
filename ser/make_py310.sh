#!/bin/sh
# fairseq 0.12.2 (with hydra 1.0.7 / omegaconf 2.0.x) cannot run on Python 3.11: the
# 3.11 dataclass rules and omegaconf 2.0's type handling contradict each other, so
# patching one breaks the other. The supported combination is Python <= 3.10, built here
# as a separate conda env so the working 3.11 environment (funasr, emotion2vec+) is
# untouched. numpy is held below 2 because fairseq 0.12.2 predates it.
set -e
docker exec ojjung_ser sh -c '
  . /opt/conda/etc/profile.d/conda.sh
  conda create -y -q -n py310 python=3.10 >/dev/null 2>&1
  conda activate py310
  python -V
  # PyPI packages and the PyTorch wheel index must be installed separately: passing
  # --index-url makes pip look *only* there, and cython/numpy are not hosted on it.
  pip install -q --root-user-action=ignore "numpy<2" cython soundfile pandas 2>&1 | tail -3
  pip install -q --root-user-action=ignore torch torchaudio \
      --index-url https://download.pytorch.org/whl/cu126 2>&1 | tail -3
  pip install -q --root-user-action=ignore "pip==24.0" 2>&1 | tail -2
  python -c "import torch, numpy; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), \"numpy\", numpy.__version__)"
'
