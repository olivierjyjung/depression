#!/bin/sh
# What GPUs does the ojjung_ser container actually see?
echo "--- host GPUs ---"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
echo "--- container NVIDIA_VISIBLE_DEVICES ---"
docker inspect ojjung_ser --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -i -E "nvidia|cuda" || echo "(none)"
echo "--- torch view inside container ---"
docker exec ojjung_ser python -c "import torch; n=torch.cuda.device_count(); print('device_count', n); [print(i, torch.cuda.get_device_name(i), round(torch.cuda.mem_get_info(i)[0]/2**30,1), 'GiB free') for i in range(n)]"
