#!/usr/bin/env bash
# Section 2 environment. CPU wheels: this machine's NVIDIA driver did not
# initialize (NVML driver/library version mismatch), so the CUDA wheel cannot run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir -r requirements.txt
mkdir -p checkpoints
if [[ ! -f checkpoints/sam_vit_b_01ec64.pth ]]; then
  wget -O checkpoints/sam_vit_b_01ec64.pth \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
fi
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
PY
ls -lh checkpoints/sam_vit_b_01ec64.pth
