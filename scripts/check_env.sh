#!/usr/bin/env bash
set -u

echo "== System =="
date
uname -a || true
cat /etc/os-release || true

echo
echo "== CPU / Memory =="
lscpu || true
free -h || true

echo
echo "== NVIDIA =="
command -v nvidia-smi || true
nvidia-smi || true

echo
echo "== CUDA =="
command -v nvcc || true
nvcc --version || true

echo
echo "== Python =="
command -v python3 || true
python3 --version || true
command -v pip3 || true
pip3 --version || true

echo
echo "== Docker =="
command -v docker || true
docker --version || true
docker info 2>/dev/null | sed -n '1,80p' || true

echo
echo "== PyTorch Probe =="
python3 - <<'PY' || true
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"device {i}:", torch.cuda.get_device_name(i))
except Exception as exc:
    print("torch probe failed:", repr(exc))
PY
