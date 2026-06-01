#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install -U pip setuptools wheel

# Let pip choose the best platform wheel first. On DGX Spark images with
# preinstalled PyTorch, this will usually be a no-op or a compatible upgrade.
python -m pip install -U torch torchaudio torchcodec

# voxcpm is the upstream TTS runtime package from OpenBMB. This repository
# only wraps it with SPARK setup scripts, benchmark tooling, and a Web UI.
# The VoxCPM2 model weights are downloaded separately into pretrained_models/.
python -m pip install -U voxcpm soundfile pandas numpy modelscope

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
