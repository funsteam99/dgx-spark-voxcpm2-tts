#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

python benchmark_voxcpm.py \
  --output-dir runs/baseline \
  --prompts prompts/baseline.jsonl \
  --no-denoiser

python benchmark_voxcpm.py \
  --output-dir runs/voice-design \
  --prompts prompts/voice_design.jsonl \
  --no-denoiser
