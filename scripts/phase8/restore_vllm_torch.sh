#!/usr/bin/env bash
# Restore Phase 7 vLLM torch pins after accidental sglang upgrade.
set -euo pipefail
cd /workspace/KernelFuse
source .venv-vllm/bin/activate
pip -q install "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \
  --index-url https://download.pytorch.org/whl/cu124
source scripts/phase8/env_torch_lib.sh
export PYTHONPATH=/workspace/KernelFuse
python setup.py build_ext --inplace
python -c "from kernelfuse._C import fused_add_rms_norm; print('kernelfuse ok')"
