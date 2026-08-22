#!/usr/bin/env bash
set -e
cd /workspace/KernelFuse
source .venv-vllm/bin/activate
source scripts/phase8/env_torch_lib.sh
python -c "import torch; print('torch', torch.__version__)"
export PYTHONPATH=/workspace/KernelFuse
SO=$(find .venv-vllm/lib -name '_C*.so' -path '*kernelfuse*' 2>/dev/null | head -1)
SO=${SO:-kernelfuse/_C.cpython-312-x86_64-linux-gnu.so}
echo "SO=$SO"
ldd "$SO" | head -15 || true
python scripts/phase8/test_kf_import.py
