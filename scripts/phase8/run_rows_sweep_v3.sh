#!/usr/bin/env bash
set -euo pipefail
cd /workspace/KernelFuse
source .venv-vllm/bin/activate
source scripts/phase8/env_torch_lib.sh
export PYTHONPATH=/workspace/KernelFuse
export TORCH_CUDA_ARCH_LIST=8.0
OUT=/workspace/kernelfuse_results/phase8
mkdir -p "$OUT"

python - <<'PY'
import sys
import torch.utils.cpp_extension as cpp
cpp._check_cuda_version = lambda *a, **k: None
sys.argv = ["setup.py", "build_ext", "--inplace"]
import runpy
runpy.run_path("setup.py", run_name="__main__")
print("rebuild ok")
PY

nvcc -c kernelfuse/csrc/fused_add_rms_norm_op.cu -o /tmp/kf_op.o \
  -I kernelfuse/csrc \
  -I .venv-vllm/lib/python3.12/site-packages/torch/include \
  -I .venv-vllm/lib/python3.12/site-packages/torch/include/torch/csrc/api/include \
  -I /usr/local/cuda/include -I .venv-vllm/include -I /usr/include/python3.12 \
  -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ \
  -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -O3 --use_fast_math \
  -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=_C -D_GLIBCXX_USE_CXX11_ABI=0 \
  -gencode=arch=compute_80,code=sm_80 -std=c++17 -Xptxas=-v 2>&1 | tee "$OUT/ptxas_v3.txt" | grep -E "stack|spill|registers|vec8|scalar" || true

python - <<'PY'
import torch
import kernelfuse
from vllm import _custom_ops as ops
torch.manual_seed(0)
for rows in [1, 512, 4096]:
    x0 = torch.randn(rows, 3584, device="cuda", dtype=torch.bfloat16)
    r0 = torch.randn(rows, 3584, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(3584, device="cuda", dtype=torch.bfloat16)
    xk, rk = x0.clone(), r0.clone()
    xv, rv = x0.clone(), r0.clone()
    kernelfuse.fused_add_rms_norm(xk, rk, w, 1e-6)
    ops.fused_add_rms_norm(xv, rv, w, 1e-6)
    dx = (xk.float() - xv.float()).abs().max().item()
    dr = (rk.float() - rv.float()).abs().max().item()
    print(f"rows={rows} dx={dx:.4e} dr={dr:.4e}")
PY

python scripts/phase8/kernel_microbench.py --graph \
  --out "$OUT/kernel_rows_sweep_graph_v3.csv" | tee "$OUT/kernel_rows_sweep_graph_v3.log"

echo DONE
