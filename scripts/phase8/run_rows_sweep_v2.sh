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

# Correctness smoke vs vLLM
python - <<'PY'
import torch
import kernelfuse
from vllm import _custom_ops as ops
torch.manual_seed(0)
for rows in [1, 32, 512, 4096]:
    x0 = torch.randn(rows, 3584, device="cuda", dtype=torch.bfloat16)
    r0 = torch.randn(rows, 3584, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(3584, device="cuda", dtype=torch.bfloat16)
    xk, rk = x0.clone(), r0.clone()
    xv, rv = x0.clone(), r0.clone()
    kernelfuse.fused_add_rms_norm(xk, rk, w, 1e-6)
    ops.fused_add_rms_norm(xv, rv, w, 1e-6)
    dx = (xk.float() - xv.float()).abs().max().item()
    dr = (rk.float() - rv.float()).abs().max().item()
    print(f"rows={rows} max_abs x={dx:.4e} r={dr:.4e}")
PY

echo "=== rows sweep graph (tight regs) ==="
python scripts/phase8/kernel_microbench.py --graph \
  --out "$OUT/kernel_rows_sweep_graph_v2.csv" | tee "$OUT/kernel_rows_sweep_graph_v2.log"

echo "=== rows sweep no-graph ==="
python scripts/phase8/kernel_microbench.py \
  --out "$OUT/kernel_rows_sweep_nograph_v2.csv" | tee "$OUT/kernel_rows_sweep_nograph_v2.log"

echo DONE
