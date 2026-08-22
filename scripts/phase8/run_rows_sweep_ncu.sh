#!/usr/bin/env bash
# ncu DRAM read compare: kernelfuse vs vLLM fused_add_rms_norm (one launch each).
set -euo pipefail
cd /workspace/KernelFuse
source .venv-vllm/bin/activate
source scripts/phase8/env_torch_lib.sh
export PYTHONPATH=/workspace/KernelFuse
export TORCH_CUDA_ARCH_LIST=8.0
OUT=/workspace/kernelfuse_results/phase8
mkdir -p "$OUT"

# Rebuild register-resident kernel
python - <<'PY'
import sys
import torch.utils.cpp_extension as cpp
cpp._check_cuda_version = lambda *a, **k: None
sys.argv = ["setup.py", "build_ext", "--inplace"]
import runpy
runpy.run_path("setup.py", run_name="__main__")
print("rebuild ok")
PY

python -c "from kernelfuse._C import fused_add_rms_norm; print('kf', fused_add_rms_norm)"

# Isolated one-shot launches for ncu (avoid capturing python overhead kernels)
cat > /tmp/ncu_kf.py <<'PY'
import torch, kernelfuse
x = torch.randn(1, 3584, device="cuda", dtype=torch.bfloat16)
r = torch.randn(1, 3584, device="cuda", dtype=torch.bfloat16)
w = torch.ones(3584, device="cuda", dtype=torch.bfloat16)
# warmup outside ncu interest
for _ in range(10):
    kernelfuse.fused_add_rms_norm(x, r, w, 1e-6)
torch.cuda.synchronize()
torch.cuda.nvtx.range_push("kf_launch")
kernelfuse.fused_add_rms_norm(x, r, w, 1e-6)
torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
PY

cat > /tmp/ncu_vllm.py <<'PY'
import torch
from vllm import _custom_ops as ops
x = torch.randn(1, 3584, device="cuda", dtype=torch.bfloat16)
r = torch.randn(1, 3584, device="cuda", dtype=torch.bfloat16)
w = torch.ones(3584, device="cuda", dtype=torch.bfloat16)
for _ in range(10):
    ops.fused_add_rms_norm(x, r, w, 1e-6)
torch.cuda.synchronize()
torch.cuda.nvtx.range_push("vllm_launch")
ops.fused_add_rms_norm(x, r, w, 1e-6)
torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
PY

METRICS="dram__bytes_read.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,gpu__time_duration.sum"

echo "=== ncu kernelfuse ==="
ncu --metrics "$METRICS" --target-processes all \
  --kernel-name-base demangled \
  --launch-skip 10 --launch-count 1 \
  python /tmp/ncu_kf.py 2>&1 | tee "$OUT/ncu_kernelfuse.txt" || true

echo "=== ncu vllm ==="
ncu --metrics "$METRICS" --target-processes all \
  --kernel-name-base demangled \
  --launch-skip 10 --launch-count 1 \
  python /tmp/ncu_vllm.py 2>&1 | tee "$OUT/ncu_vllm.txt" || true

echo "=== rows sweep graph ==="
python scripts/phase8/kernel_microbench.py --graph \
  --out "$OUT/kernel_rows_sweep_graph.csv" | tee "$OUT/kernel_rows_sweep_graph.log"

echo "=== rows sweep no-graph ==="
python scripts/phase8/kernel_microbench.py \
  --out "$OUT/kernel_rows_sweep_nograph.csv" | tee "$OUT/kernel_rows_sweep_nograph.log"

echo "=== phase9 correctness + mutation ==="
python scripts/phase9/decode_attn_correctness.py \
  --out /workspace/kernelfuse_results/phase9/decode_attn_correctness.csv

echo DONE
