#!/usr/bin/env bash
set -euo pipefail
cd /workspace/KernelFuse
source .venv-vllm/bin/activate
source scripts/phase8/env_torch_lib.sh
export PYTHONPATH=/workspace/KernelFuse
mkdir -p /workspace/kernelfuse_results/phase8 /workspace/kernelfuse_results/phase9
touch kernels/__init__.py kernels/attention/__init__.py

python -c "from kernelfuse._C import fused_add_rms_norm; print('kf ok')"

echo "=== MICROBENCH no-graph ==="
python scripts/phase8/kernel_microbench.py --rows 1 --cols 3584 \
  --out /workspace/kernelfuse_results/phase8/kernel_microbench_events_1x3584.csv
python scripts/phase8/kernel_microbench.py --rows 32 --cols 3584 \
  --out /workspace/kernelfuse_results/phase8/kernel_microbench_events_32x3584.csv

echo "=== MICROBENCH graph ==="
python scripts/phase8/kernel_microbench.py --rows 1 --cols 3584 --graph \
  --out /workspace/kernelfuse_results/phase8/kernel_microbench_events_graph_1x3584.csv
python scripts/phase8/kernel_microbench.py --rows 32 --cols 3584 --graph \
  --out /workspace/kernelfuse_results/phase8/kernel_microbench_events_graph_32x3584.csv

echo "=== PHASE9 CORRECTNESS ==="
python scripts/phase9/decode_attn_correctness.py \
  --out /workspace/kernelfuse_results/phase9/decode_attn_correctness.csv

echo "=== PHASE9 EVENT TIMING (ref vs load_inline cuda) ==="
python - <<'PY'
import time
import torch
from torch.utils.cpp_extension import load_inline
from kernels.attention.decode_attn_ref import decode_attn_ref

# reuse same kernel as correctness script via import path
import importlib.util
spec = importlib.util.spec_from_file_location(
    "corr", "/workspace/KernelFuse/scripts/phase9/decode_attn_correctness.py"
)
# Just time ref with CUDA events
def bench_ref(seq, dim=128, iters=100, trials=11, warmup=20):
    q = torch.randn(dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(seq, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(seq, dim, device="cuda", dtype=torch.bfloat16)
    for _ in range(warmup):
        decode_attn_ref(q, k, v)
    torch.cuda.synchronize()
    samples = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            decode_attn_ref(q, k, v)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1e3 / iters)
    samples.sort()
    print(f"torch_ref_events seq={seq} min_us={samples[0]:.2f} median_us={samples[len(samples)//2]:.2f}")

for seq in (512, 2048):
    bench_ref(seq)
PY

echo ALL_DONE
