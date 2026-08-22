#!/usr/bin/env bash
set -euo pipefail
cd /workspace/KernelFuse
source .venv-vllm/bin/activate
source scripts/phase8/env_torch_lib.sh
export PYTHONPATH=/workspace/KernelFuse
export TORCH_CUDA_ARCH_LIST=8.0

python - <<'PY'
import torch
from torch.utils.cpp_extension import load_inline
from kernels.attention.decode_attn_ref import decode_attn_ref
import importlib.util

# Load the same CUDA source as correctness
spec = importlib.util.spec_from_file_location(
    "dac", "scripts/phase9/decode_attn_correctness.py"
)
dac = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dac)

mod = load_inline(
    name="decode_attn_corr2",
    cpp_sources="torch::Tensor decode_attn_cuda(torch::Tensor, torch::Tensor, torch::Tensor);",
    cuda_sources=dac._CUDA_SRC,
    functions=["decode_attn_cuda"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)

def bench(fn, seq, dim=128, iters=50, trials=11, warmup=20):
    q = torch.randn(dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(seq, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(seq, dim, device="cuda", dtype=torch.bfloat16)
    for _ in range(warmup):
        fn(q, k, v)
    torch.cuda.synchronize()
    samples = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn(q, k, v)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1e3 / iters)
    samples.sort()
    return samples[0], samples[len(samples)//2]

import csv
from pathlib import Path
out = Path("/workspace/kernelfuse_results/phase9/decode_attn_events.csv")
rows = []
for seq in (512, 2048):
    rmin, rmed = bench(lambda q,k,v: decode_attn_ref(q,k,v), seq)
    cmin, cmed = bench(lambda q,k,v: mod.decode_attn_cuda(q.contiguous(), k.contiguous(), v.contiguous()), seq)
    print(f"seq={seq} torch_ref min={rmin:.2f}us  cuda_kernel min={cmin:.2f}us  ratio_cuda/ref={cmin/rmin:.2f}")
    rows.append({"seq": seq, "torch_ref_min_us": f"{rmin:.4f}", "torch_ref_median_us": f"{rmed:.4f}",
                 "cuda_min_us": f"{cmin:.4f}", "cuda_median_us": f"{cmed:.4f}",
                 "cuda_over_ref": f"{cmin/rmin:.4f}"})
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("wrote", out)

# sanity: non-zero diff on adversarial? print one max_abs again with more precision
q = torch.randn(128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(512, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn(512, 128, device="cuda", dtype=torch.bfloat16)
ref = decode_attn_ref(q,k,v)
out_t = mod.decode_attn_cuda(q.contiguous(), k.contiguous(), v.contiguous())
d = (out_t.float()-ref.float()).abs().max().item()
print(f"spot max_abs={d:.8e}")
PY
