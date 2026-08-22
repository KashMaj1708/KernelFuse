#!/usr/bin/env bash
# Phase 9 — decode attention microbench on A100 + torch reference compare.
set -euo pipefail
REPO="${REPO:-/workspace/KernelFuse}"
RESULTS_DIR="${RESULTS_DIR:-/workspace/kernelfuse_results/phase9}"
export RESULTS_DIR KERNELFUSE_STATUS_PATH="${RESULTS_DIR}/STATUS.txt"
mkdir -p "$RESULTS_DIR"
log() { echo "$(date -Is) $*" | tee -a "$KERNELFUSE_STATUS_PATH"; }

cd "$REPO"
log "PHASE9 build decode_attn_bench"
nvcc -O3 -o "${RESULTS_DIR}/decode_attn_bench" kernels/attention/decode_attn_bench.cu
log "PHASE9 microbench sweep"
for seq in 512 2048; do
  "${RESULTS_DIR}/decode_attn_bench" "$seq" 128 | tee -a "$KERNELFUSE_STATUS_PATH"
done

log "PHASE9 correctness CSV vs decode_attn_ref"
# shellcheck disable=SC1091
source .venv-vllm/bin/activate
# shellcheck disable=SC1091
source scripts/phase8/env_torch_lib.sh
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
touch kernels/__init__.py kernels/attention/__init__.py
python scripts/phase9/decode_attn_correctness.py \
  --out "${RESULTS_DIR}/decode_attn_correctness.csv" \
  | tee -a "$KERNELFUSE_STATUS_PATH"

python - <<'PY' | tee -a "$KERNELFUSE_STATUS_PATH"
import time, torch
from pathlib import Path
import sys
sys.path.insert(0, "/workspace/KernelFuse")
from kernels.attention.decode_attn_ref import decode_attn_ref

def bench(seq, dim, iters=200):
    q = torch.randn(dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(seq, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(seq, dim, device="cuda", dtype=torch.bfloat16)
    for _ in range(20):
        decode_attn_ref(q, k, v)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        decode_attn_ref(q, k, v)
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / iters * 1e6
    print(f"torch_ref seq={seq} dim={dim} avg_us={us:.2f}")

for seq in (512, 2048):
    bench(seq, 128)
PY

log "PHASE9 complete"
