#!/usr/bin/env bash
# Phase 8 — build kernelfuse extension + op smoke on rented A100 (same pins as Phase 7).
set -euo pipefail
REPO="${REPO:-/workspace/KernelFuse}"
RESULTS_DIR="${RESULTS_DIR:-/workspace/kernelfuse_results/phase8}"
export RESULTS_DIR KERNELFUSE_STATUS_PATH="${RESULTS_DIR}/STATUS.txt"
export PYTHONUNBUFFERED=1

mkdir -p "$RESULTS_DIR"
log() { echo "$(date -Is) $*" | tee -a "$KERNELFUSE_STATUS_PATH"; }

cd "$REPO"
log "PHASE8 build kernelfuse extension"

if [[ -f .venv-vllm/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv-vllm/bin/activate
  log "using existing .venv-vllm"
else
  python3 -m venv .venv-vllm
  # shellcheck disable=SC1091
  source .venv-vllm/bin/activate
  python -m pip -q install -U pip setuptools wheel ninja
  python -m pip install -q "torch==2.6.0" --index-url "https://download.pytorch.org/whl/cu124"
  python -m pip install -q "vllm==0.8.5" pyyaml huggingface_hub tqdm
fi

python -m pip -q install -U pip setuptools wheel ninja 2>/dev/null || true
log "pip install -e . (kernelfuse extension)"
python -m pip install -q -e . --no-build-isolation 2>&1 | tee -a "${RESULTS_DIR}/pip_kernelfuse.log"

python - <<'PY' | tee -a "$KERNELFUSE_STATUS_PATH"
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("capability", torch.cuda.get_device_capability(0))
import kernelfuse
from kernelfuse import fused_add_rms_norm
assert fused_add_rms_norm is not None, "extension import failed"

rows, cols = 4, 3584
x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
r = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
w = torch.ones(cols, device="cuda", dtype=torch.bfloat16)
x0, r0 = x.clone(), r.clone()

xf = x0.float() + r0.float()
r_ref = xf.to(torch.bfloat16)
out_ref = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16) * w

fused_add_rms_norm(x, r, w, 1e-6)
torch.cuda.synchronize()
max_x = (x - out_ref).float().abs().max().item()
max_r = (r - r_ref).float().abs().max().item()
print(f"op_smoke max_err x={max_x:.4e} r={max_r:.4e}")
assert max_x < 0.05 and max_r < 0.05, "op smoke failed"
print("PHASE8_OP_SMOKE OK")
PY

log "PHASE8 extension build + op smoke complete"
