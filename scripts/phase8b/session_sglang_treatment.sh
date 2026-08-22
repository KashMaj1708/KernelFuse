#!/usr/bin/env bash
# Phase 8b treatment-only: build kernelfuse against SGLang torch ABI, serve, harness.
set -euo pipefail
REPO="${REPO:-/workspace/KernelFuse}"
RESULTS_DIR="${RESULTS_DIR:-/workspace/kernelfuse_results/phase8b}"
export RESULTS_DIR KERNELFUSE_STATUS_PATH="${RESULTS_DIR}/STATUS.txt"
export PYTHONUNBUFFERED=1
export MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export REVISION="${REVISION:-a09a35458c702b33eeacc393d103063234e8bc28}"
export DTYPE=bfloat16 GPU_MEM_UTIL=0.90
PORT=30000
export KERNELFUSE_SGLANG_BASE_URL="http://127.0.0.1:${PORT}/v1"

mkdir -p "$RESULTS_DIR"
log() { echo "$(date -Is) $*" | tee -a "$KERNELFUSE_STATUS_PATH"; }

cd "$REPO"
# shellcheck disable=SC1091
source .venv-sglang/bin/activate
TORCH_LIB=$(python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

log "SGLang torch=$(python -c 'import torch; print(torch.__version__, torch.version.cuda)')"
log "build kernelfuse against SGLang torch (ABI match)"
rm -f kernelfuse/_C*.so build/lib.*/kernelfuse/_C*.so 2>/dev/null || true
# Host toolkit is often 12.8 while SGLang ships torch+cu130 — skip strict check for A100 sm_80 build.
python - <<'PY' 2>&1 | tee -a "${RESULTS_DIR}/build_sglang.log"
import sys
import torch.utils.cpp_extension as cpp
cpp._check_cuda_version = lambda *a, **k: None  # noqa: E731
sys.argv = ["setup.py", "build_ext", "--inplace"]
import runpy
runpy.run_path("setup.py", run_name="__main__")
PY
python -c "from kernelfuse._C import fused_add_rms_norm; print('kernelfuse ok', fused_add_rms_norm)"

log "patch SGLang layernorm"
python scripts/phase8b/patch_sglang_kernelfuse.py | tee -a "$KERNELFUSE_STATUS_PATH"

pkill -f 'sglang.launch_server|vllm.entrypoints' 2>/dev/null || true
sleep 3
export KERNELFUSE_FUSED_ADD_RMSNORM=1
LOG="${RESULTS_DIR}/sglang_treatment_server.log"
: > "$LOG"
log "starting SGLang treatment KERNELFUSE_FUSED_ADD_RMSNORM=1"
env PYTHONPATH="${REPO}" LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
  KERNELFUSE_FUSED_ADD_RMSNORM=1 \
  python -m sglang.launch_server \
  --model-path "$MODEL" --revision "$REVISION" \
  --host 0.0.0.0 --port "$PORT" --dtype "$DTYPE" \
  --mem-fraction-static "$GPU_MEM_UTIL" \
  >"$LOG" 2>&1 &
echo $! >"${RESULTS_DIR}/sglang_server.pid"

for i in $(seq 1 180); do
  if ! kill -0 "$(cat "${RESULTS_DIR}/sglang_server.pid")" 2>/dev/null; then
    log "SERVER_DIED"
    tail -n 100 "$LOG" | tee -a "$KERNELFUSE_STATUS_PATH"
    exit 1
  fi
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    log "SERVER READY treatment"
    break
  fi
  (( i == 180 )) && { log "SERVER_TIMEOUT"; tail -n 100 "$LOG" | tee -a "$KERNELFUSE_STATUS_PATH"; exit 1; }
  sleep 5
done

# Harness from vLLM venv (runner deps) against SGLang OpenAI server
# shellcheck disable=SC1091
source .venv-vllm/bin/activate
# shellcheck disable=SC1091
source scripts/phase8/env_torch_lib.sh
log "harness treatment"
python -m bench.runner \
  --matrix bench/config_matrix_phase8b_v1.yaml \
  --backends sglang \
  --out "${RESULTS_DIR}/results_phase8b_treatment.csv" \
  --attention-backend flashinfer \
  --server-log "$LOG" \
  2>&1 | tee -a "${RESULTS_DIR}/harness_treatment.log"

log "PHASE8b treatment complete"
