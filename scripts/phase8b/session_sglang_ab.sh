#!/usr/bin/env bash
# Phase 8b — SGLang baseline vs kernelfuse treatment A/B.
set -euo pipefail
REPO="${REPO:-/workspace/KernelFuse}"
RESULTS_DIR="${RESULTS_DIR:-/workspace/kernelfuse_results/phase8b}"
export RESULTS_DIR KERNELFUSE_STATUS_PATH="${RESULTS_DIR}/STATUS.txt"
export PYTHONUNBUFFERED=1
export MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export REVISION="${REVISION:-a09a35458c702b33eeacc393d103063234e8bc28}"
export DTYPE=bfloat16 GPU_MEM_UTIL=0.90
export SGLANG_PIN="${SGLANG_PIN:-0.5.18}"
PORT=30000
export KERNELFUSE_SGLANG_BASE_URL="http://127.0.0.1:${PORT}/v1"

mkdir -p "$RESULTS_DIR"
log() { echo "$(date -Is) $*" | tee -a "$KERNELFUSE_STATUS_PATH"; }

cd "$REPO"
# Build kernelfuse against vLLM torch pins; run SGLang from its own venv (Phase 7).
# shellcheck disable=SC1091
source .venv-vllm/bin/activate
# shellcheck disable=SC1091
source scripts/phase8/env_torch_lib.sh
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
python setup.py build_ext --inplace >/dev/null 2>&1 || python setup.py build_ext --inplace
python -c "from kernelfuse._C import fused_add_rms_norm; print('kernelfuse ok')"

if [[ ! -x .venv-sglang/bin/python ]]; then
  python3 -m venv .venv-sglang
  # shellcheck disable=SC1091
  source .venv-sglang/bin/activate
  pip -q install "sglang[all]==${SGLANG_PIN}" pyyaml huggingface_hub tqdm pydantic
else
  # shellcheck disable=SC1091
  source .venv-sglang/bin/activate
fi
SGLANG_PY="${REPO}/.venv-sglang/bin/python"

log "patch SGLang layernorm"
"$SGLANG_PY" scripts/phase8b/patch_sglang_kernelfuse.py | tee -a "$KERNELFUSE_STATUS_PATH"

start_server() {
  local tag="$1"
  local use_kf="$2"
  local logfile="${RESULTS_DIR}/sglang_${tag}_server.log"
  pkill -f 'sglang.launch_server|vllm.entrypoints' 2>/dev/null || true
  sleep 3
  if [[ "$use_kf" == "1" ]]; then
    export KERNELFUSE_FUSED_ADD_RMSNORM=1
    log "starting SGLang treatment"
  else
    unset KERNELFUSE_FUSED_ADD_RMSNORM
    log "starting SGLang baseline"
  fi
  : > "$logfile"
  env PYTHONPATH="${REPO}:${PYTHONPATH:-}" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
    "$SGLANG_PY" -m sglang.launch_server \
    --model-path "$MODEL" --revision "$REVISION" \
    --host 0.0.0.0 --port "$PORT" --dtype "$DTYPE" \
    --mem-fraction-static "$GPU_MEM_UTIL" \
    >"$logfile" 2>&1 &
  echo $! >"${RESULTS_DIR}/sglang_server.pid"
  for i in $(seq 1 180); do
    if ! kill -0 "$(cat "${RESULTS_DIR}/sglang_server.pid")" 2>/dev/null; then
      log "SERVER_DIED tag=$tag"; tail -n 80 "$logfile" | tee -a "$KERNELFUSE_STATUS_PATH"; exit 1
    fi
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
      log "SERVER READY tag=$tag"; break
    fi
    (( i == 180 )) && { log "SERVER_TIMEOUT"; tail -n 80 "$logfile" | tee -a "$KERNELFUSE_STATUS_PATH"; exit 1; }
    sleep 5
  done
}

run_harness() {
  local tag="$1"
  log "harness tag=$tag"
  # shellcheck disable=SC1091
  source "${REPO}/.venv-vllm/bin/activate"
  python -m bench.runner \
    --matrix bench/config_matrix_phase8b_v1.yaml \
    --backends sglang \
    --out "${RESULTS_DIR}/results_phase8b_${tag}.csv" \
    --attention-backend flashinfer \
    --server-log "${RESULTS_DIR}/sglang_${tag}_server.log" \
    2>&1 | tee -a "${RESULTS_DIR}/harness_${tag}.log"
}

if [[ ! -f "${RESULTS_DIR}/results_phase8b_baseline.csv" ]]; then
  start_server baseline 0
  run_harness baseline
else
  log "skip baseline — existing ${RESULTS_DIR}/results_phase8b_baseline.csv"
fi

start_server treatment 1
run_harness treatment
log "PHASE8b A/B complete"
