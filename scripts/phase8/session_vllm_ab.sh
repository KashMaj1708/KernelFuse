#!/usr/bin/env bash
# Phase 8 — baseline vs kernelfuse treatment A/B on same vLLM server pins.
set -euo pipefail
REPO="${REPO:-/workspace/KernelFuse}"
RESULTS_DIR="${RESULTS_DIR:-/workspace/kernelfuse_results/phase8}"
export RESULTS_DIR KERNELFUSE_STATUS_PATH="${RESULTS_DIR}/STATUS.txt"
export PYTHONUNBUFFERED=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export REVISION="${REVISION:-a09a35458c702b33eeacc393d103063234e8bc28}"
export DTYPE=bfloat16 GPU_MEM_UTIL=0.90 MAX_MODEL_LEN=4096

mkdir -p "$RESULTS_DIR"
log() { echo "$(date -Is) $*" | tee -a "$KERNELFUSE_STATUS_PATH"; }

start_server() {
  local tag="$1"
  local use_kf="$2"
  local logfile="${RESULTS_DIR}/vllm_${tag}_server.log"
  pkill -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
  sleep 3
  : > "$logfile"
  if [[ "$use_kf" == "1" ]]; then
    export KERNELFUSE_FUSED_ADD_RMSNORM=1
    log "starting vLLM treatment KERNELFUSE_FUSED_ADD_RMSNORM=1"
  else
    unset KERNELFUSE_FUSED_ADD_RMSNORM
    log "starting vLLM baseline (stock fused_add_rms_norm path)"
  fi
  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --revision "$REVISION" \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype "$DTYPE" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    >"$logfile" 2>&1 &
  echo $! >"${RESULTS_DIR}/vllm_server.pid"
  for i in $(seq 1 180); do
    if ! kill -0 "$(cat "${RESULTS_DIR}/vllm_server.pid")" 2>/dev/null; then
      log "SERVER_DIED tag=$tag"
      tail -n 80 "$logfile" | tee -a "$KERNELFUSE_STATUS_PATH"
      exit 1
    fi
    if curl -sf http://127.0.0.1:8000/v1/models >/dev/null; then
      log "SERVER READY tag=$tag"
      break
    fi
    if (( i == 180 )); then
      log "SERVER_TIMEOUT tag=$tag"
      tail -n 80 "$logfile" | tee -a "$KERNELFUSE_STATUS_PATH"
      exit 1
    fi
    sleep 5
  done
  grep -E 'Graph capturing finished|use_cudagraph' "$logfile" | tail -n 3 | tee -a "$KERNELFUSE_STATUS_PATH" || true
}

run_harness() {
  local tag="$1"
  export KERNELFUSE_VLLM_BASE_URL=http://127.0.0.1:8000/v1
  log "harness sweep tag=$tag"
  python -m bench.runner \
    --matrix bench/config_matrix_phase8_v1.yaml \
    --backends vllm \
    --out "${RESULTS_DIR}/results_phase8_${tag}.csv" \
    --attention-backend FLASH_ATTN \
    --server-log "${RESULTS_DIR}/vllm_${tag}_server.log" \
    2>&1 | tee -a "${RESULTS_DIR}/harness_${tag}.log"
}

cd "$REPO"
# shellcheck disable=SC1091
source .venv-vllm/bin/activate

log "PHASE8 A/B begin — patch vLLM if needed"
python scripts/phase8/patch_vllm_kernelfuse.py | tee -a "$KERNELFUSE_STATUS_PATH"

log "kernel microbench (no server)"
python scripts/phase8/kernel_microbench.py \
  --rows 1 --cols 3584 \
  --out "${RESULTS_DIR}/kernel_microbench_3584.csv" \
  | tee -a "$KERNELFUSE_STATUS_PATH"
python scripts/phase8/kernel_microbench.py \
  --rows 32 --cols 3584 \
  --out "${RESULTS_DIR}/kernel_microbench_32x3584.csv" \
  | tee -a "$KERNELFUSE_STATUS_PATH"

start_server baseline 0
run_harness baseline

start_server treatment 1
run_harness treatment

log "PHASE8 A/B complete"
