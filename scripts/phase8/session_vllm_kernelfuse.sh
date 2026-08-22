#!/usr/bin/env bash
# Phase 8 — patch vLLM, serve Qwen, identity smoke with kernelfuse op (graphs on).
set -euo pipefail
REPO="${REPO:-/workspace/KernelFuse}"
RESULTS_DIR="${RESULTS_DIR:-/workspace/kernelfuse_results/phase8}"
export RESULTS_DIR KERNELFUSE_STATUS_PATH="${RESULTS_DIR}/STATUS.txt"
export PYTHONUNBUFFERED=1
export KERNELFUSE_FUSED_ADD_RMSNORM=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export REVISION="${REVISION:-a09a35458c702b33eeacc393d103063234e8bc28}"
export DTYPE=bfloat16 GPU_MEM_UTIL=0.90 MAX_MODEL_LEN=4096

mkdir -p "$RESULTS_DIR"
log() { echo "$(date -Is) $*" | tee -a "$KERNELFUSE_STATUS_PATH"; }

cd "$REPO"
# shellcheck disable=SC1091
source .venv-vllm/bin/activate

log "patch vLLM layernorm for kernelfuse op"
python scripts/phase8/patch_vllm_kernelfuse.py | tee -a "$KERNELFUSE_STATUS_PATH"

pkill -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
sleep 2
LOG="${RESULTS_DIR}/vllm_kernelfuse_server.log"
: > "$LOG"
log "starting vLLM server KERNELFUSE_FUSED_ADD_RMSNORM=1"
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --revision "$REVISION" \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype "$DTYPE" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  >"$LOG" 2>&1 &
echo $! >"${RESULTS_DIR}/vllm_server.pid"

for i in $(seq 1 180); do
  if ! kill -0 "$(cat "${RESULTS_DIR}/vllm_server.pid")" 2>/dev/null; then
    log "SERVER_DIED"
    tail -n 80 "$LOG" | tee -a "$KERNELFUSE_STATUS_PATH"
    exit 1
  fi
  if curl -sf http://127.0.0.1:8000/v1/models >/dev/null; then
    log "SERVER READY"
    break
  fi
  if (( i == 180 )); then
    log "SERVER_TIMEOUT"
    tail -n 80 "$LOG" | tee -a "$KERNELFUSE_STATUS_PATH"
    exit 1
  fi
  sleep 5
done

grep -E 'Using .+ attention|FlashAttention|CUDA graph' "$LOG" | tail -n 5 | tee -a "$KERNELFUSE_STATUS_PATH" || true

log "identity smoke with kernelfuse op"
export KERNELFUSE_VLLM_BASE_URL=http://127.0.0.1:8000/v1
python -m bench.runner \
  --matrix bench/config_matrix_phase8_v1.yaml \
  --backends vllm \
  --smoke \
  --out "${RESULTS_DIR}/results_phase8_kernelfuse_smoke.csv" \
  --attention-backend FLASH_ATTN \
  2>&1 | tee -a "${RESULTS_DIR}/harness_kernelfuse_smoke.log"

log "PHASE8 vLLM+kernelfuse smoke complete"
