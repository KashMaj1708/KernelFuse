#!/usr/bin/env bash
# Phase 8 preamble — output equiv, variance repeat, cache cold/warm pair.
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

cd "$REPO"
# shellcheck disable=SC1091
source .venv-vllm/bin/activate

log "PHASE8 preamble begin"
python scripts/phase8/patch_vllm_kernelfuse.py | tee -a "$KERNELFUSE_STATUS_PATH"

log "1/3 output equivalence baseline vs treatment"
python scripts/phase8/output_equiv.py --results-dir "$RESULTS_DIR" \
  | tee -a "${RESULTS_DIR}/output_equiv.log"

log "2/3 variance repeat — smoke in=128 out=128 c=1 × 3 (baseline path)"
pkill -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
sleep 3
unset KERNELFUSE_FUSED_ADD_RMSNORM
LOG="${RESULTS_DIR}/vllm_preamble_server.log"
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --revision "$REVISION" \
  --host 0.0.0.0 --port 8000 --dtype "$DTYPE" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" --max-model-len "$MAX_MODEL_LEN" \
  >"$LOG" 2>&1 &
echo $! >"${RESULTS_DIR}/vllm_server.pid"
for i in $(seq 1 120); do
  curl -sf http://127.0.0.1:8000/v1/models >/dev/null && break
  sleep 5
  (( i == 120 )) && { log "preamble server timeout"; exit 1; }
done
export KERNELFUSE_VLLM_BASE_URL=http://127.0.0.1:8000/v1
for rep in 1 2 3; do
  log "variance repeat $rep/3"
  python -m bench.runner \
    --matrix bench/config_matrix_phase8_v1.yaml \
    --backends vllm --smoke \
    --out "${RESULTS_DIR}/results_phase8_variance_rep${rep}.csv" \
    --attention-backend FLASH_ATTN \
    --server-log "$LOG" \
    2>&1 | tee -a "${RESULTS_DIR}/harness_variance_rep${rep}.log"
done

log "3/3 cache pair in=2048 — definitive cold via SERVER RESTART, then warm (no flush)"
# HTTP flush alone no-ops on warm vLLM (Phase 8: ok=False, 25.6% hit). Restart is the cold gate.
pkill -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
sleep 3
unset KERNELFUSE_FUSED_ADD_RMSNORM
LOG="${RESULTS_DIR}/vllm_preamble_cold_server.log"
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --revision "$REVISION" \
  --host 0.0.0.0 --port 8000 --dtype "$DTYPE" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" --max-model-len "$MAX_MODEL_LEN" \
  >"$LOG" 2>&1 &
echo $! >"${RESULTS_DIR}/vllm_server.pid"
for i in $(seq 1 120); do
  curl -sf http://127.0.0.1:8000/v1/models >/dev/null && break
  sleep 5
  (( i == 120 )) && { log "cold server timeout"; exit 1; }
done

python -m bench.runner \
  --matrix bench/config_matrix_phase8_v1.yaml \
  --backends vllm --cache-experiment \
  --no-flush-cache-between-cells \
  --assert-cold-hit --max-cold-hit-pct 15 \
  --out "${RESULTS_DIR}/results_phase8_cache_cold.csv" \
  --attention-backend FLASH_ATTN \
  --server-log "$LOG" \
  2>&1 | tee -a "${RESULTS_DIR}/harness_cache_cold.log"

python -m bench.runner \
  --matrix bench/config_matrix_phase8_v1.yaml \
  --backends vllm --cache-experiment \
  --no-flush-cache-between-cells \
  --out "${RESULTS_DIR}/results_phase8_cache_warm.csv" \
  --attention-backend FLASH_ATTN \
  --server-log "$LOG" \
  2>&1 | tee -a "${RESULTS_DIR}/harness_cache_warm.log"

log "PHASE8 preamble complete"
