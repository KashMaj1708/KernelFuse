#!/usr/bin/env bash
# Phase 7 — TensorRT-LLM via NGC container (avoid torch pin fights).
# Engine artifacts MUST land on the persistent volume (RESULTS_DIR).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RESULTS_DIR="${RESULTS_DIR:-/workspace/kernelfuse_results}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
REVISION="${REVISION:-a09a35458c702b33eeacc393d103063234e8bc28}"
DTYPE="${DTYPE:-bfloat16}"
PORT="${PORT:-8000}"
TRTLLM_IMAGE="${TRTLLM_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:latest}"
ENGINE_DIR="${ENGINE_DIR:-$RESULTS_DIR/trtllm_engine_qwen25_7b}"
MODE="${1:-full}"

mkdir -p "$RESULTS_DIR" "$ENGINE_DIR"
export KERNELFUSE_TRTLLM_BASE_URL="http://127.0.0.1:${PORT}/v1"

echo "=== Phase 7 TRT-LLM ($MODE) ==="
echo "image=$TRTLLM_IMAGE engine_dir=$ENGINE_DIR"
echo "NOTE: build the engine once onto $ENGINE_DIR; do not rebuild if it already exists."

if [[ ! -d "$ENGINE_DIR" ]] || [[ -z "$(ls -A "$ENGINE_DIR" 2>/dev/null || true)" ]]; then
  echo "Engine dir empty — build on the instance using the NGC image docs for $MODEL,"
  echo "writing checkpoints to $ENGINE_DIR, then re-run this script."
  echo "Placeholder exit: refusing to burn hours without an explicit build step."
  exit 3
fi

# Serve from container (OpenAI API). Adjust entrypoint to match the image tag.
docker run --rm --gpus all --network host \
  -v "$ENGINE_DIR:/engine" \
  -v "$RESULTS_DIR:/results" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  "$TRTLLM_IMAGE" \
  bash -lc "trtllm-serve /engine --host 0.0.0.0 --port ${PORT}" \
  >"$RESULTS_DIR/trtllm_server.log" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  docker ps -q --filter ancestor="$TRTLLM_IMAGE" | xargs -r docker kill || true
}
trap cleanup EXIT

for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    echo "SERVER READY"
    break
  fi
  sleep 5
done

# Harness uses host Python (mock/vllm client only needs stdlib+yaml).
OUT="$RESULTS_DIR/results_phase7_v1_trtllm.csv"
EXTRA=(--attention-backend flash_attn)
case "$MODE" in
  smoke)   EXTRA+=(--smoke); OUT="$RESULTS_DIR/results_phase7_v1_trtllm_smoke.csv" ;;
  prefill) EXTRA+=(--prefill-heavy); OUT="$RESULTS_DIR/results_phase7_v1_trtllm_prefill.csv" ;;
  full)    ;;
  *) echo "usage: $0 [smoke|full|prefill]"; exit 2 ;;
esac

python3 -m bench.runner \
  --matrix bench/config_matrix_phase7_v1.yaml \
  --backends trtllm \
  --out "$OUT" \
  "${EXTRA[@]}"

echo "DONE TRT-LLM -> $OUT"
echo "Keep engine at $ENGINE_DIR on the persistent volume before terminate."
