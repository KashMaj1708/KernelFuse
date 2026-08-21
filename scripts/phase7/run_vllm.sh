#!/usr/bin/env bash
# Phase 7 — vLLM: install (optional) → serve → harness → tear down.
# Harness talks over HTTP; this venv is isolated from SGLang/TRT.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RESULTS_DIR="${RESULTS_DIR:-/workspace/kernelfuse_results}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
REVISION="${REVISION:-a09a35458c702b33eeacc393d103063234e8bc28}"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
PORT="${PORT:-8000}"
VLLM_VERSION="${VLLM_VERSION:-0.8.5}"
VENV="${VENV:-$ROOT/.venv-vllm}"
MODE="${1:-full}"   # smoke | full | prefill | serve-only

mkdir -p "$RESULTS_DIR"
export KERNELFUSE_VLLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

echo "=== Phase 7 vLLM ($MODE) ==="
echo "model=$MODEL rev=$REVISION dtype=$DTYPE enforce_eager=false pin=vllm==$VLLM_VERSION"

if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip -q install -U pip
python -m pip -q install "vllm==${VLLM_VERSION}" pyyaml huggingface_hub transformers

LOG="$RESULTS_DIR/vllm_server.log"
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --revision "$REVISION" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --no-enable-log-requests \
  >"$LOG" 2>&1 &
SERVER_PID=$!
echo "server pid=$SERVER_PID log=$LOG"

cleanup() {
  echo "tearing down vLLM pid=$SERVER_PID"
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for readiness
for i in $(seq 1 120); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    tail -n 80 "$LOG" || true
    exit 1
  fi
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    echo "SERVER READY"
    break
  fi
  sleep 5
done

# Guess attention from log
ATTN="flash_attn"
if grep -qi "flash" "$LOG"; then ATTN="flash_attn"; fi
if grep -qi "triton" "$LOG"; then ATTN="triton_attn"; fi
if grep -qi "xformers" "$LOG"; then ATTN="xformers"; fi
echo "attention_backend_guess=$ATTN"

EXTRA=(--attention-backend "$ATTN")
OUT="$RESULTS_DIR/results_phase7_v1_vllm.csv"
case "$MODE" in
  smoke)   EXTRA+=(--smoke); OUT="$RESULTS_DIR/results_phase7_v1_vllm_smoke.csv" ;;
  prefill) EXTRA+=(--prefill-heavy); OUT="$RESULTS_DIR/results_phase7_v1_vllm_prefill.csv" ;;
  full)    ;;
  serve-only) echo "serve-only; Ctrl-C to stop"; wait "$SERVER_PID"; exit 0 ;;
  *) echo "usage: $0 [smoke|full|prefill|serve-only]"; exit 2 ;;
esac

python -m bench.runner \
  --matrix bench/config_matrix_phase7_v1.yaml \
  --backends vllm \
  --out "$OUT" \
  "${EXTRA[@]}"

# Persist versions
python - <<PY
import json, vllm, torch, pathlib
p = pathlib.Path("$RESULTS_DIR") / "run_metadata_vllm.json"
# runner also writes run_metadata.json next to CSV; keep an explicit copy
print("vllm", vllm.__version__, "torch", torch.__version__)
PY

echo "DONE vLLM -> $OUT"
