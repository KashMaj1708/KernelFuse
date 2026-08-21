#!/usr/bin/env bash
# Phase 7 — SGLang: separate venv → serve → harness → tear down.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RESULTS_DIR="${RESULTS_DIR:-/workspace/kernelfuse_results}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
REVISION="${REVISION:-a09a35458c702b33eeacc393d103063234e8bc28}"
DTYPE="${DTYPE:-bfloat16}"
PORT="${PORT:-30000}"
SGLANG_VERSION="${SGLANG_VERSION:-0.4.5.post1}"
VENV="${VENV:-$ROOT/.venv-sglang}"
MODE="${1:-full}"

mkdir -p "$RESULTS_DIR"
export KERNELFUSE_SGLANG_BASE_URL="http://127.0.0.1:${PORT}/v1"
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

echo "=== Phase 7 SGLang ($MODE) ==="
echo "model=$MODEL rev=$REVISION dtype=$DTYPE pin=sglang==$SGLANG_VERSION"

if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip -q install -U pip
python -m pip -q install "sglang[all]==${SGLANG_VERSION}" pyyaml huggingface_hub transformers || \
  python -m pip -q install "sglang==${SGLANG_VERSION}" pyyaml huggingface_hub transformers

LOG="$RESULTS_DIR/sglang_server.log"
# OpenAI-compatible launch (flag names evolve — adjust on instance if needed).
python -m sglang.launch_server \
  --model-path "$MODEL" \
  --revision "$REVISION" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --mem-fraction-static "${GPU_MEM_UTIL:-0.90}" \
  >"$LOG" 2>&1 &
SERVER_PID=$!
echo "server pid=$SERVER_PID log=$LOG"

cleanup() {
  echo "tearing down SGLang pid=$SERVER_PID"
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

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

ATTN="flash_attn"
OUT="$RESULTS_DIR/results_phase7_v1_sglang.csv"
EXTRA=(--attention-backend "$ATTN")
case "$MODE" in
  smoke)   EXTRA+=(--smoke); OUT="$RESULTS_DIR/results_phase7_v1_sglang_smoke.csv" ;;
  prefill) EXTRA+=(--prefill-heavy); OUT="$RESULTS_DIR/results_phase7_v1_sglang_prefill.csv" ;;
  full)    ;;
  *) echo "usage: $0 [smoke|full|prefill]"; exit 2 ;;
esac

python -m bench.runner \
  --matrix bench/config_matrix_phase7_v1.yaml \
  --backends sglang \
  --out "$OUT" \
  "${EXTRA[@]}"

echo "DONE SGLang -> $OUT"
