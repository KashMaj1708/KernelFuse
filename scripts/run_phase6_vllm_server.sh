#!/usr/bin/env bash
# Launch vLLM OpenAI server for Phase 6 dry run (Colab T4 / Linux).
# Explicit float16 — T4 has no bf16; TinyLlama config says bf16.
# enforce-eager skips CUDA-graph capture memory cost.
set -euo pipefail

MODEL="${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-float16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.75}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
HOST="${HOST:-0.0.0.0}"

# Optional HF token for gated/rate-limited downloads (never commit the token).
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

echo "=== vLLM serve ==="
echo "model=${MODEL} dtype=${DTYPE} gpu_memory_utilization=${GPU_MEM_UTIL} enforce_eager=1"
echo "expect attention backend: XFormers on sm_75 (no FlashAttention)"

exec python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype "${DTYPE}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --enforce-eager \
  --no-enable-log-requests
