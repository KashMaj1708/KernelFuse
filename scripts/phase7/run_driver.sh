#!/usr/bin/env bash
# Phase 7 top-level driver: vLLM → SGLang → TRT-LLM.
# Usage:
#   ./scripts/phase7/run_driver.sh --dry-run          # local mock, free
#   ./scripts/phase7/run_driver.sh --smoke            # one cell each backend on instance
#   ./scripts/phase7/run_driver.sh                    # full matrix each backend
#   ./scripts/phase7/run_driver.sh --prefill          # prefill-heavy only
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RESULTS_DIR="${RESULTS_DIR:-$ROOT/reports/phase7}"
mkdir -p "$RESULTS_DIR"
export RESULTS_DIR

DRY_RUN=0
MODE=full
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --smoke) MODE=smoke; shift ;;
    --prefill) MODE=prefill; shift ;;
    --full) MODE=full; shift ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

SESSION_LOG="$RESULTS_DIR/session_log.txt"
log() { echo "[$(date -Iseconds 2>/dev/null || date)] $*" | tee -a "$SESSION_LOG"; }

log "driver start mode=$MODE dry_run=$DRY_RUN results=$RESULTS_DIR"

if [[ "$DRY_RUN" -eq 1 ]]; then
  log "DRY-RUN: mock backend, limit-cells, incremental CSV path check"
  OUT="$RESULTS_DIR/dry_run_mock.csv"
  rm -f "$OUT"
  python -m bench.runner \
    --matrix bench/config_matrix_phase7_v1.yaml \
    --backends mock \
    --smoke \
    --limit-cells 3 \
    --out "$OUT" \
    --attention-backend mock
  test -s "$OUT"
  test -s "$(dirname "$OUT")/run_metadata.json" || test -f "$RESULTS_DIR/run_metadata.json" || \
    test -f "$(dirname "$OUT")/run_metadata.json"
  # metadata written next to CSV
  test -f "$RESULTS_DIR/run_metadata.json" || test -f "$(dirname "$OUT")/run_metadata.json"
  log "DRY-RUN OK rows=$(wc -l < "$OUT") file=$OUT"
  log "Also verify tokenizer import path (no download for mock):"
  python -c "from bench.prompts import make_prompt; print(make_prompt(8,0,0,model_id='mock-toy'))"
  exit 0
fi

# Live order: bank results early
log "=== 1/3 vLLM ==="
bash "$ROOT/scripts/phase7/run_vllm.sh" "$MODE"
log "download reminder: scp $RESULTS_DIR/results_phase7_v1_vllm*.csv locally now"

log "=== 2/3 SGLang ==="
bash "$ROOT/scripts/phase7/run_sglang.sh" "$MODE"
log "download reminder: scp sglang CSVs locally now"

log "=== 3/3 TRT-LLM (last; may need engine build) ==="
bash "$ROOT/scripts/phase7/run_trtllm.sh" "$MODE" || log "TRT-LLM failed — vLLM/SGLang results still on disk"

log "driver complete. Before terminate: verify CSVs readable + metadata + TRT engine on persistent volume."
