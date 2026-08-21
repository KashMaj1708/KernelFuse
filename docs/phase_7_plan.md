# Phase 7 — Full multi-backend benchmark (paid)

## Goal

Reportable vLLM / SGLang / TensorRT-LLM comparison on Tier C after Phase 6 proved the harness.

## Pre-flight status (Part A — before renting)

See also the operator checklist you wrote; repo status:

| Item | Status |
|------|--------|
| Tokenizer prompts (`bench/prompts.py`) | done |
| `--smoke` / `--prefill-heavy` / incremental CSV | done |
| `phase7-v1` matrix pinned (Qwen2.5-7B + revision) | done |
| Predictions (decode floor / KV / budget) | [`docs/phase_7_predictions.md`](phase_7_predictions.md) |
| Session scripts | [`scripts/phase7/`](../scripts/phase7/) |
| Local `--dry-run` | `bash scripts/phase7/run_driver.sh --dry-run` |

### Model

- Repo: `Qwen/Qwen2.5-7B-Instruct`
- Revision: `a09a35458c702b33eeacc393d103063234e8bc28`
- Ungated; RMSNorm architecture (Layer 1 link)

### Matrix (`bench/config_matrix_phase7_v1.yaml`)

- Main: in {128,512,2048} × out {128,512} × c {1,2,4,8,16,32,64} × bs=1 → **42 cells/backend**
- Prefill: in=2048, out=32, c {1,4,16,32} → **4 cells/backend**
- Warmup/measure: **8 / 64**
- Launch: `dtype=bfloat16`, `enforce_eager=false`, `gpu_memory_utilization=0.90`

### Isolation

- `.venv-vllm` / `.venv-sglang` separate pins
- TRT-LLM: NGC container; engine on **persistent** `RESULTS_DIR`
- Harness is HTTP-only → process boundary already true

### Driver order

1. vLLM (validated path) → download CSV  
2. SGLang → download CSV  
3. TRT-LLM last (engine build risk)

## On-instance (Parts B–E)

Follow operator checklist: confirm A100-SXM4-40GB, CUDA 12.x, disk, idle GPU, bandwidth; per-backend smoke identity check; clocks during sweep; download after each backend; save TRT engine before terminate.

## Exit gate

Full pre-registered sweep (or justified subset after knee found) across backends with complete `run_metadata.json` each.
