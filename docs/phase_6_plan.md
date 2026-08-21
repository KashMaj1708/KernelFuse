# Phase 6 — Single-backend dry run (vLLM on free cloud)

## Goal

Prove a **real** serving backend + real small model against the Phase 5+ harness before paid Tier C time.

**Exit gate:** vLLM serves TinyLlama; harness emits coherent TTFT / TPOT / dual throughput on free hardware (prefer Colab **T4**), with `run_metadata.json` recording versions and launch flags.

## Pre-registered matrix

[`bench/config_matrix_phase6_v1.yaml`](../bench/config_matrix_phase6_v1.yaml) — do not edit after first measurement.

| Field | Values |
|-------|--------|
| `matrix_version` | `phase6-v1` |
| `backend` | `vllm` |
| `model` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| full sweep | bs 1/4/8; in 128/512/1024; out 128/256/512; c 1/2/4/8/16 |
| `colab_smoke` | bs=1; in 128/512; out 128; c 1/2/4 |

## T4 launch contract (mandatory)

| Flag | Value | Why |
|------|-------|-----|
| `--dtype` | **`float16`** | T4 (sm_75) has no bf16; TinyLlama config says bf16 — make the override explicit |
| `--enforce-eager` | on | Skip CUDA-graph capture memory on a dry run |
| `--gpu-memory-utilization` | **0.75** (tune down from 0.9) | Headroom for KV + concurrent requests |
| attention | expect **XFormers** | FlashAttention needs sm_80+; Phase 6 ≠ Phase 7 path |

**Colab install:** put `pip install vllm` in its **own** cell, then **Runtime → Restart session**, then import/serve. vLLM pins torch; Colab’s preinstall conflicts until restart.

## Harness path

```bash
# server
export HF_TOKEN=...   # optional; Colab Secrets preferred — never commit
bash scripts/run_phase6_vllm_server.sh

# client
export KERNELFUSE_VLLM_BASE_URL=http://127.0.0.1:8000/v1
python -m bench.runner \
  --matrix bench/config_matrix_phase6_v1.yaml \
  --backends vllm --smoke \
  --out reports/phase6/results_phase6_v1_colab_smoke.csv \
  --attention-backend xformers
```

Notebook: [`notebooks/phase6_colab.ipynb`](../notebooks/phase6_colab.ipynb).

## Provenance

Every result set must ship `run_metadata.json` next to the CSV with at least: `vllm_version`, `torch_version`, `dtype`, `attention_backend`, `gpu_name` / `compute_capability`, `enforce_eager`, `gpu_memory_utilization`, `model_id`, `matrix_version`. Same role as recording torch version in Phase 3 — Phase 7 will differ on all three of version / attention / dtype class.

## Framing

Concurrency is a **real** scheduling axis here (continuous batching). System throughput vs per-request rate is the comparison of interest. Absolute numbers are **not** comparable to Phase 7 Ampere+ FlashAttention runs.
