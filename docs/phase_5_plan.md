# Phase 5 — Serving benchmark harness (Tier A)

## Goal

Prove the full benchmarking harness end-to-end on the GTX 1650 against a **toy** model.
Logic only: config matrix → requests → latency percentiles → throughput. Not reportable numbers.

## Pre-registered config matrix

Written **before** any run (GordianBench-style). Do not widen mid-experiment without a new matrix version.

| Field | Values (v1) | Notes |
|-------|-------------|-------|
| `matrix_version` | `phase5-v1` | Bump if the sweep changes |
| `backend` | `mock`, `hf_local` | `vllm` / `sglang` / `trtllm` stubs only until Phase 6+ |
| `model` | `mock-toy`, `hf-internal-testing/tiny-random-gpt2` | Must fit in 4 GB; safetensors (torch 2.5.1 + modern transformers) |
| `batch_size` | 1, 2, 4 | |
| `input_tokens` | 32, 128 | Prompt length (approx) |
| `output_tokens` | 16, 64 | `max_new_tokens` |
| `concurrency` | 1, 2 | Concurrent in-flight requests |
| `num_requests` | 32 | Per cell (after warmup) |
| `warmup_requests` | 4 | Discarded |
| `metric` | p50 / p90 / p99 latency (ms), throughput (tok/s) | **Not** means |

Full machine-readable copy: [`bench/config_matrix_phase5_v1.yaml`](../bench/config_matrix_phase5_v1.yaml).

## Backend contract

Every backend implements:

- `start()` / `stop()` (optional for in-process)
- `generate(prompt, max_new_tokens) -> GenerateResult` with `latency_ms`, `output_tokens`
- `name`, `model_id`

Harness code must not branch on backend internals — only on this interface.

## Exit gate

Harness runs the full `phase5-v1` matrix on Tier A, prints a results table with p50/p90/p99 and throughput, and exits 0. Scale and real backends come in Phases 6–7.
