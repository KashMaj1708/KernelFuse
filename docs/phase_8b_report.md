# Phase 8b report — SGLang add+RMSNorm integration (A100)

## Exit gate

**Met for integration.** Same `kernelfuse` op patched into **SGLang 0.5.18**; server served Qwen2.5-7B with treatment path live.

**Not a null.** Treatment shows a **consistent ~1.4% throughput regression** across all four cells (directionally one-sided). Magnitude near a noise floor we **did not re-measure on SGLang**; underpowered at n=1 per arm.

## Provenance

| Field | Value |
|-------|--------|
| Hardware | Vast.ai 1× A100-SXM4-80GB |
| SGLang | 0.5.18 / `.venv-sglang` / torch **2.13.0+cu130** |
| Extension build | Against SGLang torch ABI; host toolkit **12.8** with **`_check_cuda_version` bypassed** |
| Matrix | `phase8b-v1` |

**Build caveat applies to every treatment number in this phase:** mismatched host CUDA vs torch CUDA, version check skipped.

## A/B — consistent small regression

| Cell | Baseline sys tok/s | Treatment sys tok/s | Δ |
|------|-------------------:|--------------------:|--:|
| in=128 out=128 c=1 | 99.7 | 98.3 | **−1.4%** |
| in=128 out=512 c=1 | 100.7 | 99.3 | **−1.4%** |
| in=128 out=128 c=64 | 3867 | 3803 | **−1.7%** |
| in=128 out=512 c=64 | 4591 | 4525 | **−1.4%** |

| Cell | Baseline TPOT | Treatment TPOT |
|------|--------------:|---------------:|
| in=128 out=128 c=1 | 9.9 ms | 10.0 ms |
| in=128 out=512 c=1 | 9.9 ms | 10.0 ms |

Four of four cells slower (sign-test p = 1/16 = 0.0625) — not conclusive, but **not** “noise scatter.” Importing a ~1% vLLM variance envelope onto SGLang was the wrong reading.

**Honest summary:** consistent ~1.4% sys-throughput regression; near noise; n=1; plausible mechanism = FlashInfer/`sgl_kernel` fused_add_rmsnorm more tuned than our vec8@3584 op **and/or** CUDA 13 codegen from the rebuilt extension.

## Gaps vs Phase 8

- **No SGLang variance repeat** (×3 smoke).
- **No greedy output-equivalence** arm on 8b.
- **No CUDA-event kernel isolation** on the SGLang-linked `_C.so`.

## Concurrency policy

Report **all** cells (including c=64) with the regression table above. Batch-1 remains the primary decode story; c=64 is secondary / still short-wall.

## Limits

- Treatment binary: bypassed CUDA version check vs host 12.8 / torch 13.0.
- Do not claim null; do not claim kernel win.
- Flush/cold issues from Phase 8 preamble are unchanged for SGLang radix flush.
