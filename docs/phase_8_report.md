# Phase 8 report — vLLM add+RMSNorm integration (A100)

## Exit gate

**Met for serving integration + e2e null.** Custom fused **add+RMSNorm** op integrated into **vLLM 0.8.5** on Qwen2.5-7B @ A100 with CUDA graphs on. Batch-1 decode e2e TPOT unchanged within smoke variance.

**Not met for kernel-isolation speedup.** Wall-clock microbenches at `[1,3584]` / `[32,3584]` timed launch overhead, not kernel work. Correct claim: **kernel-level difference is below the resolution of the recorded microbench** (see below). Do not read “modest speedup” from those rows.

## Provenance

| Field | Value |
|-------|--------|
| Hardware | Vast.ai 1× A100-SXM4-80GB (sm_80) |
| Model | `Qwen/Qwen2.5-7B-Instruct` @ `a09a35458c702b33eeacc393d103063234e8bc28` |
| torch | 2.6.0+cu124 |
| vLLM | 0.8.5 |
| Attention | FLASH_ATTN |
| dtype | bfloat16 |
| graphs | on (`enforce_eager=false`) |
| Matrix | `phase8-v1` |

## Integration

- **Kernel:** `kernelfuse` bf16 / fp32 reduce / vec8 @ 3584.
- **Hook:** `KERNELFUSE_FUSED_ADD_RMSNORM=1` patch on `layernorm.py`.
- **Graph capture:** finished ~23s baseline and treatment.

## Preamble

| Check | Result |
|-------|--------|
| Output smoke (greedy text, seed=0) | **PASS** — **generated strings** match baseline vs treatment. This is **not** tensor bitwise equivalence; argmax is robust to small logit noise. Tensor max-abs/rel vs reference remains a separate unit test on the op. |
| Variance repeat (smoke ×3) | TPOT p50: **11.1, 11.1, 11.2 ms**; sys **88.7–88.9 tok/s** |
| Cache flush arm `in=2048` | TTFT p50 **129.5 ms**, hit **25.6%**, flush **`ok=False`** |
| Cache warm arm | TTFT p50 **20.8 ms**, hit **58.0%** |

**Cold overturn (propagates to Phase 7):** the flush arm was **not cold**. See [`docs/phase_7_errata.md`](phase_7_errata.md). Implied true-cold TTFT ≈ 129.5/0.744 ≈ **174 ms** (~52% MFU). Phase 7’s 108.8 ms is unknown-cache-state, not a cold anchor.

## A/B — e2e null (batch-1 decode)

| Cell | Baseline TPOT | Treatment TPOT | Δ |
|------|--------------:|---------------:|--:|
| in=128 out=128 c=1 | 11.28 ms | 11.10 ms | ~1.6% |
| in=128 out=512 c=1 | 11.25 ms | 11.21 ms | ~0.4% |

| Cell | Baseline sys | Treatment sys |
|------|-------------:|--------------:|
| in=128 out=128 c=1 | 87.7 | 89.3 |
| in=128 out=512 c=1 | 88.7 | 88.6 |

**Null holds at serving level** (bandwidth argument unchanged).

### Concurrency policy (not post-hoc exclusion)

**Report all cells; label underpowered.** Short wall-time c=64/128 cells show larger run-to-run spread (including duplicate rows from an interrupted A/B). They are **not** used for the null claim and are **not** deleted from the CSV. Same policy as Phase 8b (which reports c=64 as secondary data).

| Cell | Baseline sys | Treatment note |
|------|-------------:|----------------|
| in=128 out=128 c=64 | 3753 | underpowered / short wall |
| in=128 out=128 c=128 | 3750 | underpowered / short wall |
| in=128 out=512 c=64 | 3716 | treatment CSV mixed with interrupted pass — interpret with spread |
| in=128 out=512 c=128 | 3723 | same |

## Kernel microbench (retracted interpretation)

Recorded wall-clock numbers (~16–32 µs @ rows=1) are **~10³×** above the HBM napkin (~10 ns for ~14 KB). They measure **launch + sync**, not the kernel. Spread 0.75×–1.29× is harness noise, not kernel variance.

**Correct claim:** kernel-level difference **unresolved** by the Phase 8 microbench. Methodology fix: CUDA events, min-of-trials, graph replay (`scripts/phase8/kernel_microbench.py`); cross-check `ncu --metrics gpu__time_duration.sum`.

## Why e2e did not move

~15 GB weights/token vs ~800 KB norm traffic / token across 28 layers.

## Limits

- No definitive cold prefill (flush no-op; see Phase 7 errata).
- Greedy string match ≠ numerical tensor identity.
- Wall-clock microbench does not support isolation speedup claims.
