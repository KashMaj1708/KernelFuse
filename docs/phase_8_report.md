# Phase 8 report — vLLM add+RMSNorm integration (A100)

## Exit gate

**Met for serving integration + e2e null.** Custom fused **add+RMSNorm** op integrated into **vLLM 0.8.5** on Qwen2.5-7B @ A100 with CUDA graphs on. Batch-1 decode e2e TPOT unchanged within smoke variance.

**Kernel isolation:** stock vLLM is faster. After amortized timing + a rows sweep + a register-resident rewrite, that ranking is stable and explainable (below). Do **not** claim a kernelfuse isolation win.

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

- **Kernel:** `kernelfuse` bf16 / fp32 reduce / vec8 @ 3584 (register-resident post-add row; see below).
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

**Report all cells; label underpowered.** Short wall-time c=64/128 cells show larger run-to-run spread. They are **not** used for the null claim and are **not** deleted from the CSV.

## Kernel microbench — framing

### The 10³× napkin does not apply at these shapes

~10 ns came from dividing ~14 KB by **device-wide** HBM bandwidth. A `[1, 3584]` row is **one block on one SM** out of 108, with a dependent chain (load → reduce → normalize → store). The realistic floor is a few DRAM round-trips plus the reduction: **hundreds of nanoseconds to ~1 µs**. Measured **3.4 µs** (vLLM, amortized graph) is roughly **3–7×** above that floor — **latency-bound / occupancy-starved**, not bandwidth-bound.

Signature: rows=1 → 3.43 µs, rows=64 → 4.02 µs (64× work, ~17% more time). Same failure mode as Phase 9’s overhead-floored `torch_ref`.

### Amortized measurement

N launches inside one CUDA-graph capture (or one event bracket), replay/sync once, divide by N. Removes per-launch CPU overhead; what remains at small rows is **kernel latency + residual graph-node cost**, not a pure HBM bound.

### Rows sweep (graph, 200 launches amortized) — register-resident kernel

| rows | kf min (µs) | vLLM min (µs) | kf/vLLM | regime note |
|-----:|------------:|--------------:|--------:|-------------|
| 1 | 6.15 | 3.43 | **0.56×** | latency / dead zone |
| 8 | 6.27 | 3.55 | **0.57×** | latency |
| 64 | 7.67 | 4.02 | **0.53×** | still latency-ish |
| 512 | 25.5 | 6.74 | **0.27×** | transition |
| 4096 | 142 | 75.2 | **0.53×** | bandwidth-bound |
| 16384 | 547 | 284 | **0.52×** | bandwidth-bound |

Artifacts: `reports/phase8/kernel_rows_sweep_graph_v3.csv`.

**Serving map:** rows=1 ≈ batch-1 decode (null already confirmed). Large rows ≈ prefill / high-concurrency decode — the only regime where a norm kernel can matter. Ranking at the bandwidth end: **~2× slower than vLLM**, stable.

## Why kernelfuse loses (~2×)

Phase 2 already noted the fused RMSNorm still **read `x` twice from global** (reduce, then normalize). Production `fused_add_rms_norm` keeps the post-add row resident between passes.

First rewrite attempt staged `float local[][]` with **runtime** indexing → ptxas reported a **128-byte stack frame** (local memory = DRAM). That accidentally recreated two-pass traffic. Fix: `#pragma unroll` over a compile-time `kVecPerThread` so indices are constant after unroll.

**ptxas (sm_80) after fix:** vec8 kernel — **0 bytes stack, 0 spills, 32 registers**.

Bandwidth-bound ratio moved from ~0.39× (stack-backed “cache”) to **~0.52×** (true registers). Most of a clean 2× story remains: still slower than vLLM at the knee. Likely residual causes (instruction mix / reduce / vectorization) — **ncu DRAM counters blocked** on this rental (`RmProfilingAdminOnly=1` / `ERR_NVGPUCTRPERM`), so the counter-level confirmation is pending a machine where profiling is allowed.

Correctness vs vLLM at hidden=3584: residual exact; output max-abs ≤ 3.125e-2 (bf16).

## Why e2e did not move

~15 GB weights/token vs ~800 KB norm traffic / token across 28 layers. Batch-1 decode never leaves the latency dead zone for this op.

## Limits

- No definitive cold prefill (flush no-op; see Phase 7 errata).
- Greedy string match ≠ numerical tensor identity.
- ncu metrics not collected on this instance (driver profiling permission).
- Gap vs vLLM at BW-bound shapes closed only partially by register residency.
