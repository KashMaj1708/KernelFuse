# Phase 8 report — vLLM add+RMSNorm integration (A100)

## Exit gate

**Met for serving integration + e2e null.** Custom fused **add+RMSNorm** op integrated into **vLLM 0.8.5** on Qwen2.5-7B @ A100 with CUDA graphs on. Batch-1 decode e2e TPOT unchanged within smoke variance.

**Kernel isolation (after register + vectorization fixes):** at bandwidth-bound shapes, kernelfuse reaches **parity** with stock vLLM (~1.66 TB/s under a four-array model). Earlier ~0.5× rankings were measurement + codegen artifacts, not an irreducible algorithm gap.

## Provenance

| Field | Value |
|-------|--------|
| Hardware | Vast.ai 1× A100-SXM4-80GB (sm_80) |
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| torch | 2.6.0+cu124 |
| vLLM | 0.8.5 |
| dtype | bfloat16 |
| graphs | on |

## Kernel microbench — framing

### The 10³× napkin does not apply at these shapes

~10 ns came from dividing ~14 KB by **device-wide** HBM. A `[1, 3584]` row is **one block on one SM**, with a dependent chain (load → reduce → normalize → store). Realistic floor: hundreds of ns to ~1 µs. Measured ~3.4 µs (vLLM, amortized) is **latency-bound / occupancy-starved**, not bandwidth-bound.

### Amortized measurement

N launches inside one CUDA-graph capture, replay once, divide by N. Removes per-launch CPU overhead.

### Rows sweep (graph, amortized) — after register + `uint4` load fix

| rows | kf min (µs) | vLLM min (µs) | kf/vLLM | regime |
|-----:|------------:|--------------:|--------:|--------|
| 1 | 3.70 | 3.42 | 0.93× | latency |
| 8 | 3.79 | 3.54 | 0.93× | latency |
| 64 | 4.10 | 4.02 | 0.98× | latency-ish |
| 512 | 6.56 | 8.19 | **1.25×** | transition |
| 4096 | 74.1 | 75.3 | **1.02×** | bandwidth |
| 16384 | 282.7 | 284.0 | **1.00×** | bandwidth |

Artifact: `reports/phase8/kernel_rows_sweep_graph_v4.csv` (remote: `kernelfuse_results/phase8/`).

## Why kernelfuse lost — then caught up

### 1. Spill / local-memory demotion (the war story)

Registers are **not addressable**. A local array indexed by a **runtime** variable cannot live in registers; nvcc silently demotes it to **local memory**, which is backed by global DRAM. The first “register-resident” rewrite still did a second global read — just under another name. That is why ptxas reported a **128-byte stack frame**.

The fix is not `__restrict__` or loop shape: **compile-time indices** (`#pragma unroll` over `kVecPerThread`) so every access is a constant after unroll. After that: **0 stack, 0 spills, ~32 registers**. Ratio moved ~0.39× → ~0.52×.

### 2. Vectorization that never survived codegen (SASS, no ncu)

`cuobjdump -sass` on the kernelfuse `.so` (pre-fix) showed **`LDG.E.U16` / `STG.E.U16`** — the intended vec8 16-byte transactions never made it into SASS. Loading a `struct { bf16[8] }` does not guarantee `LDG.E.128`.

Fix: explicit `uint4` load/store helpers. Post-fix SASS:

| binary | dominant global ops |
|--------|---------------------|
| kernelfuse vec8 | **`LDG.E.128` / `STG.E.128`** |
| vLLM `fused_add_rms_norm_kernel<BFloat16>` (one-symbol dump) | `LDG.E.U16` / `STG.E.U16` on this instantiation |

**Do not** `cuobjdump -sass` the whole vLLM `_C.so` — it thrashs. Resolve the `.so` with logging disabled, `cuobjdump -symbols | grep fused_add_rms_norm`, then `cuobjdump -fun '<mangled>' -sass`.

### 3. Four-array bandwidth arithmetic (no ncu)

At rows=16384, hidden=3584, bf16, a correct fused add+RMSNorm moves four arrays (read input, read residual, write residual, write output):

`4 × 16384 × 3584 × 2 ≈ 470 MB`

| kernel | time | effective BW (4-array) |
|--------|-----:|-----------------------:|
| kernelfuse | 282.7 µs | **1.66 TB/s** |
| vLLM | 284.0 µs | **1.65 TB/s** |

Both land near achievable A100 peak for this traffic. Residual is written **in place** (same buffer), not a fifth array.

## Why e2e did not move

~15 GB weights/token vs ~800 KB norm traffic / token. Batch-1 decode never leaves the latency dead zone for this op.

## Limits

- No definitive cold prefill (flush no-op; Phase 7 errata).
- ncu counters still blocked on this rental (`NVreg_RestrictProfilingToAdminUsers`); static SASS + byte arithmetic were enough here.
- vLLM one-symbol SASS shows U16 on the dumped instantiation; runtime parity at 1.66 TB/s is the stronger claim.
