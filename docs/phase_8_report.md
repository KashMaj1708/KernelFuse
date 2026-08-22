# Phase 8 report — vLLM add+RMSNorm integration (A100)

## Exit gate

**Met for serving integration + e2e null.** Custom fused **add+RMSNorm** op integrated into **vLLM 0.8.5** on Qwen2.5-7B @ A100 with CUDA graphs on. Batch-1 decode e2e TPOT unchanged within smoke variance.

**Kernel isolation:** three regimes, not one number.

| Regime | rows | Result |
|--------|-----:|--------|
| Latency | 1–64 | Both starved; within ~2–8% |
| **Chunked-prefill / MLP-bound** | **128–1024** | **kernelfuse ahead; peak ~1.22× at 512–1024** |
| DRAM roof | ≥4096 | Parity (~1.66 TB/s both) |

## Provenance

| Field | Value |
|-------|--------|
| Hardware | Vast.ai 1× A100-SXM4-80GB (sm_80) |
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| torch | 2.6.0+cu124 |
| vLLM | 0.8.5 |
| dtype | bfloat16 |
| graphs | on |

## Fixes that matter

### 1. Register demotion (runtime-indexed locals)

Registers are not addressable. A local array indexed by a **runtime** variable cannot live in registers; nvcc silently demotes it to **local memory** (= DRAM). The first “register-resident” rewrite still did a second global read under another name (ptxas: **128-byte stack frame**).

Requirement for residency: **compile-time indices** after unroll (`kVecPerThread`). After fix: **0 stack, 0 spills, ~32 regs**. Ratio moved ~0.39× → ~0.52× before vectorization.

### 2. Vector width that survives codegen

A `struct { bf16[8] }` load does **not** guarantee 16-byte DRAM transactions. Pre-fix SASS: **`LDG.E.U16`**. Fix: explicit `uint4` load/store → **`LDG.E.128` / `STG.E.128`**.

Coalescing ≠ vectorization. Contiguous U16 loads across a warp still fill 128-byte sectors, so U16 can hit the DRAM roof. What `.128` buys is **per-thread MLP**: one instruction, 16 bytes in flight, instead of eight × 2-byte ops. That only shows when occupancy is too low to hide latency by itself.

## Three-regime story

### Full sweep (graph, amortized)

| rows | kf (µs) | vLLM (µs) | kf/vLLM |
|-----:|--------:|----------:|--------:|
| 1 | 3.70 | 3.42 | 0.93× |
| 8 | 3.79 | 3.54 | 0.93× |
| 64 | 4.10 | 4.02 | 0.98× |
| **128** | **4.64** | **5.03** | **1.09×** |
| **256** | **5.35** | **5.80** | **1.08×** |
| **512** | **6.70** | **8.12** | **1.21×** |
| **1024** | **11.79** | **14.40** | **1.22×** |
| **2048** | **28.42** | **26.56** | **0.94×** |
| 4096 | 74.1 | 75.3 | 1.02× |
| 16384 | 282.7 | 284.0 | 1.00× |

Artifacts: `kernel_rows_sweep_graph_v4.csv`, `kernel_rows_sweep_mid.csv`.

**512 is not a lone spike.** Advantage forms a smooth hump from 128→1024 (peak ~22%), then returns to parity / slight lag by 2048 as occupancy saturates the memory system. Mechanism matches the MLP story: ~5 waves over 108 SMs at rows≈512 — not enough blocks to hide latency by occupancy, enough work that the inner loop matters.

**Serving map:** rows ≈ batch × chunk length. vLLM chunked prefill defaults sit in **512–2048**, so the win region is the shape a real prefill hands the norm kernel — not a synthetic microbench quirk.

### Roof arithmetic (rows=16384, hidden=3584, bf16)

Four arrays (read in, read residual, write residual in-place, write out):

`4 × 16384 × 3584 × 2 ≈ 470 MB` → both kernels **~1.66 TB/s**. Parity at the roof is expected once both vectorize and the device is saturated.

## vLLM U16 anomaly — resolved

Earlier one-symbol dump hit `fused_add_rms_norm_kernel<BFloat16, **0**>` (`Li0` in the mangled name) — the **scalar fallback**. Runtime dispatch selects the vectorized specialization when hidden % 8 == 0 and pointers are 16-byte aligned (true for hidden=3584).

| Instantiation | Mangled tag | Dominant DRAM ops |
|---------------|-------------|-------------------|
| scalar fallback | `…BFloat16ELi0E…` | `LDG.E.U16` / `STG.E.U16` |
| **vectorized (what ran)** | `…BFloat16ELi8E…` | **`LDG.E.128` / `STG.E.128`** |

So both kernels vectorize at this shape. Roof parity is consistent. The mid-regime edge is **not** “we vectorize and they don’t”; it is residual inner-loop / MLP difference under partial occupancy.

## Why e2e batch-1 did not move

~15 GB weights/token vs ~800 KB norm traffic/token. Batch-1 decode lives in the latency dead zone (rows≈1).

## Limits

- No definitive cold prefill (flush no-op; Phase 7 errata).
- ncu counters blocked on this rental; SASS + byte arithmetic sufficient here.
- Mid-regime win is kernel isolation only; e2e chunked-prefill A/B not re-run in this phase.
