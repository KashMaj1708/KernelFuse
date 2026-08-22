# Phase 8 — vLLM integration (custom add+RMSNorm op)

## Goal

Close the Layer 1 / Layer 2 gap: **benchmark next to a kernel** → **kernel inside a production serving stack**.

Phase 7 is closed with honest limits. Phase 8 is **not** a benchmark remount; leftover validation is a **preamble** on the same A100 session as the integration A/B.

---

## Explicit scope

| In scope | Out of scope (defer) |
|----------|----------------------|
| `add_rmsnorm` CUDA kernel (bf16, fp32 acc, vec8) | SGLang swap (follow-up after vLLM gate) |
| torch custom op + vLLM 0.8.5 call-site swap | Attention fusion |
| Correctness + graph capture + isolated kernel bench | Headline “30% faster vLLM” claim |
| A100 A/B with **pre-registered null** | Full 156-cell matrix rerun |
| Preamble: cache pair, output equiv, variance repeat | Phase 7-scale three-backend sweep |

**First engine: vLLM.** SGLang won decode ranking in Phase 7; vLLM wins **integration friction** for Phase 8.

---

## Workstreams

### WS1 — Kernel (`kernels/rmsnorm/`)

1. **`add_rmsnorm_fused.cu`** — residual add + RMSNorm, in-place residual, bf16 IO, fp32 reduce.
2. **`add_rmsnorm_fused_vec8.cu`** — 8-wide loads; hidden 3584 → 448 vec8 iter/layer-row.
3. Re-probe occupancy/smem on **3584 cols** (1650 for compile; A100 for truth).
4. Extend correctness harness: compare against `x + F.rms_norm(x, w)` with residual write-back semantics matching vLLM.

### WS2 — PyTorch extension

1. `torch.utils.cpp_extension.CUDAExtension` → `kernelfuse.add_rmsnorm(...)`.
2. No alloc/sync in forward; shapes static for graph replay.
3. Unit tests on 1650: `[1,3584]`, `[8,3584]`, `[32,3584]`.

### WS3 — vLLM integration

1. Locate Qwen2 decoder `RMSNorm` + residual call site in vLLM 0.8.5 model code.
2. Register custom op (CustomOp / platform plugin path for 0.8.x).
3. Swap **fused add+RMSNorm** at block level — do not replace norm module alone.
4. TinyLlama or Qwen-0.5B on Colab T4: server starts, one generate, graph capture log check.

### WS4 — Measurement (A100, once)

See [`docs/phase_8_predictions.md`](phase_8_predictions.md) — preamble block then baseline vs treatment A/B.

---

## Harness (already on main)

Phase 7 confound → permanent harness fixes:

- [`bench/cache_obs.py`](../bench/cache_obs.py) — prefix hit rate from log/metrics
- `prompt_tokens`, `ttft_min/max/span`, `prefix_cache_hit_pct` in CSV
- **`flush_cache_between_cells` default on** for pre-registered matrices (opt out per run)

[`scripts/phase7/grep_cache_hits.py`](../scripts/phase7/grep_cache_hits.py) — offline log grep, no GPU.

---

## Traps (read before coding)

1. **Norm-only swap** → false regression (two kernels instead of one).
2. **fp32 / vec4 / 48 KiB assumptions** → wrong kernel for bf16 3584 on A100.
3. **Graph capture violations** → silent fallback; A/B invalid.
4. **Expecting e2e TPOT to move** → null violation; see predictions doc.

---

## Milestones

| # | Milestone | Tier |
|---|-----------|------|
| M1 | add_rmsnorm correct vs golden @ 3584 bf16 | A (1650) |
| M2 | torch op + extension builds | A |
| M3 | vLLM serves TinyLlama/Qwen with op | B (T4) |
| M4 | CUDA graph capture green | B |
| M5 | A100 preamble + A/B recorded | C |
| M6 | Phase 8 report + README integration section | — |

**Rent A100 only for M5.**

---

## Exit gate

- Custom op live in vLLM on Qwen2.5-7B @ A100, graphs on.
- Null confirmed (e2e unchanged; kernel isolation shows gain).
- Phase 8 report explains bandwidth-bound decode (why norm optimization doesn't move TPOT).
- SGLang path documented as next step, not blocked scope creep.

---

## Artifacts (planned)

| Path | Role |
|------|------|
| `docs/phase_8_predictions.md` | pre-registered null + A100 block |
| `docs/phase_8_report.md` | results (gitignored until written) |
| `bench/config_matrix_phase8_v1.yaml` | minimal A/B + preamble cells |
| `kernels/rmsnorm/add_rmsnorm_*.cu` | production kernel |
| `kernelfuse/` or `kernels/torch_ext/` | extension package |
