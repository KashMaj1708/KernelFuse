# Phase 8b — SGLang integration

Same kernelfuse `fused_add_rms_norm` op, patched into SGLang 0.5.18 `layernorm.py`
(wraps `sgl_kernel.fused_add_rmsnorm`).

## Pre-registered null (state before the run)

**Not** “parity vs vLLM’s op ⇒ parity vs SGLang.” `sgl_kernel.fused_add_rmsnorm`
is a different baseline (FlashInfer lineage). The prior ~1.4% sys-throughput
regression was measured with the **pre-fix** (two-pass / U16) kernelfuse binary,
so this rerun is a **fresh** measurement, not a re-test of that number.

| Claim | Decision rule (locked before run) |
|-------|-----------------------------------|
| **Parity** | Treatment sys tok/s within **±1×** the ×3 variance noise floor of baseline on the same cells (batch-1 decode), **and** greedy string match on the output-equiv arm. |
| **Regression** | Treatment sys tok/s worse than baseline by **>1×** that noise floor on ≥2 of 4 matrix cells, **or** greedy output mismatch. |
| **Kernel isolation** | CUDA-event microbench on the **SGLang-linked** `_C.so` (same ABI / venv as treatment). Report kf vs `sgl_kernel.fused_add_rmsnorm`, not vs vLLM. Mid-regime rows (128–2048) included. |

## Must carry forward (gaps the closed 8b report named)

1. **×3 variance repeat** on SGLang baseline smoke — real SGLang noise floor, not an imported vLLM one.
2. **Greedy output-equivalence** arm (string match, same framing as Phase 8).
3. **CUDA-event kernel isolation** against `sgl_kernel.fused_add_rmsnorm` on the SGLang-built extension.

## Matrix / scripts

- Matrix: `bench/config_matrix_phase8b_v1.yaml`
- Scripts: `scripts/phase8b/session_sglang_ab.sh`
