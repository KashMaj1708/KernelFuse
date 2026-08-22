# Phase 8b — SGLang integration

Same kernelfuse `fused_add_rms_norm` op, patched into SGLang 0.5.18 `layernorm.py` (wraps `sgl_kernel.fused_add_rmsnorm`).

Pre-registered null: e2e TPOT unchanged at batch-1 decode (same bandwidth argument as Phase 8).

Matrix: `bench/config_matrix_phase8b_v1.yaml`

Scripts: `scripts/phase8b/session_sglang_ab.sh`
