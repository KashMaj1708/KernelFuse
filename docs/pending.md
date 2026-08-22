## Phase 8 — vLLM integration

**Closed.** Report: `docs/phase_8_report.md` (local). Git: kernelfuse op + scripts on main.

## Phase 8b — SGLang integration

**Closed on A100.** Null confirmed (TPOT ~9.9→10.0 ms). Build extension **per-venv** (SGLang torch 2.13 ≠ vLLM torch 2.6).

| Doc / script | Path |
|--------------|------|
| Plan | `docs/phase_8b_plan.md` |
| Matrix | `bench/config_matrix_phase8b_v1.yaml` |
| Treatment | `scripts/phase8b/session_sglang_treatment.sh` |

## Phase 9 — decode attention (ceiling)

**Closed as optional stretch.** Naive CUDA microbench on A100; slower than torch_ref — no serving claim.

| Doc / script | Path |
|--------------|------|
| Plan | `docs/phase_9_plan.md` |
| Kernel | `kernels/attention/decode_attn_bench.cu` |
| Bench | `scripts/phase9/session_decode_attn_bench.sh` |

## Phase 7

**Closed.** Harness cache observability on main.
