## Phase 8 — vLLM integration (active)

**Headline:** custom **add+RMSNorm** op in vLLM 0.8.5, not benchmark hygiene.

| Doc | Path |
|-----|------|
| Plan | [`docs/phase_8_plan.md`](phase_8_plan.md) |
| Predictions (null pre-registered) | [`docs/phase_8_predictions.md`](phase_8_predictions.md) |
| Matrix | [`bench/config_matrix_phase8_v1.yaml`](../bench/config_matrix_phase8_v1.yaml) |

| Milestone | Status |
|-----------|--------|
| **M1** add_rmsnorm @ bf16/3584 + vec8 | **done** — `add_rmsnorm_fused.exe`, 7/7 shapes vs vLLM native golden |
| **M2** torch extension | **Rented A100** — `scripts/phase8/session_build_op.sh` |
| **M3** vLLM 0.8.5 call-site swap | **Rented A100** (same session as M2) |
| **M4** CUDA graph capture smoke | **Rented A100** |
| **M5** preamble + A/B | **Same rental** — do not tear down between M2 and M5 |

Build/test: `.\\scripts\\run_add_rmsnorm_tests.ps1`

**Gates before A100:** G0 green locally. **One rental** for G1–G5 (extension → vLLM swap → graph → A/B).

Build on box: `bash scripts/phase8/session_build_op.sh`

**First engine: vLLM.** SGLang follow-up after gate.

## Phase 7

**Closed** (report gitignored). Harness fixes on main: `bench/cache_obs.py`, flush-between-cells default for pre-registered matrices.

## Phase 6

**Closed.** [`docs/phase_6_report.md`](phase_6_report.md)

## Docker / WSL

Engine + NVIDIA toolkit; ext4 loop on E:. GPU smoke passed.
