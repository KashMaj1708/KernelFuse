## Phase 8 — vLLM integration (active)

**Headline:** custom **add+RMSNorm** op in vLLM 0.8.5, not benchmark hygiene.

| Doc | Path |
|-----|------|
| Plan | [`docs/phase_8_plan.md`](phase_8_plan.md) |
| Predictions (null pre-registered) | [`docs/phase_8_predictions.md`](phase_8_predictions.md) |
| Matrix | [`bench/config_matrix_phase8_v1.yaml`](../bench/config_matrix_phase8_v1.yaml) |

**Gates before A100:** add_rmsnorm kernel @ bf16/3584 → torch op → vLLM swap on T4 → CUDA graph capture green.

**A100 session (one rental):** preamble (cache pair, output equiv, variance repeat) + baseline vs op A/B. See predictions doc.

**First engine: vLLM.** SGLang follow-up after gate.

## Phase 7

**Closed** (report gitignored). Harness fixes on main: `bench/cache_obs.py`, flush-between-cells default for pre-registered matrices.

## Phase 6

**Closed.** [`docs/phase_6_report.md`](phase_6_report.md)

## Docker / WSL

Engine + NVIDIA toolkit; ext4 loop on E:. GPU smoke passed.
