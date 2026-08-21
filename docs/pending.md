## Phase 7 pre-flight

**Part A (local) — ready to rent when pushed.**

- Matrix: [`bench/config_matrix_phase7_v1.yaml`](../bench/config_matrix_phase7_v1.yaml) — Qwen2.5-7B @ `a09a354…`, c→64, prefill cells, warmup 8 / n 64
- Predictions: [`docs/phase_7_predictions.md`](phase_7_predictions.md)
- Scripts: [`scripts/phase7/run_driver.sh`](../scripts/phase7/run_driver.sh) (`--dry-run` / `--smoke` / full)
- Harness: tokenizer prompts + incremental CSV after each cell
- Local mock dry-run: passed

**On instance:** Parts B–E of the operator checklist; `RESULTS_DIR` on persistent volume; TRT engine never left on ephemeral disk.

## Phase 6

**Closed.** [`docs/phase_6_report.md`](phase_6_report.md)

## Docker / WSL

Engine + NVIDIA toolkit; ext4 loop on E:. GPU smoke passed.
