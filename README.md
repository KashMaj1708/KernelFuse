# KernelFuse

Two-layer inference-performance project:

1. **Serving benchmark** — compare vLLM / SGLang / TensorRT-LLM on throughput and latency percentiles.
2. **Hand-written fused CUDA kernel** — RMSNorm (naive → fused), correctness-gated, then profiled to show why fusion helps on a memory-bound path.

Develop on a GTX 1650 (or free cloud) first; rent a 24 GB-class card only for final measurement. See [`KernelFuse_build_plan.md`](KernelFuse_build_plan.md) for the phased plan and [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) for machine/toolchain notes.

Local write-ups (gitignored): [`docs/phase_0_report.md`](docs/phase_0_report.md), [`docs/phase_1_report.md`](docs/phase_1_report.md), [`docs/phase_2_report.md`](docs/phase_2_report.md), [`docs/phase_3_report.md`](docs/phase_3_report.md), [`docs/phase_4_report.md`](docs/phase_4_report.md), [`docs/fix_report_1.md`](docs/fix_report_1.md).

## Status

- **Phase 0** — environment setup and sanity check (complete)
- **Phase 1** — naive CUDA RMSNorm matches CPU golden reference (complete)
- **Phase 2** — fused RMSNorm (single launch) matches same harness (complete)
- **fixes_1** — pre-Phase-3 baseline: four variants, extended harness (complete; local [`docs/fix_report_1.md`](docs/fix_report_1.md))
- **Pre-Phase-3 blockers** — `.venv-cuda` (cu121), dynamic smem, PATH asserts, vec4 bulk+tail (complete)
- **Phase 3** — relative CUDA-event timing; fused beats naive (complete; local [`docs/phase_3_report.md`](docs/phase_3_report.md))
- **Phase 4** — occupancy + T4 Nsight; twins isolate vectorization (MLP, not issue-bound); napkin 2× vs 2.7× closed by static smem
- **Phase 5** — serving harness + full `phase5-v1` matrix on mock and hf_local (complete; local [`docs/phase_5_report.md`](docs/phase_5_report.md))
- **Phase 6** — vLLM + TinyLlama dry run on Colab T4 (complete; local [`docs/phase_6_report.md`](docs/phase_6_report.md))
- **Phase 7** — multi-backend Tier C (preflight ready; plan [`docs/phase_7_plan.md`](docs/phase_7_plan.md), predictions [`docs/phase_7_predictions.md`](docs/phase_7_predictions.md), matrix [`bench/config_matrix_phase7_v1.yaml`](bench/config_matrix_phase7_v1.yaml), driver `scripts/phase7/run_driver.sh`)
- Open follow-ups: [`docs/pending.md`](docs/pending.md)
- Phase 8 — gated; not started

## Quick checks (Phase 0)

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.\.venv-cuda\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

```powershell
.\scripts\build_hello.ps1
```

## RMSNorm correctness (Phases 1–2 + fixes_1)

```powershell
.\scripts\run_rmsnorm_tests.ps1
```

Builds and checks **all four** backends against the CPU golden reference (extended shapes + edge values).

| Piece | Path |
|-------|------|
| CPU golden | `kernels/rmsnorm/rmsnorm_ref.py` |
| Naive (2 launches) | `kernels/rmsnorm/rmsnorm_naive.cu` |
| Fused (global re-read) | `kernels/rmsnorm/rmsnorm_fused.cu` |
| Fused + smem stage | `kernels/rmsnorm/rmsnorm_fused_smem.cu` |
| Fused + smem + float4 | `kernels/rmsnorm/rmsnorm_fused_vec4.cu` |
| Harness | `tests/test_rmsnorm_correctness.py` |
| Pre-Phase-3 checklist | `docs/fixes_1.md` |

## Phase 3 — relative timing

```powershell
.\scripts\run_rmsnorm_bench.ps1
```

Uses `.venv-cuda`, CUDA events (median), and `torch.nn.functional.rms_norm` as the PyTorch baseline. See local [`docs/phase_3_report.md`](docs/phase_3_report.md).

## Phase 4 — profiling / why

```powershell
.\scripts\run_phase4_profile.ps1
```

Occupancy probe (no Nsight counters) explains the smem 4096→8192 regression: static shared drops residency to **3 vs 1** blocks/SM (not napkin 4 vs 2). Full DRAM/stall metrics need unlocked counters or Tier B:

- Notebook: [`notebooks/phase4_colab.ipynb`](notebooks/phase4_colab.ipynb) (prefer T4 / sm_75)
- Linux: `scripts/run_phase4_profile.sh`

See local [`docs/phase_4_report.md`](docs/phase_4_report.md).

## Phase 5 — serving harness (Tier A)

Pre-registered matrix: [`bench/config_matrix_phase5_v1.yaml`](bench/config_matrix_phase5_v1.yaml) (immutable harness proof). Phase 6 matrix: [`bench/config_matrix_phase6_v1.yaml`](bench/config_matrix_phase6_v1.yaml). Plans: [`docs/phase_5_plan.md`](docs/phase_5_plan.md), [`docs/phase_6_plan.md`](docs/phase_6_plan.md). Open items: [`docs/pending.md`](docs/pending.md).

```powershell
# harness math + mock backend (no GPU)
.\.venv\Scripts\python.exe .\tests\test_bench_metrics.py
.\.venv\Scripts\python.exe .\bench\runner.py --backends mock --limit-cells 4

# full mock matrix
.\.venv\Scripts\python.exe .\bench\runner.py --backends mock

# toy HF model on GPU (needs transformers in .venv-cuda)
.\.venv-cuda\Scripts\python.exe .\bench\runner.py --backends hf_local --limit-cells 2
```

## Phase 6 — vLLM dry run (Tier B)

```powershell
# unit test (mocked HTTP stream — no GPU)
.\.venv\Scripts\python.exe .\tests\test_vllm_http.py
```

On Colab T4: [`notebooks/phase6_colab.ipynb`](notebooks/phase6_colab.ipynb) — install cell → restart → serve (`float16`, `enforce_eager`, mem util 0.75) → `--smoke`. Set Colab secret `HF_TOKEN` if needed. Expect XFormers on sm_75; record `run_metadata.json`.

WSL Docker Engine + GPU passthrough is ready for containers; the Phase 6 exit gate is still Colab T4, not the 1650.
