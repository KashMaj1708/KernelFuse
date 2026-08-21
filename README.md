# KernelFuse

Two-layer inference-performance project:

1. **Serving benchmark** — compare vLLM / SGLang / TensorRT-LLM on throughput and latency percentiles.
2. **Hand-written fused CUDA kernel** — RMSNorm (naive → fused), correctness-gated, then profiled to show why fusion helps on a memory-bound path.

Develop on a GTX 1650 (or free cloud) first; rent a 24 GB-class card only for final measurement. See [`KernelFuse_build_plan.md`](KernelFuse_build_plan.md) for the phased plan and [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) for machine/toolchain notes.

Local write-ups (gitignored): [`docs/phase_0_report.md`](docs/phase_0_report.md), [`docs/phase_1_report.md`](docs/phase_1_report.md), [`docs/phase_2_report.md`](docs/phase_2_report.md), [`docs/fix_report_1.md`](docs/fix_report_1.md).

## Status

- **Phase 0** — environment setup and sanity check (complete)
- **Phase 1** — naive CUDA RMSNorm matches CPU golden reference (complete)
- **Phase 2** — fused RMSNorm (single launch) matches same harness (complete)
- **fixes_1** — pre-Phase-3 baseline: four variants, extended harness (complete; local [`docs/fix_report_1.md`](docs/fix_report_1.md))
- **Pre-Phase-3 blockers** — `.venv-cuda` (cu121), dynamic smem, PATH asserts, vec4 bulk+tail (complete)
- Phases 3–8 — gated; not started

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
