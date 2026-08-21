# KernelFuse

Two-layer inference-performance project:

1. **Serving benchmark** — compare vLLM / SGLang / TensorRT-LLM on throughput and latency percentiles.
2. **Hand-written fused CUDA kernel** — RMSNorm (naive → fused), correctness-gated, then profiled to show why fusion helps on a memory-bound path.

Develop on a GTX 1650 (or free cloud) first; rent a 24 GB-class card only for final measurement. See [`KernelFuse_build_plan.md`](KernelFuse_build_plan.md) for the phased plan and [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) for machine/toolchain notes.

Local write-ups (gitignored): [`docs/phase_0_report.md`](docs/phase_0_report.md), [`docs/phase_1_report.md`](docs/phase_1_report.md), [`docs/phase_2_report.md`](docs/phase_2_report.md).

## Status

- **Phase 0** — environment setup and sanity check (complete; see [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) and local [`docs/phase_0_report.md`](docs/phase_0_report.md))
- **Phase 1** — naive CUDA RMSNorm matches CPU golden reference (complete; local [`docs/phase_1_report.md`](docs/phase_1_report.md))
- **Phase 2** — fused RMSNorm (single launch) matches same harness (complete; local [`docs/phase_2_report.md`](docs/phase_2_report.md))
- Phases 3–8 — gated; not started

## Quick checks (Phase 0)

```powershell
nvidia-smi
nvcc --version
ncu --version
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.tensor([1.0])+1)"
```

Build and run the trivial CUDA hello kernel:

```powershell
.\scripts\build_hello.ps1
.\kernels\hello\hello.exe
```

## Phase 1–2 — RMSNorm correctness

```powershell
.\scripts\run_rmsnorm_tests.ps1
```

Runs **naive and fused** against the CPU golden reference (20 shapes).

| Piece | Path |
|-------|------|
| CPU golden | `kernels/rmsnorm/rmsnorm_ref.py` |
| Naive CUDA (2 launches) | `kernels/rmsnorm/rmsnorm_naive.cu` |
| Fused CUDA (1 launch) | `kernels/rmsnorm/rmsnorm_fused.cu` |
| Harness | `tests/test_rmsnorm_correctness.py` |
