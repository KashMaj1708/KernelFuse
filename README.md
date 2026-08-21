# KernelFuse

Two-layer inference-performance project:

1. **Serving benchmark** — compare vLLM / SGLang / TensorRT-LLM on throughput and latency percentiles.
2. **Hand-written fused CUDA kernel** — RMSNorm (naive → fused), correctness-gated, then profiled to show why fusion helps on a memory-bound path.

Develop on a GTX 1650 (or free cloud) first; rent a 24 GB-class card only for final measurement. See [`KernelFuse_build_plan.md`](KernelFuse_build_plan.md) for the phased plan and [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) for machine/toolchain notes.

Local write-ups (gitignored): [`docs/phase_0_report.md`](docs/phase_0_report.md).

## Status

- **Phase 0** — environment setup and sanity check (complete; see [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) and local [`docs/phase_0_report.md`](docs/phase_0_report.md))
- **Phase 1** — naive CUDA RMSNorm matches CPU golden reference on all harness shapes (complete)
- Phases 2–8 — gated; not started

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

## Phase 1 — RMSNorm correctness

```powershell
.\scripts\run_rmsnorm_tests.ps1
```

CPU golden reference: `kernels/rmsnorm/rmsnorm_ref.py`  
Naive CUDA (two unfused kernels): `kernels/rmsnorm/rmsnorm_naive.cu`  
Harness: `tests/test_rmsnorm_naive.py`
