# KernelFuse — Environment notes (Phase 0)

Captured so later benchmark/profile numbers stay reproducible. Update this file when the machine or toolchain changes.

## Machine (Tier A)

| Item | Value |
|------|--------|
| GPU | NVIDIA GeForce GTX 1650 (Turing, `sm_75`) |
| VRAM | 4096 MiB |
| Driver | 526.47 |
| Driver-reported CUDA | 12.0 |
| OS | Windows 10/11 (build 26200) |
| Host compiler | MSVC 19.40 (VS 2022 17.10.3 at `E:\VSC`) |
| Fallback MSVC | 14.33 / VS Build Tools 17.3.6 (links against current UCRT fail on this box) |

## Toolchain installed for Phase 0

| Tool | Version / path | Notes |
|------|----------------|-------|
| CUDA Toolkit | 12.0.140 (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0`) | Matched to driver CUDA 12.0 max. winget’s CUDA 13.3 would not run on this driver. |
| `nvcc` | 12.0, V12.0.140 | Needs `-allow-unsupported-compiler` with MSVC 19.40 |
| Nsight Compute | 2025.4.1 (winget) | **Incompatible with driver 526.47** |
| Nsight Compute | 2022.4.1.6 (CUDA 12.0 redist, extracted under `tools/`) | Driver-compatible; counters still blocked without admin / NVGPUCTRPERM fix |
| Python | 3.10.11 (`E:\Python\Python310`) | Project venv: `.venv` |
| PyTorch | 2.13.0+cpu | CPU-only wheel; `torch.cuda.is_available() == False` by design for Phase 0 |
| NumPy | 2.2.6 | Installed into `.venv` |

## Sanity checks run

1. **`nvidia-smi`** — OK (GTX 1650, 4 GB, driver 526.47).
2. **`nvcc --version`** — OK after CUDA 12.0.1 toolkit install.
3. **Hello kernel** — `kernels/hello/hello.cu` compiles (`scripts/build_hello.ps1`), launches 4 threads, prints from device, host sync succeeds.
4. **Python** — `import torch` and a CPU tensor op succeed in `.venv`.
5. **Nsight Compute (`ncu`)** — CLI launches and attaches to `hello.exe`. Report status:

   - Nsight **2025.4.1**: `Cuda driver is not compatible with Nsight Compute` on driver 526.47.
   - Nsight **2022.4.1** (CUDA 12.0 redist): driver OK; still fails with `ERR_NVGPUCTRPERM` / “No kernels were profiled” when run as a normal user. Elevated run was not completed (UAC canceled).
   - **No `.ncu-rep` written yet** until counter permission is enabled once.

### Profiling implication (not a Phase 0 blocker)

On this GTX 1650, **memory/occupancy counters are restricted for non-admin users today**. Phase 4 should plan on **Tier B (Colab/Kaggle) or Tier C** for the clean profiling pass, unless we unlock local counters:

1. NVIDIA Control Panel → Desktop (enable Developer settings if needed) → Developer → enable **“Allow access to GPU performance counters to all users”**, then reboot, **or**
2. Run `ncu` elevated once for a smoke report, **and**
3. Prefer Nsight **2022.4.x** with driver 526.47 (or update the driver if using a newer `ncu`).

## Build helpers

```powershell
.\scripts\build_hello.ps1      # compile + run hello kernel
.\scripts\profile_hello.ps1    # ncu pass (expects ncu on PATH / known install)
.\.venv\Scripts\Activate.ps1
python -c "import torch; print(torch.__version__)"
```

## Cost / hardware ladder reminder

Phases 0–5 stay on Tier A ($0). Paid Tier C time is reserved for finished measurement only.
