# KernelFuse — Environment notes (Phase 0)

Captured so later benchmark/profile numbers stay reproducible. Update this file when the machine or toolchain changes.

## Machine (Tier A)

| Item | Value |
|------|--------|
| GPU | NVIDIA GeForce GTX 1650 **Mobile** (PCI `10DE:1F99`, Turing) |
| VRAM | 4096 MiB **GDDR6** (nvidia-smi Max Memory Clock **6001 MHz** = half the data rate ⇒ 12 Gbps × 128-bit ≈ **192 GB/s** peak; GDDR5 tops out well below this clock and ≈128 GB/s) |
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
| Python | 3.10.11 (`E:\Python\Python310`) | Two venvs (see below) |
| PyTorch (CPU golden) | 2.13.0+cpu in `.venv` | Correctness reference only; `cuda_available=False` |
| PyTorch (GPU baseline) | 2.5.1+cu121 in `.venv-cuda` | Phase 3 timing vs built-in; verified on GTX 1650 |
| NumPy | 2.2.6 | Both venvs |

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

**Mobile clocks:** laptop GPUs throttle under sustained load; Nsight Compute replays kernels many times (longer thermal load than a short bench). Sample before/after every Phase 3/4 run:

```text
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu --format=csv
```

(`bench_rmsnorm.py` already logs this per shape.) Try `nvidia-smi -lgc` when elevated — on this machine it failed with “current user does not have permission”; many mobile parts refuse locks even with admin. If clocks cannot be locked, treat before/after samples as mandatory and do not compare %peak across runs where SM/mem clocks drifted.

## WSL2 (Phase 5+ serving)

| Item | Value |
|------|--------|
| Distro | Ubuntu (WSL2), default |
| Kernel | 6.6.87.2-microsoft-standard-WSL2 |
| GPU in WSL | **Yes** — `nvidia-smi` sees GTX 1650, 4096 MiB, driver 526.47 |
| Python in WSL | 3.12.3 (`/usr/bin/python3`) |
| Docker in Ubuntu distro | not installed (`docker` missing); `docker-desktop` WSL distro exists but is separate |
| CUDA toolkit in WSL | not confirmed under `/usr/local/cuda*` yet |

**Why it matters:** vLLM / SGLang expect Linux. On this Windows laptop, serving backends should run **inside WSL2** (or Tier B/C), not native Win32. The Phase 5 harness itself is pure Python and runs on Windows against `mock` / `hf_local`; only Linux-native engines need WSL.

**Quick check:**

```powershell
wsl -d Ubuntu -- nvidia-smi --query-gpu=name,memory.total --format=csv
```

**Still to do for vLLM in WSL (Phase 6 prep, not Phase 5 blocker):** install CUDA toolkit or use pip wheels that bundle CUDA runtime; optionally NVIDIA Container Toolkit if using Docker Desktop with GPU.

## Docker Desktop

| Item | Value |
|------|--------|
| Registered install | `C:\Program Files\Docker\Docker` (Docker Desktop **4.78.0**) — **folder missing / not present** |
| Service | `com.docker.service` — **Stopped** (Manual) |
| Old Desktop data on E: | `E:\Docker\wsl\...` (Desktop WSL disk; not needed) |
| `docker` on PATH / Ubuntu | not installed yet |

**Preferred path (Phase 6): Docker Engine inside WSL2 Ubuntu, space on E:**

Do **not** depend on Docker Desktop. Do **not** put `data-root` directly on `/mnt/e` (9p/DrvFs) — containerd fails with `mkdir .../dev/shm: no such file or directory`.

Use an **ext4 loop image on E:** mounted at `/var/lib/docker` (space on E:, Linux FS semantics):

```powershell
# Fresh install (Engine + toolkit + ext4 data-root):
wsl -d Ubuntu -- bash /mnt/c/Users/kashy/Desktop/KernelFuse/scripts/install_docker_wsl_e.sh

# If Engine is already installed and GPU smoke failed on /mnt/e:
wsl -d Ubuntu -- bash /mnt/c/Users/kashy/Desktop/KernelFuse/scripts/fix_docker_dataroot_ext4.sh
```

Defaults: image `E:\Docker\docker-data.img` (80G sparse), mount `/var/lib/docker`, fstab entry for persistence. After that, `docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi` should work. Optional cleanup of the broken 9p tree: `sudo rm -rf /mnt/e/Docker/engine`.

Phase 5 does **not** need Docker (`mock` / `hf_local`).

## Build helpers

```powershell
.\scripts\build_hello.ps1
.\scripts\run_rmsnorm_tests.ps1   # uses .venv (CPU) for golden + CUDA binaries
.\.venv\Scripts\Activate.ps1      # correctness / golden
.\.venv-cuda\Scripts\Activate.ps1 # Phase 3 GPU PyTorch baseline
```

## Python environments

| Env | Purpose | Torch |
|-----|---------|-------|
| `.venv` | Golden RMSNorm reference + correctness harness | `2.13.0+cpu` |
| `.venv-cuda` | Phase 3 GPU baseline (`torch.nn` / built-in ops on device) | `2.5.1+cu121` (`cuda_available=True` on this 1650) |

Do not replace the CPU env — Phase 1–2 correctness stays on `.venv`.

## RMSNorm kernels (pre-Phase-3)

| Binary | Source | Notes |
|--------|--------|-------|
| `rmsnorm_naive` | `rmsnorm_naive.cu` | Two launches; fp32 `fmaf`; prints `PATH naive` |
| `rmsnorm_fused` | `rmsnorm_fused.cu` | Global re-read fuse; `-DACC_DOUBLE` optional; `PATH fused` |
| `rmsnorm_fused_smem` | `rmsnorm_fused_smem.cu` | Dynamic smem budget via `cudaDevAttrMaxSharedMemoryPerBlockOptin`; `PATH smem\|global` |
| `rmsnorm_fused_vec4` | `rmsnorm_fused_vec4.cu` | Smem + float4 bulk + scalar tail; `PATH vec4\|vec4_tail\|global` |

**Shared-memory budget:** queried at runtime (not hardcoded). On this GTX 1650, opt-in to the advertised max fails and the binaries fall back to the default **48 KiB** → `MAX_SMEM_COLS 11776`. On A100/3090 the same code should opt in to the larger limit so a 16384-wide row can stay on the smem path. Harness asserts `PATH` against `MAX_SMEM_COLS`.

**Harness:** `tests/test_rmsnorm_correctness.py --backend all` — 29 shapes + 14 edge cases (incl. zeros/small at 4096/8192). Wide cols (`>=4096`) use rtol=5e-4 / atol=5e-5.

## Cost / hardware ladder reminder

Phases 0–5 stay on Tier A ($0). Paid Tier C time is reserved for finished measurement only.
