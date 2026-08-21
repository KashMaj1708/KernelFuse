# Build kernels/rmsnorm/occupancy_probe.cu (Phase 4 Tier-A occupancy API)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "cuda_build_common.ps1")
Build-CudaKernel `
    -Src (Join-Path $Root "kernels\rmsnorm\occupancy_probe.cu") `
    -Out (Join-Path $Root "kernels\rmsnorm\occupancy_probe.exe")
