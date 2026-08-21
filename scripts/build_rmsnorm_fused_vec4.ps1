# Build kernels/rmsnorm/rmsnorm_fused_vec4.cu
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "cuda_build_common.ps1")
Build-CudaKernel `
    -Src (Join-Path $Root "kernels\rmsnorm\rmsnorm_fused_vec4.cu") `
    -Out (Join-Path $Root "kernels\rmsnorm\rmsnorm_fused_vec4.exe")
