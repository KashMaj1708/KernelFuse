# Build kernels/rmsnorm/add_rmsnorm_fused.cu (Phase 8 bf16 fused add+RMSNorm)
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "cuda_build_common.ps1")
Build-CudaKernel `
    -Src (Join-Path $Root "kernels\rmsnorm\add_rmsnorm_fused.cu") `
    -Out (Join-Path $Root "kernels\rmsnorm\add_rmsnorm_fused.exe")
