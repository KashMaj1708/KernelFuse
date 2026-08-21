# Build kernels/rmsnorm/rmsnorm_fused.cu (fp32 accum; pass -DACC_DOUBLE via ExtraNvccArgs to rebuild double path)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "cuda_build_common.ps1")
Build-CudaKernel `
    -Src (Join-Path $Root "kernels\rmsnorm\rmsnorm_fused.cu") `
    -Out (Join-Path $Root "kernels\rmsnorm\rmsnorm_fused.exe")
