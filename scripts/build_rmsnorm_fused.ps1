# Build kernels/rmsnorm/rmsnorm_fused.cu (Phase 2)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Src = Join-Path $Root "kernels\rmsnorm\rmsnorm_fused.cu"
$Out = Join-Path $Root "kernels\rmsnorm\rmsnorm_fused.exe"

$CudaBin = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin"
if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    if (Test-Path (Join-Path $CudaBin "nvcc.exe")) {
        $env:Path = "$CudaBin;" + $env:Path
        $env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
    } else {
        Write-Error "nvcc not found. Install CUDA Toolkit 12.0 and re-open this shell."
    }
}

$vcvarsCandidates = @(
    "E:\VSC\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)
$vcvars = $vcvarsCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $vcvars) {
    Write-Error "vcvars64.bat not found. Install VS Build Tools with the C++ workload."
}

cmd /c "`"$vcvars`" && nvcc -allow-unsupported-compiler -O2 -arch=sm_75 -o `"$Out`" `"$Src`""
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Built: $Out"
