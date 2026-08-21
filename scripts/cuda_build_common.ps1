# Shared nvcc helper used by build_rmsnorm_*.ps1 scripts.
function Build-CudaKernel {
    param(
        [Parameter(Mandatory = $true)][string]$Src,
        [Parameter(Mandatory = $true)][string]$Out,
        [string]$ExtraNvccArgs = ""
    )
    $ErrorActionPreference = "Stop"

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

    $nvccCmd = "nvcc -allow-unsupported-compiler -O2 -arch=sm_75 $ExtraNvccArgs -o `"$Out`" `"$Src`""
    cmd /c "`"$vcvars`" && $nvccCmd"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host "Built: $Out"
}
