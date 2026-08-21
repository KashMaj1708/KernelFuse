# Profile kernels/hello/hello.exe with Nsight Compute (Phase 0)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Bin = Join-Path $Root "kernels\hello\hello.exe"
$Report = Join-Path $Root "kernels\hello\hello"

if (-not (Test-Path $Bin)) {
    Write-Error "Missing $Bin — run scripts\build_hello.ps1 first."
}

$ncuCandidates = @(
    (Join-Path $Root "tools\nsight-compute-2022.4.1\nsight_compute-windows-x86_64-2022.4.1.6-archive\nsight-compute\2022.4.1\target\windows-desktop-win7-x64\ncu.exe"),
    "C:\Program Files\NVIDIA Corporation\Nsight Compute 2022.4.1\ncu.bat",
    "C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.4.1\ncu.bat"
)
$ncu = $null
if (Get-Command ncu -ErrorAction SilentlyContinue) {
    $ncu = (Get-Command ncu).Source
} else {
    $ncu = $ncuCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $ncu) {
    Write-Error "ncu not found. Install Nsight Compute (prefer 2022.4.x with driver 526.x) and re-open this shell."
}

Write-Host "Using: $ncu"
# Thin first pass. On GTX 1650 + old drivers, expect ERR_NVGPUCTRPERM and/or driver mismatch;
# that is recorded in docs/ENVIRONMENT.md and routes Phase 4 profiling to Tier B/C.
& $ncu --target-processes all -o $Report --force-overwrite $Bin
Write-Host "Attempted report path: $Report*.ncu-rep (may be missing if counters/driver blocked)"
