# Build all RMSNorm variants, then run Phase 3 relative timing with .venv-cuda.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

$builders = @(
    "build_rmsnorm_naive.ps1",
    "build_rmsnorm_fused.ps1",
    "build_rmsnorm_fused_smem.ps1",
    "build_rmsnorm_fused_vec4.ps1"
)
foreach ($b in $builders) {
    & (Join-Path $PSScriptRoot $b)
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$py = Join-Path $Root ".venv-cuda\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Error "Missing .venv-cuda python - install requirements-cuda.txt first."
}

& $py (Join-Path $Root "tests\bench_rmsnorm.py") @args
exit $LASTEXITCODE
