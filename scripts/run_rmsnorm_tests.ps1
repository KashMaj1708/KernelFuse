# Build all RMSNorm variants, then run the full correctness harness.
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

$py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Error "Missing venv python at $py - create the Phase 0 venv first."
}

& $py (Join-Path $Root "tests\test_rmsnorm_correctness.py") --backend all
exit $LASTEXITCODE
