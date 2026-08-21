# Build naive RMSNorm, then run the Phase 1 correctness harness.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot "build_rmsnorm_naive.ps1")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Error "Missing venv python at $py - create the Phase 0 venv first."
}

& $py (Join-Path $Root "tests\test_rmsnorm_naive.py")
exit $LASTEXITCODE
