# Build add_rmsnorm + run Phase 8 correctness gate (M1).
$Root = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot "build_add_rmsnorm_fused.ps1")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python (Join-Path $Root "tests\test_add_rmsnorm_correctness.py")
exit $LASTEXITCODE
