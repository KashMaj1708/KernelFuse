# Phase 5 smoke: metrics unit test + mock harness (no GPU required).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Error "Missing .venv" }

& $py -c "import yaml" 2>$null
if ($LASTEXITCODE -ne 0) {
    & $py -m pip install "PyYAML>=6.0"
}

& $py (Join-Path $Root "tests\test_bench_metrics.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$out = Join-Path $Root "reports\phase5\smoke_mock.csv"
& $py (Join-Path $Root "bench\runner.py") --backends mock --limit-cells 4 --out $out
exit $LASTEXITCODE
