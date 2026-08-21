# Phase 4 local profile attempt + occupancy probe.
# Nsight on this 1650 needs ERR_NVGPUCTRPERM unlocked; occupancy API always works.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

function Write-GpuClocks([string]$Tag) {
    Write-Host "=== GPU clocks ($Tag) ==="
    nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,temperature.gpu,power.draw --format=csv
}

function Find-Ncu {
    $candidates = @(
        (Join-Path $Root "tools\nsight-compute-2022.4.1\nsight_compute-windows-x86_64-2022.4.1.6-archive\nsight-compute\2022.4.1\target\windows-desktop-win7-x64\ncu.exe"),
        "C:\Program Files\NVIDIA Corporation\Nsight Compute 2022.4.1\ncu.bat",
        "C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.4.1\ncu.bat"
    )
    if (Get-Command ncu -ErrorAction SilentlyContinue) {
        return (Get-Command ncu).Source
    }
    return $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}

function Write-Bins([string]$Py, [string]$XPath, [string]$WPath, [int]$Rows, [int]$Cols) {
    $env:KF_P4_X = $XPath
    $env:KF_P4_W = $WPath
    $env:KF_P4_ROWS = "$Rows"
    $env:KF_P4_COLS = "$Cols"
    & $Py -c "import os,numpy as np; r=int(os.environ['KF_P4_ROWS']); c=int(os.environ['KF_P4_COLS']); rng=np.random.default_rng(0); rng.standard_normal((r,c),dtype=np.float32).tofile(os.environ['KF_P4_X']); rng.uniform(0.5,2.0,size=(c,)).astype(np.float32).tofile(os.environ['KF_P4_W']); print('wrote',r,c)"
}

Write-GpuClocks "suite-start"

& (Join-Path $PSScriptRoot "build_occupancy_probe.ps1")
$probe = Join-Path $Root "kernels\rmsnorm\occupancy_probe.exe"
Write-Host "=== occupancy probe (no counters) ==="
& $probe
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$ncu = Find-Ncu
if (-not $ncu) {
    Write-Warning "ncu not found - occupancy probe only. Use Colab notebook for full Phase 4."
    Write-GpuClocks "suite-end"
    exit 0
}

$ncu2022 = Join-Path $Root "tools\nsight-compute-2022.4.1\nsight_compute-windows-x86_64-2022.4.1.6-archive\nsight-compute\2022.4.1\target\windows-desktop-win7-x64\ncu.exe"
if (Test-Path $ncu2022) { $ncu = $ncu2022 }
Write-Host "Using ncu: $ncu"

& (Join-Path $PSScriptRoot "build_rmsnorm_fused_smem.ps1")
& (Join-Path $PSScriptRoot "build_rmsnorm_fused_vec4.ps1")
& (Join-Path $PSScriptRoot "build_rmsnorm_fused.ps1")

$OutDir = Join-Path $Root "reports\phase4"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$Tmp = Join-Path $env:TEMP "kf_p4_$PID"
New-Item -ItemType Directory -Force -Path $Tmp | Out-Null

$py = Join-Path $Root ".venv-cuda\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = Join-Path $Root ".venv\Scripts\python.exe" }

$cases = @(
    @{ Name = "smem_4096"; Exe = "rmsnorm_fused_smem.exe"; Rows = 2048; Cols = 4096 },
    @{ Name = "smem_8192"; Exe = "rmsnorm_fused_smem.exe"; Rows = 2048; Cols = 8192 },
    @{ Name = "vec4_8192"; Exe = "rmsnorm_fused_vec4.exe"; Rows = 2048; Cols = 8192 },
    @{ Name = "fused_8192"; Exe = "rmsnorm_fused.exe"; Rows = 2048; Cols = 8192 }
)

$metrics = "sm__warps_active.avg.pct_of_peak_sustained_active,launch__occupancy_limit_shared_mem,launch__occupancy_limit_blocks,launch__occupancy_limit_registers,dram__bytes_read.sum,dram__bytes_write.sum,dram__bytes.sum.per_second,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,smsp__warps_issue_stalled_barrier.avg.pct_of_peak_sustained_active,smsp__warps_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_active,smsp__warps_issue_stalled_short_scoreboard.avg.pct_of_peak_sustained_active,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,gpu__time_duration.avg"

$ncuOk = $false
foreach ($c in $cases) {
    Write-GpuClocks $c.Name
    $x = Join-Path $Tmp ("x_" + $c.Name + ".bin")
    $w = Join-Path $Tmp ("w_" + $c.Name + ".bin")
    Write-Bins -Py $py -XPath $x -WPath $w -Rows $c.Rows -Cols $c.Cols
    $exe = Join-Path $Root ("kernels\rmsnorm\" + $c.Exe)
    $rep = Join-Path $OutDir $c.Name
    Write-Host ("=== ncu " + $c.Name + " ===")
    & $ncu --target-processes all --launch-count 1 `
        --metrics $metrics `
        -o $rep --force-overwrite `
        $exe bench $c.Rows $c.Cols 1e-6 0 1 $x $w 2>&1 |
        Tee-Object -FilePath (Join-Path $OutDir ($c.Name + "_ncu.log"))
    if ($LASTEXITCODE -eq 0) { $ncuOk = $true }
}

Write-GpuClocks "suite-end"
Remove-Item -Recurse -Force $Tmp -ErrorAction SilentlyContinue

if (-not $ncuOk) {
    Write-Host ""
    Write-Host "Nsight counters blocked on Tier A (ERR_NVGPUCTRPERM or driver mismatch)."
    Write-Host "Occupancy probe above is valid. For DRAM/stall metrics run:"
    Write-Host "  notebooks/phase4_colab.ipynb  (Tier B, prefer T4 = sm_75)"
    exit 0
}
Write-Host "Nsight reports under $OutDir"
exit 0
