# Paste into a new Colab cell (T4, KernelFuse already extracted).
# 1) CUDA-event bench OUTSIDE ncu  2) ncu issue/inst metrics  3) clocks to file

import os, csv, subprocess, tempfile
from pathlib import Path
import numpy as np

ROOT = Path("/content/KernelFuse")
OUT = ROOT / "reports" / "phase4"
OUT.mkdir(parents=True, exist_ok=True)
os.chdir(ROOT)

def clocks(tag: str):
    line = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,temperature.gpu,power.draw",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    print(f"CLOCKS[{tag}]: {line}")
    with (OUT / "clocks.csv").open("a", newline="") as f:
        f.write(f"{tag},{line}\n")
    return line

(OUT / "clocks.csv").write_text(
    "tag,name,sm,mem,sm_max,mem_max,temp,power\n"
)
clocks("followup-start")

# --- rebuild if needed ---
arch = "sm_75"
bins = {
    "smem": OUT / "rmsnorm_fused_smem",
    "vec4": OUT / "rmsnorm_fused_vec4",
    "fused": OUT / "rmsnorm_fused",
}
for src, dst in [
    ("kernels/rmsnorm/rmsnorm_fused_smem.cu", bins["smem"]),
    ("kernels/rmsnorm/rmsnorm_fused_vec4.cu", bins["vec4"]),
    ("kernels/rmsnorm/rmsnorm_fused.cu", bins["fused"]),
]:
    if not dst.exists():
        subprocess.run(
            ["nvcc", "-O3", "-std=c++17", f"-arch={arch}", src, "-o", str(dst)],
            check=True,
        )

cases = [
    ("smem_4096", bins["smem"], 2048, 4096, 8),
    ("smem_8192", bins["smem"], 2048, 8192, 8),
    ("vec4_8192", bins["vec4"], 2048, 8192, 8),
    ("fused_8192", bins["fused"], 2048, 8192, 12),
]

# --- 1) CUDA events outside ncu (warmup 20, iters 100) ---
print("\n=== CUDA-event bench (outside ncu) ===")
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    rows_out = []
    for name, exe, rows, cols, bpe in cases:
        clocks(name + "-pre")
        rng = np.random.default_rng(0)
        x = td / f"x_{name}.bin"
        w = td / f"w_{name}.bin"
        rng.standard_normal((rows, cols), dtype=np.float32).tofile(x)
        rng.uniform(0.5, 2.0, size=(cols,)).astype(np.float32).tofile(w)
        proc = subprocess.run(
            [str(exe), "bench", str(rows), str(cols), "1e-6", "20", "100", str(x), str(w)],
            capture_output=True,
            text=True,
            check=True,
        )
        print(proc.stdout)
        med = None
        for line in proc.stdout.splitlines():
            if line.startswith("MEDIAN_MS"):
                med = float(line.split()[1])
        nbytes = bpe * rows * cols
        gbps = (nbytes / (med * 1e-3)) / 1e9 if med else None
        print(f"{name}: median_ms={med:.4f}  modeled_eff_bw={gbps:.1f} GB/s  (bpe={bpe})")
        rows_out.append((name, med, gbps, bpe))
        clocks(name + "-post")

with (OUT / "t4_event_bench.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["name", "median_ms", "modeled_eff_bw_GBs", "bytes_per_elem"])
    w.writerows(rows_out)
print("wrote", OUT / "t4_event_bench.csv")

# --- 2) ncu: issue + instruction metrics (occupancy twins) ---
print("\n=== ncu issue/inst (smem_8192 vs vec4_8192) ===")
metrics_safe = ",".join(
    [
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "smsp__issue_active.avg.pct_of_peak_sustained_active",
        "smsp__inst_executed.sum",
        "dram__bytes.sum.per_second",
        "gpu__time_duration.avg",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
        "smsp__warps_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_active",
        "smsp__warps_issue_stalled_barrier.avg.pct_of_peak_sustained_active",
    ]
)

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    for name, exe, rows, cols, _bpe in [
        ("smem_8192_issue", bins["smem"], 2048, 8192, 8),
        ("vec4_8192_issue", bins["vec4"], 2048, 8192, 8),
    ]:
        clocks(name)
        x = td / "x.bin"
        w = td / "w.bin"
        rng = np.random.default_rng(0)
        rng.standard_normal((rows, cols), dtype=np.float32).tofile(x)
        rng.uniform(0.5, 2.0, size=(cols,)).astype(np.float32).tofile(w)
        rep = OUT / name
        cmd = [
            "ncu",
            "--target-processes",
            "all",
            "--launch-count",
            "1",
            "--metrics",
            metrics_safe,
            "-o",
            str(rep),
            "--force-overwrite",
            str(exe),
            "bench",
            str(rows),
            str(cols),
            "1e-6",
            "0",
            "1",
            str(x),
            str(w),
        ]
        print("running", name)
        subprocess.run(cmd, check=False)
        subprocess.run(
            ["ncu", "--import", f"{rep}.ncu-rep", "--csv", "--page", "raw"],
            stdout=open(OUT / f"{name}.csv", "w"),
            check=False,
        )

clocks("followup-end")

# Zip everything needed for the laptop report (skip Linux ELF binaries).
import zipfile

zip_path = Path("/content/phase4_followup.zip")
include_globs = [
    "t4_event_bench.csv",
    "clocks.csv",
    "*_issue.csv",
    "*_issue.ncu-rep",
    "*_issue_ncu.log",
    "*_event.log",
    "occupancy_probe.txt",
    "smem_*.csv",
    "vec4_*.csv",
    "fused_*.csv",
    "smem_*.ncu-rep",
    "vec4_*.ncu-rep",
    "fused_*.ncu-rep",
    "*_ncu.log",
]
seen = set()
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for pattern in include_globs:
        for p in OUT.glob(pattern):
            if not p.is_file() or p.name in seen:
                continue
            if p.stat().st_size > 5_000_000:  # skip huge ELF rebuilds if any
                continue
            seen.add(p.name)
            zf.write(p, arcname=f"reports/phase4/{p.name}")
print(f"\nWrote {zip_path} ({zip_path.stat().st_size} bytes, {len(seen)} files)")
print("Download /content/phase4_followup.zip from the Colab Files panel.")
print("Interpretation:")
print("  events ~170-200 GB/s vs ncu dram ~24  => ncu absolute is artifact; ratios OK")
print("  inst_executed scalar/vec4 ~4  AND issue_active high  => issue-bound")
print("  issue_active low on both                     => MLP / bytes-in-flight")
