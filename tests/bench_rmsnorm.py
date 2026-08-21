"""
Phase 3 — local relative timing (Tier A / GTX 1650).

Methodology (hard rules):
  - CUDA: events around kernel launch(es) only; .bin I/O outside timed region
  - Warmup launches before sampling; report MEDIAN ms (not mean)
  - PyTorch baseline: torch.nn.functional.rms_norm (API name); see report for
    whether this torch build actually fuses (2.5.1+cu121 is composite)
  - Record torch version next to every baseline number; compare within-tier only
  - Effective bandwidth = bytes_moved / median_time (roofline-ready)

Usage:
  .\\.venv-cuda\\Scripts\\python.exe .\\tests\\bench_rmsnorm.py
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]

BACKENDS = {
    "naive": ROOT / "kernels" / "rmsnorm" / "rmsnorm_naive.exe",
    "fused": ROOT / "kernels" / "rmsnorm" / "rmsnorm_fused.exe",
    "fused_smem": ROOT / "kernels" / "rmsnorm" / "rmsnorm_fused_smem.exe",
    "fused_vec4": ROOT / "kernels" / "rmsnorm" / "rmsnorm_fused_vec4.exe",
}

BENCH_SHAPES = [
    (256, 1024),
    (1024, 4096),
    (2048, 4096),
    (4096, 4096),
    (2048, 8192),
    (1000, 3584),
    (4, 4095),
    (4, 4096),
    (4, 4097),
]

EPS = 1e-6
DEFAULT_WARMUP = 20
DEFAULT_ITERS = 100

# Bytes moved per element (fp32), for effective-bandwidth / roofline.
# fused/naive: read x twice + write out = 12 B/elem (weight amortized separately).
# smem/vec4:   read x once + write out = 8 B/elem (perfect staging).
# pytorch:     multi-kernel composite on 2.5.1 — no single-stream byte count.
BYTES_PER_ELEM = {
    "naive": 12,
    "fused": 12,
    "fused_smem": 8,
    "fused_vec4": 8,
    "pytorch": None,
}

# GTX 1650 Mobile (PCI 10DE:1F99), Max Memory Clock 6001 MHz => GDDR6 @ 12 Gbps,
# 128-bit bus => 192 GB/s peak. (GDDR5 parts top out near 128 GB/s.)
PEAK_BW_GB_S = 192.0

# sm_75: 64 KiB smem / SM, 1024 threads / SM, blockDim=256.
# cols=4096 => 16 KiB/block => 4 blocks => 1024 threads (full).
# cols=8192 => 32 KiB/block => 2 blocks => 512 threads (half) => ~2x occupancy
# prediction; measured smem BW drop 122.3 -> 45.8 GB/s is ~2.7x (Phase 4 gap).


def query_gpu_telemetry() -> dict[str, str]:
    """Sample SM/mem clocks and temp for mobile-throttle logging."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return {}
    parts = [p.strip() for p in out.split(",")]
    if len(parts) < 6:
        return {"raw": out}
    return {
        "sm_mhz": parts[0],
        "mem_mhz": parts[1],
        "sm_max_mhz": parts[2],
        "mem_max_mhz": parts[3],
        "temp_c": parts[4],
        "power_w": parts[5],
    }


def fmt_telemetry(t: dict[str, str]) -> str:
    if not t:
        return "n/a"
    if "raw" in t and len(t) == 1:
        return t["raw"]
    return (
        f"sm={t.get('sm_mhz', '?')} MHz (max {t.get('sm_max_mhz', '?')})  "
        f"mem={t.get('mem_mhz', '?')} MHz (max {t.get('mem_max_mhz', '?')})  "
        f"temp={t.get('temp_c', '?')} C  power={t.get('power_w', '?')} W"
    )


def bytes_moved(backend: str, rows: int, cols: int) -> int | None:
    bpe = BYTES_PER_ELEM.get(backend)
    if bpe is None:
        return None
    return bpe * rows * cols


def eff_bw_gb_s(nbytes: int | None, median_ms: float) -> float | None:
    if nbytes is None or median_ms <= 0:
        return None
    return (nbytes / (median_ms * 1e-3)) / 1e9


def parse_bench_stdout(stdout: str) -> tuple[float, str | None, int | None]:
    median = None
    path = None
    max_cols = None
    for line in stdout.splitlines():
        line = line.strip()
        m = re.match(r"MEDIAN_MS\s+([0-9.eE+-]+)", line)
        if m:
            median = float(m.group(1))
        m = re.match(r"PATH\s+(\S+)", line)
        if m:
            path = m.group(1)
        m = re.match(r"MAX_SMEM_COLS\s+(\d+)", line)
        if m:
            max_cols = int(m.group(1))
    if median is None:
        raise RuntimeError(f"missing MEDIAN_MS in stdout:\n{stdout}")
    return median, path, max_cols


def bench_cuda(
    exe: Path, rows: int, cols: int, x: np.ndarray, w: np.ndarray, warmup: int, iters: int
) -> tuple[float, str | None, int | None]:
    with tempfile.TemporaryDirectory(prefix="kf_bench_") as td:
        td_path = Path(td)
        xp = td_path / "x.bin"
        wp = td_path / "w.bin"
        x.tofile(xp)
        w.tofile(wp)
        cmd = [
            str(exe),
            "bench",
            str(rows),
            str(cols),
            repr(float(EPS)),
            str(warmup),
            str(iters),
            str(xp),
            str(wp),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"bench failed rc={proc.returncode}\ncmd={' '.join(cmd)}\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        return parse_bench_stdout(proc.stdout)


def bench_pytorch(rows: int, cols: int, x: np.ndarray, w: np.ndarray, warmup: int, iters: int) -> float:
    """Time with torch.cuda.Event + synchronize (not Python wall clock)."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available in this interpreter - use .venv-cuda")

    xt = torch.from_numpy(x).cuda()
    wt = torch.from_numpy(w).cuda()
    normalized_shape = (cols,)

    for _ in range(warmup):
        _ = F.rms_norm(xt, normalized_shape, weight=wt, eps=EPS)
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = F.rms_norm(xt, normalized_shape, weight=wt, eps=EPS)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))

    samples.sort()
    n = len(samples)
    if n % 2 == 1:
        return samples[n // 2]
    return 0.5 * (samples[n // 2 - 1] + samples[n // 2])


def fmt_ms(ms: float) -> str:
    return f"{ms:10.4f}"


def fmt_bw(gb: float | None) -> str:
    if gb is None:
        return f"{'n/a':>10}"
    return f"{gb:10.1f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 RMSNorm relative timing")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["naive", "fused", "fused_smem", "fused_vec4", "pytorch"],
    )
    parser.add_argument(
        "--peak-bw",
        type=float,
        default=PEAK_BW_GB_S,
        help="Device peak DRAM GB/s for %%peak (default 192 = GDDR6 1650)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available() and "pytorch" in args.backends:
        print(
            "FAIL: need .venv-cuda with CUDA torch for pytorch baseline "
            f"(got {torch.__version__}, cuda={torch.cuda.is_available()})",
            file=sys.stderr,
        )
        return 1

    torch_ver = torch.__version__
    cuda_runtime = getattr(torch.version, "cuda", None)
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a"

    print("=== Phase 3 relative timing (Tier A) ===")
    print(f"device: {device}")
    print(f"torch: {torch_ver}  torch.version.cuda: {cuda_runtime}")
    print("pytorch API: torch.nn.functional.rms_norm")
    print(
        "pytorch note: torch.profiler on 2.5.1+cu121 shows composite "
        "(pow/mean/rsqrt/mul) - not a single fused CUDA kernel"
    )
    print(f"timing: torch.cuda.Event + synchronize; CUDA binaries use cudaEventElapsedTime")
    print(f"warmup={args.warmup}  iters={args.iters}  metric=median_ms")
    print(
        f"peak_bw={args.peak_bw:.1f} GB/s (GDDR6 1650 Mobile PCI 1F99, "
        "mem clock max 6001 MHz)"
    )
    print(
        "bytes/elem: fused/naive=12 (x x2 + out), smem/vec4=8 (x x1 + out); "
        "pytorch=n/a (multi-kernel)"
    )
    print(
        "note: at cols=4095/4097, fused_vec4 ~ fused_smem is expected "
        "(per-row 16B misalignment -> scalar path after row 0)"
    )
    print(
        "motivation: torch 2.5.1 F.rms_norm is composite (pow/mean/rsqrt/mul) — "
        "same multi-pass tax as naive; naive-vs-fused is the controlled version"
    )
    print()

    # Two tables: time and effective bandwidth
    print("--- median ms ---")
    header = f"{'shape':>14}  " + "  ".join(f"{b:>10}" for b in args.backends)
    print(header)
    print("-" * len(header))

    results: list[tuple[int, int, dict[str, float], dict[str, str | None]]] = []

    before_all = query_gpu_telemetry()
    print(f"GPU before suite: {fmt_telemetry(before_all)}")
    print()

    for rows, cols in BENCH_SHAPES:
        rng = np.random.default_rng(rows * 10007 + cols)
        x = rng.standard_normal((rows, cols), dtype=np.float32)
        w = rng.uniform(0.5, 2.0, size=(cols,)).astype(np.float32)

        tele_before = query_gpu_telemetry()
        times: dict[str, float] = {}
        paths: dict[str, str | None] = {}
        cells: list[str] = []
        path_notes: list[str] = []
        for b in args.backends:
            if b == "pytorch":
                ms = bench_pytorch(rows, cols, x, w, args.warmup, args.iters)
                times[b] = ms
                paths[b] = f"composite@{torch_ver}"
                cells.append(fmt_ms(ms))
                path_notes.append(f"pytorch={torch_ver}")
            else:
                exe = BACKENDS[b]
                if not exe.is_file():
                    print(f"FAIL: missing {exe}", file=sys.stderr)
                    return 1
                ms, path, _maxc = bench_cuda(
                    exe, rows, cols, x, w, args.warmup, args.iters
                )
                times[b] = ms
                paths[b] = path
                cells.append(fmt_ms(ms))
                path_notes.append(f"{b}:{path}")

        tele_after = query_gpu_telemetry()
        shape_s = f"({rows},{cols})"
        print(shape_s.rjust(14) + "  " + "  ".join(cells))
        print(" " * 16 + "  ".join(path_notes))
        print(
            " " * 16
            + f"clocks before: {fmt_telemetry(tele_before)} | "
            + f"after: {fmt_telemetry(tele_after)}"
        )
        results.append((rows, cols, times, paths))

    print()
    print(f"--- effective bandwidth GB/s  (peak {args.peak_bw:.0f}) ---")
    header2 = f"{'shape':>14}  " + "  ".join(f"{b:>10}" for b in args.backends)
    print(header2)
    print("-" * len(header2))
    for rows, cols, times, _paths in results:
        cells = []
        for b in args.backends:
            gb = eff_bw_gb_s(bytes_moved(b, rows, cols), times[b])
            cells.append(fmt_bw(gb))
        print(f"({rows},{cols})".rjust(14) + "  " + "  ".join(cells))

    print()
    print("--- % of peak (custom kernels only) ---")
    header3 = f"{'shape':>14}  " + "  ".join(
        f"{b:>10}" for b in args.backends if b != "pytorch"
    )
    print(header3)
    print("-" * len(header3))
    for rows, cols, times, _paths in results:
        cells = []
        for b in args.backends:
            if b == "pytorch":
                continue
            gb = eff_bw_gb_s(bytes_moved(b, rows, cols), times[b])
            if gb is None:
                cells.append(f"{'n/a':>10}")
            else:
                cells.append(f"{100.0 * gb / args.peak_bw:9.1f}%")
        print(f"({rows},{cols})".rjust(14) + "  " + "  ".join(cells))

    print()
    after_all = query_gpu_telemetry()
    print(f"GPU after suite:  {fmt_telemetry(after_all)}")
    print(
        "note: Nsight Compute replays kernels many times (longer thermal load than "
        "100-iter bench). Compare before/after clocks; sag shifts %peak. "
        "nvidia-smi -lgc needs admin and often fails on mobile anyway."
    )
    print()
    print("Harness self-check: (4096,4096) and (2048,8192) share element count;")
    print("fused times should nearly match if timing is measuring bytes, not shape quirks.")
    print(
        "Phase 4: smem 4096->8192 occupancy predicts ~2x (4 vs 2 blocks/SM); "
        "measured BW 122.3->45.8 is ~2.7x — find the extra 0.7x "
        "(DRAM throughput, stall reasons, __syncthreads)."
    )
    print("Exit-gate: fused should beat naive at large widths (Tier A relative only).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
