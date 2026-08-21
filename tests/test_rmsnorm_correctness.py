"""
RMSNorm correctness harness (Phases 1–2 + fixes_1 + pre-Phase-3).

Compares CUDA binaries against the NumPy/PyTorch golden reference.
Fails loudly on mismatch. Asserts PATH instrumentation for smem/vec4 backends.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kernels" / "rmsnorm"))

from rmsnorm_ref import rmsnorm_numpy, rmsnorm_torch  # noqa: E402

BASE_SHAPES = [
    (1, 1),
    (1, 3),
    (1, 7),
    (1, 8),
    (1, 16),
    (1, 17),
    (1, 64),
    (1, 127),
    (1, 128),
    (1, 256),
    (1, 1024),
    (2, 7),
    (3, 16),
    (4, 33),
    (8, 64),
    (16, 128),
    (32, 256),
    (64, 512),
    (7, 768),
    (13, 1024),
]

REALISTIC_SHAPES = [
    (4, 3584),
    (4, 4096),
    (4, 8192),
    (4, 4095),
    (4, 4097),
    (4096, 4096),
    (2048, 8192),
    (1000, 3584),
    (2, 20000),  # smem-budget guard → global fallback on 1650
]

TEST_SHAPES = BASE_SHAPES + REALISTIC_SHAPES

EPS = 1e-6
ATOL = 1e-5
RTOL = 1e-4
WIDE_COLS = 4096
WIDE_RTOL = 5e-4
WIDE_ATOL = 5e-5

BACKENDS = {
    "naive": ROOT / "kernels" / "rmsnorm" / "rmsnorm_naive.exe",
    "fused": ROOT / "kernels" / "rmsnorm" / "rmsnorm_fused.exe",
    "fused_smem": ROOT / "kernels" / "rmsnorm" / "rmsnorm_fused_smem.exe",
    "fused_vec4": ROOT / "kernels" / "rmsnorm" / "rmsnorm_fused_vec4.exe",
}

ALL_BACKENDS = ["naive", "fused", "fused_smem", "fused_vec4"]

# Shapes where we assert a specific PATH (cols-keyed after device max is known).
PATH_ASSERT_COLS = {
    "fused_smem": {8192: "smem", 20000: "global", 4096: "smem"},
    "fused_vec4": {
        8192: "vec4",
        4096: "vec4",
        4095: "vec4_tail",
        4097: "vec4_tail",
        20000: "global",
    },
}


def tolerances_for(cols: int) -> tuple[float, float]:
    if cols >= WIDE_COLS:
        return WIDE_RTOL, WIDE_ATOL
    return RTOL, ATOL


def make_weight(rng: np.random.Generator, cols: int) -> np.ndarray:
    return rng.uniform(0.5, 2.0, size=(cols,)).astype(np.float32)


@dataclass
class CudaRun:
    out: np.ndarray
    path: str | None
    max_smem_cols: int | None
    stdout: str


def parse_runner_stdout(stdout: str) -> tuple[str | None, int | None]:
    path = None
    max_cols = None
    for line in stdout.splitlines():
        m = re.match(r"PATH\s+(\S+)", line.strip())
        if m:
            path = m.group(1)
        m = re.match(r"MAX_SMEM_COLS\s+(\d+)", line.strip())
        if m:
            max_cols = int(m.group(1))
    return path, max_cols


def expected_path(backend: str, cols: int, max_smem_cols: int | None) -> str | None:
    if backend == "naive":
        return "naive"
    if backend == "fused":
        return "fused"
    if backend in ("fused_smem", "fused_vec4"):
        if max_smem_cols is None:
            return None
        if cols > max_smem_cols:
            return "global"
        if backend == "fused_smem":
            return "smem"
        return "vec4" if cols % 4 == 0 else "vec4_tail"
    return None


def run_cuda(exe: Path, x: np.ndarray, weight: np.ndarray, eps: float) -> CudaRun:
    rows, cols = x.shape
    with tempfile.TemporaryDirectory(prefix="kf_rmsnorm_") as td:
        td_path = Path(td)
        x_path = td_path / "x.bin"
        w_path = td_path / "w.bin"
        o_path = td_path / "o.bin"
        x.astype(np.float32).tofile(x_path)
        weight.astype(np.float32).tofile(w_path)
        cmd = [
            str(exe),
            str(rows),
            str(cols),
            repr(float(eps)),
            str(x_path),
            str(w_path),
            str(o_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"CUDA runner failed (exit {proc.returncode})\n"
                f"cmd: {' '.join(cmd)}\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        out = np.fromfile(o_path, dtype=np.float32)
        if out.size != rows * cols:
            raise RuntimeError(
                f"CUDA output size {out.size} != expected {rows * cols} for shape {(rows, cols)}"
            )
        path, max_cols = parse_runner_stdout(proc.stdout)
        return CudaRun(
            out=out.reshape(rows, cols),
            path=path,
            max_smem_cols=max_cols,
            stdout=proc.stdout,
        )


def assert_path(backend: str, cols: int, run: CudaRun) -> None:
    exp = expected_path(backend, cols, run.max_smem_cols)
    if exp is None:
        return
    if run.path is None:
        raise AssertionError(
            f"{backend} cols={cols}: missing PATH line in stdout:\n{run.stdout}"
        )
    if run.path != exp:
        raise AssertionError(
            f"{backend} cols={cols}: PATH={run.path!r} expected {exp!r} "
            f"(MAX_SMEM_COLS={run.max_smem_cols})"
        )
    # Extra keyed asserts from the pre-Phase-3 checklist.
    keyed = PATH_ASSERT_COLS.get(backend, {}).get(cols)
    if keyed is not None and run.max_smem_cols is not None:
        # Only enforce keyed expectation when it matches device capability math.
        if expected_path(backend, cols, run.max_smem_cols) != keyed:
            # Device has unusual budget; skip rigid keyed check.
            return
        if run.path != keyed:
            raise AssertionError(
                f"{backend} cols={cols}: keyed PATH assert failed: "
                f"got {run.path!r}, want {keyed!r}"
            )


def compare_to_golden(
    label: str,
    case: str,
    cuda_out: np.ndarray,
    ref: np.ndarray,
    cols: int,
) -> None:
    rtol, atol = tolerances_for(cols)
    if not np.allclose(cuda_out, ref, rtol=rtol, atol=atol, equal_nan=True):
        abs_err = np.abs(cuda_out - ref)
        abs_err_safe = np.where(np.isfinite(abs_err), abs_err, -1.0)
        i = int(np.argmax(abs_err_safe))
        rel = abs_err / np.maximum(np.abs(ref), 1e-8)
        raise AssertionError(
            f"{label} mismatch vs NumPy golden for {case}\n"
            f"  max abs err = {float(np.nanmax(abs_err)):.6e} at flat index {i} "
            f"(row={i // cols}, col={i % cols})\n"
            f"  cuda={float(cuda_out.flat[i]):.8g} ref={float(ref.flat[i]):.8g}\n"
            f"  max rel err = {float(np.nanmax(rel)):.6e}\n"
            f"  atol={atol} rtol={rtol}"
        )


def check_random_shape(exe: Path, label: str, rows: int, cols: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, cols), dtype=np.float32)
    weight = make_weight(rng, cols)
    rtol, atol = tolerances_for(cols)

    ref_np = rmsnorm_numpy(x, weight, eps=EPS)
    ref_torch = rmsnorm_torch(
        torch.from_numpy(x.copy()), torch.from_numpy(weight.copy()), eps=EPS
    )
    ref_torch_np = ref_torch.detach().cpu().numpy()

    if not np.allclose(ref_np, ref_torch_np, rtol=rtol, atol=atol):
        max_diff = float(np.max(np.abs(ref_np - ref_torch_np)))
        raise AssertionError(
            f"CPU refs disagree for shape {(rows, cols)}: max|np-torch|={max_diff}"
        )

    run = run_cuda(exe, x, weight, EPS)
    compare_to_golden(label, f"shape=({rows}, {cols})", run.out, ref_np, cols)
    assert_path(label, cols, run)
    path_note = run.path or "?"
    return path_note


def _edge_rows(cols: int, kind: str) -> np.ndarray:
    row = np.zeros((1, cols), dtype=np.float32)
    if kind == "zeros":
        return row
    if kind == "large_mag":
        mag = np.float32(1e18 if cols <= 128 else 1e16)
        row[:] = mag
        return row
    if kind == "small_mag":
        row[:] = np.float32(1e-20)
        return row
    if kind == "mixed_signs":
        signs = np.where(np.arange(cols) % 2 == 0, 1.0, -1.0).astype(np.float32)
        row[0] = signs * np.linspace(0.5, 1.5, cols, dtype=np.float32)
        return row
    if kind == "single_nonzero":
        row[0, cols // 2] = np.float32(7.5)
        return row
    raise ValueError(kind)


# Full edge suite at moderate widths; zeros/small also at wide widths (fp32 accum).
EDGE_FULL_WIDTHS = (128, 1024)
EDGE_FULL_KINDS = ("zeros", "large_mag", "small_mag", "mixed_signs", "single_nonzero")
EDGE_WIDE_WIDTHS = (4096, 8192)
EDGE_WIDE_KINDS = ("zeros", "small_mag")


def check_edge_cases(exe: Path, label: str, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    lines: list[str] = []
    cases: list[tuple[int, str]] = []
    for cols in EDGE_FULL_WIDTHS:
        for kind in EDGE_FULL_KINDS:
            cases.append((cols, kind))
    for cols in EDGE_WIDE_WIDTHS:
        for kind in EDGE_WIDE_KINDS:
            cases.append((cols, kind))

    for cols, kind in cases:
        weight = make_weight(rng, cols)
        x = _edge_rows(cols, kind)
        ref = rmsnorm_numpy(x, weight, eps=EPS)
        run = run_cuda(exe, x, weight, EPS)
        case = f"edge:{kind}:cols={cols}"
        compare_to_golden(label, case, run.out, ref, cols)
        assert_path(label, cols, run)
        lines.append(f"  PASS {case} PATH={run.path}")
    return lines


def run_backend(name: str, exe: Path) -> int:
    if not exe.is_file():
        print(f"FAIL: missing CUDA binary for {name}: {exe}", file=sys.stderr)
        print(f"Build it with: .\\scripts\\build_rmsnorm_{name}.ps1", file=sys.stderr)
        return 1

    print(f"\n=== backend={name} ===")
    print(f"Using CUDA binary: {exe}")
    n_edge = len(EDGE_FULL_WIDTHS) * len(EDGE_FULL_KINDS) + len(EDGE_WIDE_WIDTHS) * len(
        EDGE_WIDE_KINDS
    )
    print(
        f"Shapes: {len(TEST_SHAPES)}  edge_cases={n_edge}  "
        f"eps={EPS}  atol={ATOL}/{WIDE_ATOL}  rtol={RTOL}/{WIDE_RTOL}"
    )
    print("Weight: uniform[0.5, 2.0] (non-trivial)")

    failures = 0
    for i, (rows, cols) in enumerate(TEST_SHAPES):
        tag = f"[{i + 1}/{len(TEST_SHAPES)}] shape=({rows}, {cols})"
        try:
            path = check_random_shape(exe, name, rows, cols, seed=1000 + i)
            print(f"  PASS {tag} PATH={path}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL {tag}", file=sys.stderr)
            print(f"       {exc}", file=sys.stderr)

    try:
        for line in check_edge_cases(exe, name, seed=42):
            print(line)
    except Exception as exc:  # noqa: BLE001
        failures += 1
        print("  FAIL edge-cases", file=sys.stderr)
        print(f"       {exc}", file=sys.stderr)

    n_cases = len(TEST_SHAPES) + n_edge
    if failures:
        print(f"\nFAILED ({name}): {failures} failure group(s)", file=sys.stderr)
        return 1

    print(f"\nOK ({name}): all {n_cases} cases match CPU golden reference")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="RMSNorm correctness harness")
    parser.add_argument(
        "--backend",
        choices=["naive", "fused", "fused_smem", "fused_vec4", "both", "all"],
        default="all",
        help="Which CUDA binary to check (default: all)",
    )
    parser.add_argument(
        "--exe",
        type=Path,
        default=None,
        help="Override binary path (implies a single backend run)",
    )
    args = parser.parse_args()

    if args.exe is not None:
        name = args.backend if args.backend not in ("both", "all") else "custom"
        return run_backend(name, args.exe.resolve())

    if args.backend == "all":
        targets = ALL_BACKENDS
    elif args.backend == "both":
        targets = ["naive", "fused"]
    else:
        targets = [args.backend]

    rc = 0
    for name in targets:
        rc = run_backend(name, BACKENDS[name].resolve()) or rc
    if rc == 0:
        print("\nOK: all requested backends match CPU golden reference")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
