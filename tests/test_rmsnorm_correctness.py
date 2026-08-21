"""
RMSNorm correctness harness (Phases 1–2).

Compares a CUDA binary (naive or fused) against the NumPy/PyTorch golden reference.
Same shapes and tolerances for every kernel version — fails loudly on mismatch.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kernels" / "rmsnorm"))

from rmsnorm_ref import rmsnorm_numpy, rmsnorm_torch  # noqa: E402

TEST_SHAPES = [
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

EPS = 1e-6
ATOL = 1e-5
RTOL = 1e-4

BACKENDS = {
    "naive": ROOT / "kernels" / "rmsnorm" / "rmsnorm_naive.exe",
    "fused": ROOT / "kernels" / "rmsnorm" / "rmsnorm_fused.exe",
}


def run_cuda(exe: Path, x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
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
        return out.reshape(rows, cols)


def check_shape(exe: Path, label: str, rows: int, cols: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, cols), dtype=np.float32)
    weight = rng.standard_normal((cols,), dtype=np.float32)

    ref_np = rmsnorm_numpy(x, weight, eps=EPS)
    ref_torch = rmsnorm_torch(
        torch.from_numpy(x.copy()), torch.from_numpy(weight.copy()), eps=EPS
    )
    ref_torch_np = ref_torch.detach().cpu().numpy()

    if not np.allclose(ref_np, ref_torch_np, rtol=RTOL, atol=ATOL):
        max_diff = float(np.max(np.abs(ref_np - ref_torch_np)))
        raise AssertionError(
            f"CPU refs disagree for shape {(rows, cols)}: max|np-torch|={max_diff}"
        )

    cuda_out = run_cuda(exe, x, weight, EPS)
    if not np.allclose(cuda_out, ref_np, rtol=RTOL, atol=ATOL):
        abs_err = np.abs(cuda_out - ref_np)
        rel = abs_err / np.maximum(np.abs(ref_np), 1e-8)
        i = int(np.argmax(abs_err))
        raise AssertionError(
            f"{label} mismatch vs NumPy golden for shape {(rows, cols)}\n"
            f"  max abs err = {float(abs_err.max()):.6e} at flat index {i} "
            f"(row={i // cols}, col={i % cols})\n"
            f"  cuda={float(cuda_out.flat[i]):.8g} ref={float(ref_np.flat[i]):.8g}\n"
            f"  max rel err = {float(rel.max()):.6e}\n"
            f"  atol={ATOL} rtol={RTOL}"
        )


def run_backend(name: str, exe: Path) -> int:
    if not exe.is_file():
        print(f"FAIL: missing CUDA binary for {name}: {exe}", file=sys.stderr)
        print(f"Build it with: .\\scripts\\build_rmsnorm_{name}.ps1", file=sys.stderr)
        return 1

    print(f"\n=== backend={name} ===")
    print(f"Using CUDA binary: {exe}")
    print(f"Shapes: {len(TEST_SHAPES)}  eps={EPS}  atol={ATOL}  rtol={RTOL}")

    failures = 0
    for i, (rows, cols) in enumerate(TEST_SHAPES):
        label = f"[{i + 1}/{len(TEST_SHAPES)}] shape=({rows}, {cols})"
        try:
            check_shape(exe, name, rows, cols, seed=1000 + i)
            print(f"  PASS {label}")
        except Exception as exc:  # noqa: BLE001 — harness must surface any failure loudly
            failures += 1
            print(f"  FAIL {label}", file=sys.stderr)
            print(f"       {exc}", file=sys.stderr)

    if failures:
        print(
            f"\nFAILED ({name}): {failures}/{len(TEST_SHAPES)} shapes mismatched",
            file=sys.stderr,
        )
        return 1

    print(f"\nOK ({name}): all {len(TEST_SHAPES)} shapes match CPU golden reference")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="RMSNorm correctness harness")
    parser.add_argument(
        "--backend",
        choices=["naive", "fused", "both"],
        default="both",
        help="Which CUDA binary to check (default: both)",
    )
    parser.add_argument(
        "--exe",
        type=Path,
        default=None,
        help="Override binary path (implies a single backend run)",
    )
    args = parser.parse_args()

    if args.exe is not None:
        name = args.backend if args.backend != "both" else "custom"
        return run_backend(name, args.exe.resolve())

    targets = ["naive", "fused"] if args.backend == "both" else [args.backend]
    rc = 0
    for name in targets:
        rc = run_backend(name, BACKENDS[name].resolve()) or rc
    if rc == 0:
        print("\nOK: all requested backends match CPU golden reference")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
