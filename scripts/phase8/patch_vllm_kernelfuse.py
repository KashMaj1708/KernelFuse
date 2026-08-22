#!/usr/bin/env python3
"""Patch vLLM 0.8.5 layernorm.py to route fused_add_rms_norm through kernelfuse when enabled."""

from __future__ import annotations

import os
import sys
from pathlib import Path


MARKER = "# KERNELFUSE_PATCH"


def find_layernorm() -> Path:
    import vllm  # noqa: WPS433

    p = Path(vllm.__file__).resolve().parent / "model_executor" / "layers" / "layernorm.py"
    if not p.is_file():
        raise SystemExit(f"layernorm.py not found at {p}")
    return p


def patch(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"already patched: {path}")
        return

    needles = [
        '''def fused_add_rms_norm(
        x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
        variance_epsilon: float) -> Tuple[torch.Tensor, torch.Tensor]:
    from vllm import _custom_ops as ops
    ops.fused_add_rms_norm(
        x,
        residual,
        weight,
        variance_epsilon,
    )
    return x, residual''',
        '''def fused_add_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
    variance_epsilon: float) -> Tuple[torch.Tensor, torch.Tensor]:
    from vllm import _custom_ops as ops
    ops.fused_add_rms_norm(
        x,
        residual,
        weight,
        variance_epsilon,
    )
    return x, residual''',
    ]

    replacement_tpl = '''def fused_add_rms_norm(
{param_indent}x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
{param_indent}variance_epsilon: float) -> Tuple[torch.Tensor, torch.Tensor]:
    {MARKER}
    if os.environ.get("KERNELFUSE_FUSED_ADD_RMSNORM", "") == "1":
        import kernelfuse
        kernelfuse.fused_add_rms_norm(x, residual, weight, variance_epsilon)
        return x, residual
    from vllm import _custom_ops as ops
    ops.fused_add_rms_norm(
        x,
        residual,
        weight,
        variance_epsilon,
    )
    return x, residual'''

    matched = None
    for needle in needles:
        if needle in text:
            matched = needle
            break
    if matched is None:
        raise SystemExit("fused_add_rms_norm block not found — vLLM version mismatch?")

    param_indent = "        " if matched.startswith(
        "def fused_add_rms_norm(\n        x:"
    ) else "    "
    replacement = replacement_tpl.format(param_indent=param_indent, MARKER=MARKER)

    if "import os\n" not in text and "import os\r\n" not in text:
        text = text.replace("import torch\n", "import os\nimport torch\n", 1)

    path.write_text(text.replace(matched, replacement, 1), encoding="utf-8")
    print(f"patched: {path}")


def main() -> int:
    if os.environ.get("KERNELFUSE_UNPATCH") == "1":
        print("KERNELFUSE_UNPATCH=1 — skip")
        return 0
    patch(find_layernorm())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
