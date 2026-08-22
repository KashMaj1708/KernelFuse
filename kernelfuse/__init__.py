"""KernelFuse Phase 8 — custom CUDA ops for serving integration."""

from __future__ import annotations

try:
    from kernelfuse._C import fused_add_rms_norm  # noqa: F401
except ImportError:
    fused_add_rms_norm = None  # type: ignore[misc, assignment]

__all__ = ["fused_add_rms_norm"]
