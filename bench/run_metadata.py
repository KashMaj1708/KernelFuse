"""Capture Phase 6 run provenance (version / dtype / attention) next to CSVs."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _safe_import_version(mod: str) -> str | None:
    try:
        m = __import__(mod)
        return getattr(m, "__version__", None)
    except Exception:
        return None


def collect_run_metadata(
    *,
    matrix_version: str,
    model_id: str,
    dtype: str = "float16",
    enforce_eager: bool = True,
    gpu_memory_utilization: float = 0.75,
    attention_backend: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build metadata dict. attention_backend should be set from vLLM logs when known."""
    gpu_name = None
    compute_cap = None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
        if out:
            parts = [p.strip() for p in out.split(",")]
            if len(parts) >= 1:
                gpu_name = parts[0]
            if len(parts) >= 2:
                compute_cap = parts[1]
    except Exception:
        pass

    meta: dict[str, Any] = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_version": matrix_version,
        "model_id": model_id,
        "dtype": dtype,
        "enforce_eager": enforce_eager,
        "gpu_memory_utilization": gpu_memory_utilization,
        "attention_backend": attention_backend or "unknown_expect_xformers_on_sm75",
        "attention_note": (
            "FlashAttention needs sm_80+. T4/sm_75 uses XFormers — "
            "Phase 6 vs Phase 7 numbers are not cross-tier comparable."
        ),
        "vllm_version": _safe_import_version("vllm"),
        "torch_version": _safe_import_version("torch"),
        "transformers_version": _safe_import_version("transformers"),
        "gpu_name": gpu_name,
        "compute_capability": compute_cap,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if extra:
        meta.update(extra)
    return meta


def write_run_metadata(path: Path, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
