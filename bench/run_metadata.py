"""Capture Phase 6/7 run provenance (version / dtype / attention) next to CSVs."""

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


def attention_note_for(backend: str | None, compute_capability: str | None) -> str:
    """Phase-aware note — do not carry T4/XFormers text onto A100 artifacts."""
    cc = (compute_capability or "").strip()
    try:
        cc_f = float(cc) if cc else 0.0
    except ValueError:
        cc_f = 0.0
    attn = (backend or "").lower()
    if cc_f >= 8.0 or "flash" in attn:
        return (
            "sm_80+ path: expect FlashAttention / FlashInfer (not Phase 6 T4 "
            "TRITON_ATTN/XFormers). Absolute numbers are not cross-tier comparable "
            "to Phase 6 eager/Turing runs."
        )
    if "triton" in attn or "xformers" in attn:
        return (
            "Turing/sm_75 path: FlashAttention unavailable; engine fell back "
            "(XFormers or Triton). Document as a limitation if comparing to Ampere+."
        )
    return (
        "Record the attention backend the engine actually logged. "
        "Phase 6 (T4) and Phase 7 (A100) are not cross-tier comparable."
    )


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
        "attention_backend": attention_backend or "unknown",
        "attention_note": attention_note_for(attention_backend, compute_cap),
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


def parse_vllm_attention_backend(log_text: str) -> str:
    """Parse vLLM server log for the selected attention backend."""
    import re

    m = re.search(
        r"Using\s+(\w+)\s+attention backend",
        log_text,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()
    if re.search(r"Using FlashAttention version", log_text, re.I):
        return "FLASH_ATTN"
    if re.search(r"\bxformers\b", log_text, re.I):
        return "XFORMERS"
    if re.search(r"TRITON_ATTN", log_text):
        return "TRITON_ATTN"
    return "unknown"
