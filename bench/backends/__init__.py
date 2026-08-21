"""Backend registry."""

from __future__ import annotations

from bench.backends.hf_local import HfLocalBackend
from bench.backends.mock import MockBackend


def make_backend(name: str, model_id: str):
    if name == "mock":
        return MockBackend()
    if name == "hf_local":
        return HfLocalBackend(model_id=model_id)
    raise ValueError(
        f"unknown backend {name!r} — Phase 5 supports mock|hf_local; "
        "vllm/sglang/trtllm land in Phase 6+"
    )
