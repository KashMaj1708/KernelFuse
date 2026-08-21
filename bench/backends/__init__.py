"""Backend registry."""

from __future__ import annotations

import os

from bench.backends.hf_local import HfLocalBackend
from bench.backends.mock import MockBackend
from bench.backends.vllm_http import VllmHttpBackend


def make_backend(name: str, model_id: str):
    if name == "mock":
        return MockBackend()
    if name == "hf_local":
        return HfLocalBackend(model_id=model_id)
    if name == "vllm":
        return VllmHttpBackend(model_id=model_id)
    if name == "sglang":
        return VllmHttpBackend(
            model_id=model_id,
            base_url=os.environ.get(
                "KERNELFUSE_SGLANG_BASE_URL", "http://127.0.0.1:30000/v1"
            ),
        )
    if name == "trtllm":
        return VllmHttpBackend(
            model_id=model_id,
            base_url=os.environ.get(
                "KERNELFUSE_TRTLLM_BASE_URL", "http://127.0.0.1:8000/v1"
            ),
        )
    raise ValueError(
        f"unknown backend {name!r} — supported: mock|hf_local|vllm|sglang|trtllm"
    )
