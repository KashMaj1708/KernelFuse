"""Unit tests for VllmHttpBackend streaming parse (mocked HTTP, no GPU)."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.backends.vllm_http import VllmHttpBackend
from bench.schema import GenerateRequest


def _sse(*payloads: dict) -> bytes:
    lines = []
    for p in payloads:
        lines.append(f"data: {json.dumps(p)}\n\n".encode())
    lines.append(b"data: [DONE]\n\n")
    return b"".join(lines)


class _Resp:
    def __init__(self, raw: bytes) -> None:
        self._buf = io.BytesIO(raw)

    def readline(self) -> bytes:
        return self._buf.readline()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_streaming_ttft_tpot():
    chunks = _sse(
        {"choices": [{"text": "Hello"}]},
        {"choices": [{"text": " world"}]},
        {
            "choices": [{"text": "!"}],
            "usage": {"completion_tokens": 3, "prompt_tokens": 2, "total_tokens": 5},
        },
    )
    backend = VllmHttpBackend(model_id="toy", base_url="http://127.0.0.1:9/v1")
    with patch("bench.backends.vllm_http.urlopen", return_value=_Resp(chunks)):
        r = backend.generate(
            GenerateRequest(prompt="hi", max_new_tokens=8, request_id="t1")
        )
    assert r.ok
    assert r.output_tokens == 3
    assert r.ttft_ms is not None and r.ttft_ms >= 0
    assert r.tpot_ms is not None
    assert r.latency_ms >= r.ttft_ms


def test_error_path():
    from urllib.error import URLError

    backend = VllmHttpBackend(model_id="toy", base_url="http://127.0.0.1:9/v1")
    with patch("bench.backends.vllm_http.urlopen", side_effect=URLError("down")):
        r = backend.generate(
            GenerateRequest(prompt="hi", max_new_tokens=4, request_id="t2")
        )
    assert not r.ok
    assert r.output_tokens == 0
    assert r.error


if __name__ == "__main__":
    test_streaming_ttft_tpot()
    test_error_path()
    print("ok")
