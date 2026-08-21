"""vLLM OpenAI-compatible HTTP backend with streaming TTFT / TPOT.

Server lifecycle is external (Colab / scripts/run_phase6_vllm_server.sh).
This client fills the Phase 5+ streaming schema — first real engine path.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from bench.schema import GenerateRequest, GenerateResult, tpot_from_e2e


class VllmHttpBackend:
    name = "vllm"

    def __init__(
        self,
        model_id: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 600.0,
    ) -> None:
        self.model_id = model_id
        self.base_url = (
            base_url
            or os.environ.get("KERNELFUSE_VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("KERNELFUSE_VLLM_API_KEY", "EMPTY")
        self.timeout_s = timeout_s

    def start(self) -> None:
        # Probe /models so harness fails fast if the server is down.
        url = f"{self.base_url}/models"
        req = Request(url, headers=self._headers())
        try:
            with urlopen(req, timeout=min(30.0, self.timeout_s)) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            data = json.loads(body)
            ids = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
            if ids and self.model_id not in ids:
                # vLLM often reports the served model id; accept first if single.
                if len(ids) == 1:
                    self.model_id = str(ids[0])
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"vLLM not reachable at {self.base_url} ({e}). "
                "Start the server first (see scripts/run_phase6_vllm_server.sh)."
            ) from e

    def stop(self) -> None:
        return None

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

    def generate(self, req: GenerateRequest) -> GenerateResult:
        t0 = time.perf_counter()
        payload: dict[str, Any] = {
            "model": self.model_id,
            "prompt": req.prompt,
            "max_tokens": int(req.max_new_tokens),
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        body = json.dumps(payload).encode("utf-8")
        url = f"{self.base_url}/completions"
        http_req = Request(url, data=body, headers=self._headers(), method="POST")

        ttft_ms: float | None = None
        text_parts: list[str] = []
        completion_tokens: int | None = None

        try:
            with urlopen(http_req, timeout=self.timeout_s) as resp:
                # Iterate SSE lines
                while True:
                    raw = resp.readline()
                    if not raw:
                        break
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    usage = chunk.get("usage")
                    if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                        completion_tokens = int(usage["completion_tokens"])
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    text = choices[0].get("text") or ""
                    if text:
                        now = time.perf_counter()
                        if ttft_ms is None:
                            ttft_ms = (now - t0) * 1000.0
                        text_parts.append(text)

            latency_ms = (time.perf_counter() - t0) * 1000.0
            joined = "".join(text_parts)
            # Prefer server usage; else rough whitespace/token proxy for dry-run accounting.
            if completion_tokens is not None:
                out_tok = completion_tokens
            else:
                out_tok = max(1, len(joined.split())) if joined else 0
            if ttft_ms is None and out_tok > 0:
                ttft_ms = latency_ms
            tpot = (
                tpot_from_e2e(ttft_ms, latency_ms, out_tok)
                if ttft_ms is not None
                else None
            )
            return GenerateResult(
                request_id=req.request_id,
                latency_ms=latency_ms,
                output_tokens=out_tok,
                ok=True,
                ttft_ms=ttft_ms,
                tpot_ms=tpot,
            )
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            err = str(e)
            if isinstance(e, HTTPError):
                try:
                    err = e.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    pass
            return GenerateResult(
                request_id=req.request_id,
                latency_ms=latency_ms,
                output_tokens=0,
                ok=False,
                error=err,
            )
