"""HuggingFace transformers local generate — toy model for Tier A (4 GB).

Uses TextIteratorStreamer so TTFT / TPOT are real streaming metrics, matching
the contract Phase 6 engines will satisfy.
"""

from __future__ import annotations

import threading
import time

from bench.schema import GenerateRequest, GenerateResult, tpot_from_e2e


class HfLocalBackend:
    name = "hf_local"

    def __init__(
        self,
        model_id: str = "hf-internal-testing/tiny-random-gpt2",
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self._device = device
        self._tok = None
        self._model = None
        self._max_pos = 512
        self._lock = threading.Lock()

    def start(self) -> None:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = AutoConfig.from_pretrained(self.model_id)
        self._max_pos = int(
            getattr(cfg, "max_position_embeddings", None)
            or getattr(cfg, "n_positions", None)
            or 512
        )
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                use_safetensors=True,
                dtype=torch.float32,
            )
        except Exception as first:
            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    dtype=torch.float32,
                )
            except Exception as second:
                raise RuntimeError(
                    f"failed to load {self.model_id}. Prefer a safetensors model "
                    f"(torch {torch.__version__} + current transformers block .bin). "
                    f"safetensors err: {first}; fallback err: {second}"
                ) from second
        self._model.to(self._device)
        self._model.eval()

    def stop(self) -> None:
        self._model = None
        self._tok = None

    def generate(self, req: GenerateRequest) -> GenerateResult:
        import torch
        from transformers import TextIteratorStreamer

        assert self._model is not None and self._tok is not None
        t0 = time.perf_counter()
        try:
            # Serialize CUDA work — generate + streamer thread share one device.
            with self._lock:
                max_input = max(8, self._max_pos - int(req.max_new_tokens) - 2)
                inputs = self._tok(
                    req.prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_input,
                )
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                streamer = TextIteratorStreamer(
                    self._tok,
                    skip_prompt=True,
                    skip_special_tokens=True,
                )
                gen_kwargs = dict(
                    **inputs,
                    max_new_tokens=req.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tok.pad_token_id,
                    streamer=streamer,
                )

                err_box: list[BaseException] = []

                def _run() -> None:
                    try:
                        with torch.inference_mode():
                            self._model.generate(**gen_kwargs)
                        if self._device.startswith("cuda"):
                            torch.cuda.synchronize()
                    except BaseException as e:  # noqa: BLE001
                        err_box.append(e)

                th = threading.Thread(target=_run, daemon=True)
                th.start()

                ttft_ms: float | None = None
                intervals: list[float] = []
                pieces: list[str] = []
                t_prev = t0
                for piece in streamer:
                    now = time.perf_counter()
                    if not pieces:
                        ttft_ms = (now - t0) * 1000.0
                    else:
                        intervals.append((now - t_prev) * 1000.0)
                    t_prev = now
                    pieces.append(piece)

                th.join()
                if err_box:
                    raise err_box[0]

            latency_ms = (time.perf_counter() - t0) * 1000.0
            text = "".join(pieces)
            out_tok = (
                len(self._tok.encode(text, add_special_tokens=False)) if text else 0
            )
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
                token_intervals_ms=tuple(intervals),
            )
        except Exception as e:  # noqa: BLE001
            if self._device and str(self._device).startswith("cuda"):
                try:
                    import torch

                    torch.cuda.synchronize()
                except Exception:
                    pass
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return GenerateResult(
                request_id=req.request_id,
                latency_ms=latency_ms,
                output_tokens=0,
                ok=False,
                error=str(e),
            )
