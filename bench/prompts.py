"""Token-accurate prompt construction for the serving harness.

Word-count proxies overshoot real tokenizers (Phase 6 in=512 → HTTP 400 on
TinyLlama max_model_len). Prefer HF tokenizer length; fall back to words for mock.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=8)
def _tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    return tok


def make_prompt(
    n_tokens: int,
    seed: int,
    idx: int,
    *,
    model_id: str | None = None,
    max_prompt_tokens: int | None = None,
) -> str:
    """Return a prompt whose *tokenized* length is ~n_tokens (capped if set)."""
    n = max(1, int(n_tokens))
    if max_prompt_tokens is not None:
        n = min(n, max(1, int(max_prompt_tokens)))

    if not model_id or model_id in ("mock-toy", "mock"):
        word = f"w{seed}_{idx % 97}"
        return " ".join([word] * n)

    try:
        tok = _tokenizer(model_id)
    except Exception:
        word = f"w{seed}_{idx % 97}"
        return " ".join([word] * n)

    pieces: list[str] = []
    ids: list[int] = []
    k = 0
    # Grow then truncate — Llama-family often ≈1 id per short alphanumeric piece.
    while len(ids) < n and k < n * 4:
        pieces.append(f"{seed}_{idx}_{k}")
        ids = tok.encode(" ".join(pieces), add_special_tokens=True)
        k += 1
    ids = ids[:n]
    return tok.decode(ids, skip_special_tokens=True)


def prompt_token_len(text: str, model_id: str | None) -> int:
    if not model_id or model_id in ("mock-toy", "mock"):
        return len(text.split())
    try:
        tok = _tokenizer(model_id)
        return len(tok.encode(text, add_special_tokens=True))
    except Exception:
        return len(text.split())
