"""Reference decode attention: one query token vs KV cache (fp32 acc)."""

from __future__ import annotations

import torch


def decode_attn_ref(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """q [D], k_cache [S,D], v_cache [S,D] -> out [D]"""
    d = q.shape[-1]
    if scale is None:
        scale = d**-0.5
    scores = torch.matmul(k_cache.float(), q.float()) * scale
    attn = torch.softmax(scores, dim=0)
    out = torch.matmul(attn, v_cache.float())
    return out.to(q.dtype)
