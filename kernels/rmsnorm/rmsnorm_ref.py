"""Golden RMSNorm reference (CPU). Source of truth for kernel correctness."""

from __future__ import annotations

import numpy as np
import torch


def rmsnorm_numpy(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMSNorm over the last axis. x: [..., H], weight: [H]."""
    x = np.asarray(x, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    if weight.ndim != 1 or weight.shape[0] != x.shape[-1]:
        raise ValueError(f"weight shape {weight.shape} incompatible with x {x.shape}")
    # Cast sum to float64 for a slightly stabler mean, then back — still a plain reference.
    sq_mean = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    inv_rms = (1.0 / np.sqrt(sq_mean + eps)).astype(np.float32)
    return (x * inv_rms * weight).astype(np.float32)


def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Same formula on PyTorch CPU tensors."""
    x = x.to(dtype=torch.float32, device="cpu")
    weight = weight.to(dtype=torch.float32, device="cpu")
    if weight.ndim != 1 or weight.shape[0] != x.shape[-1]:
        raise ValueError(f"weight shape {tuple(weight.shape)} incompatible with x {tuple(x.shape)}")
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    return x * inv_rms * weight


def rmsnorm(x, weight, eps: float = 1e-6):
    """Dispatch: torch tensors stay torch; otherwise NumPy."""
    if isinstance(x, torch.Tensor) or isinstance(weight, torch.Tensor):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        if not isinstance(weight, torch.Tensor):
            weight = torch.as_tensor(weight, dtype=torch.float32)
        return rmsnorm_torch(x, weight, eps=eps)
    return rmsnorm_numpy(np.asarray(x), np.asarray(weight), eps=eps)
