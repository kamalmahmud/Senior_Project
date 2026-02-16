from __future__ import annotations

from typing import List

import torch


def validate_T(T: int, allowed=(16, 32)) -> None:
    if T not in allowed:
        raise ValueError(f"T must be one of {list(allowed)}, got {T}")


def ensure_length_cthw(x: torch.Tensor, T: int) -> torch.Tensor:
    """
    Ensures a CTHW tensor has exactly T frames.
    - If longer: center-crop in time
    - If shorter: replicate-pad last frame
    """
    if x.ndim != 4:
        raise ValueError(f"Expected CTHW, got shape {tuple(x.shape)}")
    C, Tin, H, W = x.shape
    validate_T(T)

    if Tin == T:
        return x

    if Tin > T:
        start = (Tin - T) // 2
        return x[:, start:start + T, :, :]

    # Tin < T: replicate pad last frame
    pad = T - Tin
    last = x[:, -1:, :, :].repeat(1, pad, 1, 1)
    return torch.cat([x, last], dim=1)
