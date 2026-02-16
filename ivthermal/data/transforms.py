from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


class TransformError(RuntimeError):
    pass


def _check_cthw(t: torch.Tensor) -> Tuple[int, int, int, int]:
    """
    Validate a CTHW tensor (C=1 expected for thermal raw channel) or (C>=1 in general).
    Returns (C, T, H, W).
    """
    if not isinstance(t, torch.Tensor):
        raise TransformError(f"Expected torch.Tensor, got {type(t)}")
    if t.ndim != 4:
        raise TransformError(f"Expected 4D CTHW tensor, got shape {tuple(t.shape)}")
    C, T, H, W = t.shape
    if T < 1 or H < 1 or W < 1:
        raise TransformError(f"Invalid spatial/temporal dims: {tuple(t.shape)}")
    return C, T, H, W


def temporal_smooth_cthw(
    t_raw: torch.Tensor,
    *,
    window: int = 5,
) -> torch.Tensor:
    """
    Deterministic temporal smoothing: centered moving average over time axis.
    - Input: t_raw in CTHW (usually C=1)
    - Padding: replicate at temporal edges
    - window must be odd >= 1
    - Output: same shape CTHW
    """
    C, T, H, W = _check_cthw(t_raw)

    if window < 1 or window % 2 == 0:
        raise TransformError(f"window must be odd and >=1, got {window}")
    if window == 1:
        return t_raw.clone()

    # Replicate pad along time dimension only.
    # We'll reshape to (C*H*W, 1, T) to use conv1d for a stable mean filter.
    pad = window // 2

    x = t_raw.permute(0, 2, 3, 1).contiguous()  # C,H,W,T
    x = x.view(C * H * W, 1, T)                 # N,1,T

    # replicate padding: pad left/right with edge values
    x_pad = torch.nn.functional.pad(x, (pad, pad), mode="replicate")  # N,1,T+2*pad

    # mean filter kernel
    kernel = torch.ones(1, 1, window, device=t_raw.device, dtype=t_raw.dtype) / float(window)

    # conv1d: output length = T
    y = torch.nn.functional.conv1d(x_pad, kernel)  # (N,1,T)
    y = y.view(C, H, W, T).permute(0, 3, 1, 2).contiguous()  # C,T,H,W

    return y


def deltaT_cthw(
    t_raw: torch.Tensor,
    *,
    baseline: str = "first_frame",
) -> torch.Tensor:
    """
    Deterministic baseline subtraction to create DeltaT:
      DeltaT[t] = T_raw[t] - T_raw[0]
    - Input: t_raw in CTHW
    - Output: same shape CTHW
    """
    _check_cthw(t_raw)
    if baseline != "first_frame":
        raise TransformError(f"Unsupported baseline='{baseline}' (v1.0 supports only 'first_frame')")

    base = t_raw[:, 0:1, :, :]  # C,1,H,W
    return t_raw - base


@dataclass(frozen=True)
class ThermalThreeChannelBuilder:
    """
    Builds the contract channels in fixed order:
      [T_raw, T_smooth, DeltaT]
    Output shape:
      - If input is CTHW (C=1): returns (3, T, H, W)
      - If input is NCTHW (C=1): returns (N, 3, T, H, W)
    """
    smooth_window: int = 5
    deltaT_baseline: str = "first_frame"

    def __call__(self, t_raw: torch.Tensor) -> torch.Tensor:
        if not isinstance(t_raw, torch.Tensor):
            raise TransformError(f"Expected torch.Tensor, got {type(t_raw)}")

        if t_raw.ndim == 4:
            # CTHW
            C, T, H, W = _check_cthw(t_raw)
            if C != 1:
                raise TransformError(f"Expected C=1 for thermal raw input, got C={C}")

            t_smooth = temporal_smooth_cthw(t_raw, window=self.smooth_window)
            dT = deltaT_cthw(t_raw, baseline=self.deltaT_baseline)

            # Concatenate along channel dim -> (3,T,H,W)
            out = torch.cat([t_raw, t_smooth, dT], dim=0)
            return out

        if t_raw.ndim == 5:
            # NCTHW
            N, C, T, H, W = t_raw.shape
            if C != 1:
                raise TransformError(f"Expected C=1 for thermal raw input, got C={C}")

            outs = []
            for i in range(N):
                out_i = self(t_raw[i])  # (3,T,H,W)
                outs.append(out_i)
            return torch.stack(outs, dim=0)  # (N,3,T,H,W)

        raise TransformError(f"Expected input as CTHW or NCTHW, got shape {tuple(t_raw.shape)}")


def normalize_ncthw(
    x: torch.Tensor,
    *,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> torch.Tensor:
    """
    Per-channel affine normalization for NCTHW input (B,3,T,H,W).
    """
    if x.ndim != 5:
        raise TransformError(f"Expected NCTHW (B,3,T,H,W), got shape {tuple(x.shape)}")
    B, C, T, H, W = x.shape
    if C != 3:
        raise TransformError(f"Expected C=3, got C={C}")
    if any(s == 0.0 for s in std):
        raise TransformError("std entries must be non-zero")

    device = x.device
    dtype = x.dtype
    m = torch.tensor(mean, device=device, dtype=dtype).view(1, 3, 1, 1, 1)
    s = torch.tensor(std, device=device, dtype=dtype).view(1, 3, 1, 1, 1)
    return (x - m) / s


def clamp_ncthw(x: torch.Tensor, *, min_val: float, max_val: float) -> torch.Tensor:
    if min_val >= max_val:
        raise TransformError("min_val must be < max_val")
    return torch.clamp(x, min=min_val, max=max_val)
