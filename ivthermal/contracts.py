# ivthermal/contracts.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch


class ContractViolation(RuntimeError):
    """Raised when the input/output contract is violated."""


@dataclass(frozen=True)
class ContractSpec:
    version: str = "1.0"
    input_format: str = "NCTHW"
    channels: tuple[str, str, str] = ("T_raw", "T_smooth", "DeltaT")
    allowed_T: tuple[int, int] = (16, 32)
    H: int = 112
    W: int = 112
    output: str = "logits[B,1]"


SPEC = ContractSpec()


def _raise(msg: str) -> None:
    raise ContractViolation(msg)


def assert_contract_config(
    *,
    version: str,
    input_format: str,
    channels: Sequence[str],
    allowed_T: Sequence[int],
    H: int,
    W: int,
    output: str,
) -> None:
    """Checks that config.yaml declares the exact contract we expect."""
    if version != SPEC.version:
        _raise(f"Contract version mismatch: expected {SPEC.version}, got {version}")
    if input_format != SPEC.input_format:
        _raise(f"input_format mismatch: expected {SPEC.input_format}, got {input_format}")
    if tuple(channels) != SPEC.channels:
        _raise(f"channels mismatch: expected {list(SPEC.channels)}, got {list(channels)}")
    if tuple(allowed_T) != SPEC.allowed_T:
        _raise(f"allowed_T mismatch: expected {list(SPEC.allowed_T)}, got {list(allowed_T)}")
    if H != SPEC.H or W != SPEC.W:
        _raise(f"Spatial mismatch: expected H=W={SPEC.H}, got H={H}, W={W}")
    if output != SPEC.output:
        _raise(f"Output contract mismatch: expected '{SPEC.output}', got '{output}'")


def assert_input_tensor(
    x: torch.Tensor,
    *,
    expected_T: Optional[int] = None,
    require_contiguous: bool = False,
    require_finite: bool = True,
) -> None:
    """
    Strict input contract:
      - x shape: [B, 3, T, 112, 112]
      - dtype: float32
      - T ∈ {16, 32} and optionally equals expected_T
      - channel order is fixed by contract (enforced via config, not tensor)
    """
    if not isinstance(x, torch.Tensor):
        _raise(f"Input must be a torch.Tensor, got {type(x)}")

    if x.dtype != torch.float32:
        _raise(f"Input dtype must be torch.float32, got {x.dtype}")

    if x.ndim != 5:
        _raise(f"Input must be 5D NCTHW [B,3,T,H,W], got shape {tuple(x.shape)}")

    B, C, T, H, W = x.shape

    if B < 1:
        _raise(f"Batch size B must be >= 1, got B={B}")

    if C != 3:
        _raise(f"Channel dim must be C=3, got C={C}")

    if T not in SPEC.allowed_T:
        _raise(f"T must be one of {list(SPEC.allowed_T)}, got T={T}")

    if expected_T is not None and T != expected_T:
        _raise(f"T mismatch with config: expected T={expected_T}, got T={T}")

    if H != SPEC.H or W != SPEC.W:
        _raise(f"H and W must be {SPEC.H}x{SPEC.W}, got {H}x{W}")

    if require_contiguous and not x.is_contiguous():
        _raise("Input tensor must be contiguous (require_contiguous=True)")

    if require_finite:
        # NaN/Inf can silently break training; fail fast
        if not torch.isfinite(x).all().item():
            _raise("Input contains NaN or Inf values (require_finite=True)")


def assert_output_logits(
    logits: torch.Tensor,
    *,
    batch_size: Optional[int] = None,
    require_finite: bool = True,
) -> None:
    """
    Strict output contract:
      - logits shape: [B, 1]
      - dtype: float32
    """
    if not isinstance(logits, torch.Tensor):
        _raise(f"Output logits must be a torch.Tensor, got {type(logits)}")

    if logits.dtype != torch.float32:
        _raise(f"Logits dtype must be torch.float32, got {logits.dtype}")

    if logits.ndim != 2:
        _raise(f"Logits must be 2D [B,1], got shape {tuple(logits.shape)}")

    B, C = logits.shape
    if C != 1:
        _raise(f"Logits second dim must be 1, got {C}")

    if batch_size is not None and B != batch_size:
        _raise(f"Logits batch size mismatch: expected B={batch_size}, got B={B}")

    if require_finite:
        if not torch.isfinite(logits).all().item():
            _raise("Logits contain NaN or Inf values (require_finite=True)")
