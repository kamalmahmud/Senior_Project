# scripts/sanity_check_contract.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Ensure project root is on PYTHONPATH when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ivthermal.config_schema import load_config
from ivthermal.contracts import (
    assert_contract_config,
    assert_input_tensor,
    assert_output_logits,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sanity check the I/O contract.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--batch", type=int, default=2, help="Dummy batch size B")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Assert the contract block matches the expected immutable spec
    assert_contract_config(
        version=cfg.contract.version,
        input_format=cfg.contract.input_format,
        channels=cfg.contract.channels,
        allowed_T=cfg.contract.allowed_T,
        H=cfg.contract.H,
        W=cfg.contract.W,
        output=cfg.contract.output,
    )

    B = args.batch
    T = cfg.clip.T
    H = cfg.contract.H
    W = cfg.contract.W

    x = torch.randn(B, 3, T, H, W, dtype=torch.float32)
    assert_input_tensor(x, expected_T=T)

    # Dummy logits (what your model must return)
    logits = torch.randn(B, 1, dtype=torch.float32)
    assert_output_logits(logits, batch_size=B)

    print("✅ Contract sanity check passed.")
    print(f"   Input : {tuple(x.shape)} dtype={x.dtype}")
    print(f"   Output: {tuple(logits.shape)} dtype={logits.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
