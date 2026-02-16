from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader

from ivthermal.contracts import assert_input_tensor
from ivthermal.data.transforms import ThermalThreeChannelBuilder
from ivthermal.data.clip_sampling import validate_T


def _normalize_cthw(x: torch.Tensor, mean: list[float], std: list[float]) -> torch.Tensor:
    # x: (3,T,H,W)
    if x.ndim != 4 or x.shape[0] != 3:
        raise ValueError(f"Expected (3,T,H,W), got {tuple(x.shape)}")
    m = torch.tensor(mean, dtype=x.dtype, device=x.device).view(3, 1, 1, 1)
    s = torch.tensor(std, dtype=x.dtype, device=x.device).view(3, 1, 1, 1)
    return (x - m) / s


@dataclass(frozen=True)
class DummyThermalDatasetConfig:
    num_samples: int = 512
    T: int = 16
    H: int = 112
    W: int = 112
    seed: int = 123
    smooth_window: int = 5
    deltaT_baseline: str = "first_frame"
    normalize_enabled: bool = True
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: tuple[float, float, float] = (1.0, 1.0, 1.0)
    clamp_enabled: bool = False
    clamp_min: float = -10.0
    clamp_max: float = 10.0


class DummyThermalDataset(Dataset):
    """
    Produces deterministic dummy samples with the final contract tensor:
      x: (3, T, H, W) float32
      y: (1,) float32 in {0,1}  (collates to (B,1))
    """
    def __init__(self, cfg: DummyThermalDatasetConfig):
        super().__init__()
        validate_T(cfg.T)
        self.cfg = cfg
        self.builder = ThermalThreeChannelBuilder(
            smooth_window=cfg.smooth_window,
            deltaT_baseline=cfg.deltaT_baseline,
        )

    def __len__(self) -> int:
        return self.cfg.num_samples

    def _make_raw_clip(self, idx: int) -> torch.Tensor:
        """
        Returns t_raw: (1, T, H, W) float32.
        Deterministic by (seed + idx) regardless of DataLoader workers.
        """
        g = torch.Generator(device="cpu")
        g.manual_seed(self.cfg.seed + int(idx))

        T, H, W = self.cfg.T, self.cfg.H, self.cfg.W

        # base thermal field
        base = torch.randn(1, 1, H, W, generator=g) * 2.0  # (1,1,H,W)

        # per-pixel drift direction (positive/negative), small magnitude
        drift = torch.randn(1, 1, H, W, generator=g) * 0.5

        # time ramp 0..1
        ramp = torch.linspace(0.0, 1.0, steps=T).view(1, T, 1, 1)

        # noise
        noise = torch.randn(1, T, H, W, generator=g) * 0.2

        # (1,T,H,W)
        t_raw = base.expand(1, T, H, W) + ramp * drift.expand(1, T, H, W) + noise
        return t_raw.to(dtype=torch.float32)

    def __getitem__(self, idx: int):
        t_raw = self._make_raw_clip(idx)               # (1,T,H,W)
        x = self.builder(t_raw)                        # (3,T,H,W) in contract order

        if self.cfg.normalize_enabled:
            x = _normalize_cthw(x, list(self.cfg.mean), list(self.cfg.std))

        if self.cfg.clamp_enabled:
            x = torch.clamp(x, min=self.cfg.clamp_min, max=self.cfg.clamp_max)

        x = x.contiguous()

        # label derived from mean DeltaT at last frame (balanced-ish)
        dT_last_mean = x[2, -1].mean()  # channel 2 = DeltaT
        y = (dT_last_mean > 0).to(torch.float32).view(1)  # (1,)

        return x, y


def make_dummy_dataloader_from_app_config(app_cfg, *, split: str, num_samples: int | None = None) -> DataLoader:
    """
    Builds a DataLoader using your validated AppConfig (config.yaml).
    """
    assert split in ("train", "val"), "split must be 'train' or 'val'"

    ds_cfg = DummyThermalDatasetConfig(
        num_samples=(num_samples if num_samples is not None else (512 if split == "train" else 128)),
        T=app_cfg.clip.T,
        H=app_cfg.contract.H,
        W=app_cfg.contract.W,
        seed=app_cfg.train.seed + (0 if split == "train" else 999),
        smooth_window=app_cfg.clip.smooth_window,
        deltaT_baseline=app_cfg.clip.deltaT_baseline,
        normalize_enabled=app_cfg.preprocess.normalize.enabled,
        mean=tuple(app_cfg.preprocess.normalize.mean),
        std=tuple(app_cfg.preprocess.normalize.std),
        clamp_enabled=app_cfg.preprocess.clamp.enabled,
        clamp_min=app_cfg.preprocess.clamp.min,
        clamp_max=app_cfg.preprocess.clamp.max,
    )

    dataset = DummyThermalDataset(ds_cfg)

    loader = DataLoader(
        dataset,
        batch_size=app_cfg.train.batch_size,
        shuffle=(split == "train"),
        num_workers=app_cfg.data.num_workers,
        pin_memory=app_cfg.data.pin_memory,
        drop_last=(split == "train"),
    )
    return loader


def contract_check_one_batch(app_cfg) -> None:
    """
    Convenience: fetch one batch and run strict contract assertions.
    """
    loader = make_dummy_dataloader_from_app_config(app_cfg, split="train", num_samples=max(app_cfg.train.batch_size * 2, 8))
    x, y = next(iter(loader))

    # x should be (B,3,T,H,W) float32
    assert_input_tensor(x, expected_T=app_cfg.clip.T)

    # y should be (B,1) float32
    if y.ndim == 1:
        y = y.view(-1, 1)
    if y.shape != (x.shape[0], 1):
        raise RuntimeError(f"Labels must be shape (B,1), got {tuple(y.shape)}")
    if y.dtype != torch.float32:
        raise RuntimeError(f"Labels dtype must be float32, got {y.dtype}")
