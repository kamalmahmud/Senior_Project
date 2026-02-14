# ivthermal/config_schema.py
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

from ivthermal.contracts import SPEC


# -------------------------
# Config schema (Pydantic v2)
# -------------------------

class ContractConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal["1.0"]
    input_format: Literal["NCTHW"]
    channels: list[Literal["T_raw", "T_smooth", "DeltaT"]]
    allowed_T: list[Literal[16, 32]]
    H: Literal[112]
    W: Literal[112]
    output: Literal["logits[B,1]"]

    @model_validator(mode="after")
    def _match_spec(self):
        # Enforce exact declaration (order matters)
        if self.channels != list(SPEC.channels):
            raise ValueError(f"contract.channels must be {list(SPEC.channels)}")
        if self.allowed_T != list(SPEC.allowed_T):
            raise ValueError(f"contract.allowed_T must be {list(SPEC.allowed_T)}")
        return self


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root_dir: str
    train_list: str
    val_list: str
    num_workers: int = Field(ge=0, le=64, default=4)
    pin_memory: bool = True


class ClipConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    T: Literal[16, 32] = 16
    fps: int = Field(ge=1, le=240, default=8)
    decode_backend: Literal["opencv", "decord"] = "opencv"
    smooth_window: int = Field(default=5, ge=1)
    deltaT_baseline: Literal["first_frame"] = "first_frame"

    @field_validator("smooth_window")
    @classmethod
    def _odd_window(cls, v: int) -> int:
        if v % 2 == 0:
            raise ValueError("clip.smooth_window must be an odd integer")
        return v


class NormalizeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    mean: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    std: list[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0])

    @model_validator(mode="after")
    def _len3(self):
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("preprocess.normalize.mean and std must have length 3")
        if any(s == 0.0 for s in self.std):
            raise ValueError("preprocess.normalize.std entries must be non-zero")
        return self


class ClampConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    min: float = -10.0
    max: float = 10.0

    @model_validator(mode="after")
    def _min_lt_max(self):
        if self.min >= self.max:
            raise ValueError("preprocess.clamp.min must be < preprocess.clamp.max")
        return self


class PreprocessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalize: NormalizeConfig = Field(default_factory=NormalizeConfig)
    clamp: ClampConfig = Field(default_factory=ClampConfig)


class TSMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    n_div: int = Field(default=8, ge=2, le=64)
    mode: Literal["shift"] = "shift"


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["tsm_mobilenetv2"]
    width_mult: float = Field(default=1.0, ge=0.25, le=2.0)
    tsm: TSMConfig = Field(default_factory=TSMConfig)
    dropout: float = Field(default=0.2, ge=0.0, le=0.9)


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: Literal["cuda", "cpu"] = "cuda"
    batch_size: int = Field(default=16, ge=1, le=4096)
    epochs: int = Field(default=30, ge=1, le=10000)
    lr: float = Field(default=5e-4, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    amp: bool = True
    grad_clip_norm: float = Field(default=1.0, ge=0.0)
    log_every: int = Field(default=20, ge=1)
    eval_every: int = Field(default=1, ge=1)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)


class OptimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["adamw", "sgd"] = "adamw"
    betas: list[float] = Field(default_factory=lambda: [0.9, 0.999])

    @model_validator(mode="after")
    def _betas(self):
        if self.name == "adamw":
            if len(self.betas) != 2:
                raise ValueError("optim.betas must have length 2 for adamw")
        return self


class SchedulerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["cosine", "step", "none"] = "cosine"
    warmup_epochs: int = Field(default=2, ge=0, le=1000)
    min_lr: float = Field(default=1e-6, ge=0.0)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract: ContractConfig
    data: DataConfig
    clip: ClipConfig
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    model: ModelConfig
    train: TrainConfig
    optim: OptimConfig = Field(default_factory=OptimConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    output_dir: str

    @model_validator(mode="after")
    def _cross_checks(self):
        # Ensure clip.T is allowed by contract (redundant but explicit)
        if self.clip.T not in self.contract.allowed_T:
            raise ValueError("clip.T must be included in contract.allowed_T")
        return self


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("config.yaml must parse to a top-level mapping/dict")

    return AppConfig.model_validate(raw)
