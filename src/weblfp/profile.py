from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class ModelProfile(BaseModel):
    id: str
    display_name: str
    model_type: Literal["clip_lfp_inference"]
    checkpoint: str
    checkpoint_sha256: str
    epoch: int
    target_sample_rate_hz: float = Field(gt=0)
    window_sec: float = Field(gt=0)
    hop_sec: float = Field(gt=0)
    normalization: Literal["robust_zscore_per_window_channel"]
    pool: Literal["cls", "mean"]
    embedding_dim: int = Field(gt=0)
    embedding_normalization: Literal["l2"]
    recommended_channels: int = Field(gt=0)
    max_channels: int = Field(gt=0)
    max_time_samples: int = Field(gt=0)
    architecture: dict[str, Any]
    projector: dict[str, Any]
    limitations: list[str]

    @property
    def window_samples(self) -> int:
        return round(self.window_sec * self.target_sample_rate_hz)

    @property
    def hop_samples(self) -> int:
        return round(self.hop_sec * self.target_sample_rate_hz)

    def checkpoint_path(self, package_dir: Path) -> Path:
        return package_dir / self.checkpoint

    def inference_config(self) -> dict[str, Any]:
        return {
            "encoder": self.architecture,
            "pool": self.pool,
            "projector": self.projector,
        }


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_model_dir() -> Path:
    return project_root() / "models" / "clip-lfp-best"


@lru_cache(maxsize=8)
def load_model_profile(package_dir: str | Path | None = None) -> ModelProfile:
    directory = Path(package_dir) if package_dir else default_model_dir()
    with (directory / "model.json").open("r", encoding="utf-8") as file:
        return ModelProfile.model_validate(json.load(file))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def verify_checkpoint(package_dir: str | Path | None = None) -> Path:
    directory = Path(package_dir) if package_dir else default_model_dir()
    profile = load_model_profile(directory)
    checkpoint = profile.checkpoint_path(directory)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")
    actual_hash = sha256_file(checkpoint)
    if actual_hash.lower() != profile.checkpoint_sha256.lower():
        raise ValueError(
            f"Checkpoint SHA-256 mismatch: expected {profile.checkpoint_sha256}, "
            f"got {actual_hash}."
        )
    return checkpoint
