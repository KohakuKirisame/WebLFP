from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, Field

from .downstream_schema import SpikeTypeDecodeResult, SpikeTypeLabel
from .model_service import DeviceChoice, resolve_device
from .models.spike_count import SpikeCountPresenceHead
from .profile import project_root, sha256_file


class SpikeTypeDecoderProfile(BaseModel):
    schema_version: Literal[1] = 1
    id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    model_type: Literal["lfp_spike_count_presence"]
    checkpoint: str = Field(min_length=1)
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    epoch: int = Field(ge=0)
    target_sample_rate_hz: float = Field(gt=0)
    window_sec: float = Field(gt=0)
    hop_sec: float = Field(gt=0)
    normalization: Literal["robust_zscore_per_selected_range_channel"]
    pool: Literal["cls"]
    presence_threshold: float = Field(ge=0, le=1)
    recommended_channels: int = Field(gt=0)
    max_channels: int = Field(gt=0)
    max_time_samples: int = Field(gt=0)
    feature_dim: int = Field(gt=0)
    hidden_dim: int = Field(gt=0)
    dropout: float = Field(ge=0, lt=1)
    labels: list[SpikeTypeLabel] = Field(min_length=2, max_length=2)
    architecture: dict[str, Any]
    reference_metrics: dict[str, float]
    limitations: list[str]

    @property
    def window_samples(self) -> int:
        return round(self.window_sec * self.target_sample_rate_hz)

    @property
    def hop_samples(self) -> int:
        return round(self.hop_sec * self.target_sample_rate_hz)


def default_decoder_dir() -> Path:
    return project_root() / "models" / "spike-type-decoder"


@lru_cache(maxsize=4)
def load_decoder_profile(package_dir: str | Path | None = None) -> SpikeTypeDecoderProfile:
    directory = Path(package_dir) if package_dir else default_decoder_dir()
    try:
        return SpikeTypeDecoderProfile.model_validate_json(
            (directory / "model.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Invalid spike type decoder profile: {error}") from error


@lru_cache(maxsize=4)
def load_decoder_runtime(
    package_dir: str | Path | None = None,
    device_choice: DeviceChoice = "auto",
) -> tuple[SpikeCountPresenceHead, SpikeTypeDecoderProfile, torch.device]:
    directory = Path(package_dir) if package_dir else default_decoder_dir()
    profile = load_decoder_profile(directory)
    checkpoint = directory / profile.checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Spike type decoder checkpoint not found: {checkpoint}")
    actual_hash = sha256_file(checkpoint)
    if actual_hash.lower() != profile.checkpoint_sha256.lower():
        raise ValueError(
            f"Spike type decoder SHA-256 mismatch: expected {profile.checkpoint_sha256}, "
            f"got {actual_hash}."
        )

    payload: Any = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Spike type decoder checkpoint must contain a dictionary.")
    if payload.get("format_version") != 1 or payload.get("model_type") != "spike_type_inference":
        raise ValueError("Unsupported or non-inference spike type checkpoint format.")
    feature_state = payload.get("feature_extractor")
    head_state = payload.get("head")
    if not isinstance(feature_state, dict) or not isinstance(head_state, dict):
        raise ValueError("Spike type decoder checkpoint is missing model state dictionaries.")

    head = SpikeCountPresenceHead(
        feature_dim=profile.feature_dim,
        hidden_dim=profile.hidden_dim,
        dropout=profile.dropout,
    )
    try:
        head.load_state_dict(head_state, strict=True)
    except RuntimeError as error:
        raise ValueError(f"Spike type decoder architecture mismatch: {error}") from error

    device = resolve_device(device_choice)
    head.requires_grad_(False).to(device).eval()
    return head, profile, device


def decode_spike_types(
    features: np.ndarray,
    window_start_sec: np.ndarray,
    batch_size: int = 32,
    device_choice: DeviceChoice = "auto",
    package_dir: str | Path | None = None,
) -> SpikeTypeDecodeResult:
    directory = Path(package_dir) if package_dir else default_decoder_dir()
    profile = load_decoder_profile(directory)
    features = np.asarray(features, dtype=np.float32)
    window_start_sec = np.asarray(window_start_sec, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"Expected LFP features with shape [N, D], got {features.shape}.")
    if len(features) == 0:
        raise ValueError("No LFP features are available for downstream decoding.")
    if features.shape[1] != profile.feature_dim:
        raise ValueError(
            f"The decoder expects {profile.feature_dim}D LFP features from the unified "
            f"checkpoint, got {features.shape[1]}D."
        )
    if window_start_sec.ndim != 1 or len(window_start_sec) != len(features):
        raise ValueError("window_start_sec must be a 1D array with the same length as features.")

    head, profile, device = load_decoder_runtime(directory, device_choice)
    count_batches: list[torch.Tensor] = []
    presence_batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for offset in range(0, len(features), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(features[offset : offset + batch_size]))
            batch = batch.to(device=device, dtype=torch.float32)
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    counts, presence_probabilities = head.predict(batch)
            else:
                counts, presence_probabilities = head.predict(batch)
            count_batches.append(counts.float().cpu())
            presence_batches.append(presence_probabilities.float().cpu())

    counts = torch.cat(count_batches).numpy()
    probabilities = torch.cat(presence_batches).numpy()
    rounded = np.maximum(np.rint(counts), 0).astype(np.int64)
    presence = probabilities >= profile.presence_threshold
    label_ids = [label.id for label in profile.labels]

    return SpikeTypeDecodeResult(
        decoder_id=profile.id,
        display_name=profile.display_name,
        device=str(device),
        labels=profile.labels,
        window_sec=profile.window_sec,
        hop_sec=profile.hop_sec,
        window_start_sec=window_start_sec.tolist(),
        predicted_counts=counts.tolist(),
        rounded_counts=rounded.tolist(),
        presence_probabilities=probabilities.tolist(),
        presence=presence.tolist(),
        mean_counts={label: float(counts[:, index].mean()) for index, label in enumerate(label_ids)},
        total_predicted_counts={
            label: float(counts[:, index].sum()) for index, label in enumerate(label_ids)
        },
        presence_rates={
            label: float(presence[:, index].mean()) for index, label in enumerate(label_ids)
        },
        reference_metrics=profile.reference_metrics,
        limitations=profile.limitations,
    )
