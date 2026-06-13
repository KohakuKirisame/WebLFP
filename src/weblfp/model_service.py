from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
from torch import nn

from .models.lfp_encoder import LFPEncoder
from .profile import ModelProfile, default_model_dir, load_model_profile, verify_checkpoint


DeviceChoice = Literal["auto", "cpu", "cuda"]
ProgressCallback = Callable[[float, str], None]


def resolve_device(choice: DeviceChoice) -> torch.device:
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if choice == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@lru_cache(maxsize=4)
def load_runtime(
    package_dir: str | Path | None = None,
    device_choice: DeviceChoice = "auto",
) -> tuple[nn.Module, ModelProfile, torch.device]:
    directory = Path(package_dir) if package_dir else default_model_dir()
    profile = load_model_profile(directory)
    checkpoint = verify_checkpoint(directory)
    device = resolve_device(device_choice)
    model = _load_lfp_feature_encoder(checkpoint, profile, device)
    return model, profile, device


def _load_lfp_feature_encoder(
    checkpoint_path: str | Path,
    profile: ModelProfile,
    device: torch.device,
) -> LFPEncoder:
    model = LFPEncoder(**profile.architecture)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("LFP feature checkpoint must contain an inference package dictionary.")
    if checkpoint.get("format_version") != 1 or checkpoint.get("model_type") != "spike_type_inference":
        raise ValueError("Unsupported or non-inference LFP feature checkpoint format.")
    state = checkpoint.get("feature_extractor")
    if not isinstance(state, dict):
        raise ValueError("LFP feature checkpoint is missing feature_extractor.")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(f"LFP feature checkpoint architecture mismatch: {error}") from error
    model.requires_grad_(False).eval()
    return model.to(device)


def extract_embeddings(
    windows: np.ndarray,
    batch_size: int = 32,
    package_dir: str | Path | None = None,
    device_choice: DeviceChoice = "auto",
    progress_callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, ModelProfile, str]:
    model, profile, device = load_runtime(package_dir, device_choice)
    if progress_callback:
        progress_callback(0.05, f"模型已加载，正在使用 {device} 生成 LFP feature。")
    if windows.ndim != 3:
        raise ValueError(f"Expected [N, C, T] windows, got shape={windows.shape}.")
    if windows.shape[1] > profile.max_channels:
        raise ValueError(
            f"Selected {windows.shape[1]} channels; model supports at most "
            f"{profile.max_channels}."
        )
    if windows.shape[2] > profile.max_time_samples:
        raise ValueError(
            f"Window has {windows.shape[2]} samples; model supports at most "
            f"{profile.max_time_samples}."
        )

    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(windows[start : start + batch_size]))
            batch = batch.to(device=device, dtype=torch.float32)
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    embedding = _encode_batch(model, profile, batch)
            else:
                embedding = _encode_batch(model, profile, batch)
            outputs.append(embedding.float().cpu())
            if progress_callback:
                completed = min(start + len(batch), len(windows))
                progress_callback(completed / len(windows), f"正在编码窗口 {completed}/{len(windows)}。")
    values = torch.cat(outputs, dim=0).numpy()
    return np.asarray(values, dtype=np.float32), profile, str(device)


def _encode_batch(model: nn.Module, profile: ModelProfile, batch: torch.Tensor) -> torch.Tensor:
    if not isinstance(model, LFPEncoder):
        raise TypeError("Unified LFP runtime expected an LFPEncoder.")
    return model.forward_features(batch, pool=profile.pool)
