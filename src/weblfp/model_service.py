from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch

from .models import CLIPLFPEncoder, load_clip_lfp_encoder
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
) -> tuple[CLIPLFPEncoder, ModelProfile, torch.device]:
    directory = Path(package_dir) if package_dir else default_model_dir()
    profile = load_model_profile(directory)
    checkpoint = verify_checkpoint(directory)
    device = resolve_device(device_choice)
    model = load_clip_lfp_encoder(checkpoint, profile.inference_config(), device)
    return model, profile, device


def extract_embeddings(
    windows: np.ndarray,
    batch_size: int = 32,
    package_dir: str | Path | None = None,
    device_choice: DeviceChoice = "auto",
    progress_callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, ModelProfile, str]:
    model, profile, device = load_runtime(package_dir, device_choice)
    if progress_callback:
        progress_callback(0.05, f"模型已加载，正在使用 {device} 生成隐空间。")
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
                    embedding = model(batch)
            else:
                embedding = model(batch)
            outputs.append(embedding.float().cpu())
            if progress_callback:
                completed = min(start + len(batch), len(windows))
                progress_callback(completed / len(windows), f"正在编码窗口 {completed}/{len(windows)}。")
    values = torch.cat(outputs, dim=0).numpy()
    return np.asarray(values, dtype=np.float32), profile, str(device)
