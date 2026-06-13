from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .model_service import DeviceChoice, extract_embeddings
from .preprocessing import (
    make_windows,
    resample_traces,
    robust_zscore_per_window_channel,
)
from .profile import ModelProfile, default_model_dir, load_model_profile
from .recording import SourceConfig, open_recording


@dataclass
class InferenceResult:
    embeddings: np.ndarray
    window_start_sec: np.ndarray
    window_end_sec: np.ndarray
    pca_2d: np.ndarray
    adjacent_cosine_similarity: np.ndarray
    device: str
    profile: ModelProfile
    source_sample_rate_hz: float
    selected_channel_ids: list[str]


def _pca_2d(values: np.ndarray) -> np.ndarray:
    if len(values) == 1:
        return np.zeros((1, 2), dtype=np.float32)
    centered = values - values.mean(axis=0, keepdims=True)
    u, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    dimensions = min(2, u.shape[1])
    coordinates = u[:, :dimensions] * singular_values[:dimensions]
    if dimensions == 1:
        coordinates = np.column_stack([coordinates, np.zeros(len(values))])
    return np.asarray(coordinates, dtype=np.float32)


def _adjacent_similarity(values: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.empty(0, dtype=np.float32)
    return np.asarray(np.sum(values[:-1] * values[1:], axis=1), dtype=np.float32)


def run_inference(
    source: SourceConfig,
    start_sec: float,
    end_sec: float,
    channel_ids: list[str] | None = None,
    batch_size: int = 32,
    device_choice: DeviceChoice = "auto",
    package_dir: str | Path | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> InferenceResult:
    def report(progress: float, message: str) -> None:
        if progress_callback:
            progress_callback(progress, message)

    report(0.03, "正在读取模型配置。")
    directory = Path(package_dir) if package_dir else default_model_dir()
    profile = load_model_profile(directory)
    report(0.08, "正在打开 LFP 记录。")
    recording = open_recording(source)
    metadata = recording.metadata

    if start_sec < 0 or end_sec <= start_sec:
        raise ValueError("Inference range must satisfy 0 <= start_sec < end_sec.")
    if end_sec > metadata.duration_sec + 1e-9:
        raise ValueError(
            f"end_sec={end_sec} exceeds recording duration {metadata.duration_sec:.6f} s."
        )
    selected = channel_ids or metadata.channel_ids[: profile.recommended_channels]
    if len(selected) > profile.max_channels:
        raise ValueError(f"At most {profile.max_channels} channels can be selected.")

    start_frame = round(start_sec * metadata.sampling_rate_hz)
    end_frame = round(end_sec * metadata.sampling_rate_hz)
    report(0.18, "正在读取所选通道和时间范围。")
    traces = recording.get_traces(start_frame, end_frame, selected)
    report(0.3, "正在按模型采样率重采样。")
    traces = resample_traces(
        traces,
        source_rate_hz=metadata.sampling_rate_hz,
        target_rate_hz=profile.target_sample_rate_hz,
    )
    report(0.4, "正在切分推理窗口。")
    windows, starts = make_windows(traces, profile.window_samples, profile.hop_samples)
    report(0.48, "正在执行逐窗口 Robust z-score。")
    windows = robust_zscore_per_window_channel(windows)
    report(0.54, "正在加载模型并生成隐空间。")

    def embedding_progress(progress: float, message: str) -> None:
        report(0.54 + progress * 0.4, message)

    embeddings, profile, device = extract_embeddings(
        windows,
        batch_size=batch_size,
        package_dir=directory,
        device_choice=device_choice,
        progress_callback=embedding_progress,
    )

    report(0.96, "正在整理隐空间可视化结果。")
    window_start_sec = start_sec + starts / profile.target_sample_rate_hz
    window_end_sec = window_start_sec + profile.window_sec
    result = InferenceResult(
        embeddings=embeddings,
        window_start_sec=np.asarray(window_start_sec, dtype=np.float64),
        window_end_sec=np.asarray(window_end_sec, dtype=np.float64),
        pca_2d=_pca_2d(embeddings),
        adjacent_cosine_similarity=_adjacent_similarity(embeddings),
        device=device,
        profile=profile,
        source_sample_rate_hz=metadata.sampling_rate_hz,
        selected_channel_ids=selected,
    )
    report(1, "隐空间生成完成。")
    return result
