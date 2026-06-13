from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .model_service import DeviceChoice, extract_embeddings
from .preprocessing import (
    make_windows,
    resample_traces,
    robust_zscore_channels,
)
from .profile import ModelProfile, default_model_dir, load_model_profile
from .recording import SourceConfig
from .segment_cache import TraceSegmentCache


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
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.maximum(norms, 1e-12)
    return np.asarray(np.sum(normalized[:-1] * normalized[1:], axis=1), dtype=np.float32)


def run_inference(
    source: SourceConfig,
    start_sec: float,
    end_sec: float,
    channel_ids: list[str] | None = None,
    batch_size: int = 32,
    device_choice: DeviceChoice = "auto",
    package_dir: str | Path | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    segment_cache: TraceSegmentCache | None = None,
) -> InferenceResult:
    def report(progress: float, message: str) -> None:
        if progress_callback:
            progress_callback(progress, message)

    report(0.03, "正在读取模型配置。")
    directory = Path(package_dir) if package_dir else default_model_dir()
    profile = load_model_profile(directory)
    if channel_ids is not None and len(channel_ids) > profile.max_channels:
        raise ValueError(f"At most {profile.max_channels} channels can be selected.")

    report(0.08, "正在读取或复用所选 LFP 片段。")
    cache = segment_cache or TraceSegmentCache()
    segment, cache_hit = cache.get(
        source=source,
        start_sec=start_sec,
        end_sec=end_sec,
        channel_ids=channel_ids,
        default_channel_count=profile.recommended_channels,
    )
    metadata = segment.metadata
    selected = segment.channel_ids
    if len(selected) > profile.max_channels:
        raise ValueError(f"At most {profile.max_channels} channels can be selected.")
    traces = segment.traces
    report(
        0.3 if cache_hit else 0.18,
        "已从内存复用所选片段。" if cache_hit else "所选片段已读取并缓存到内存。",
    )
    report(0.3, "正在按模型采样率重采样。")
    traces = resample_traces(
        traces,
        source_rate_hz=metadata.sampling_rate_hz,
        target_rate_hz=profile.target_sample_rate_hz,
    )
    report(0.36, "正在执行所选范围逐通道 Robust z-score。")
    traces = robust_zscore_channels(traces)
    report(0.4, "正在切分推理窗口。")
    windows, starts = make_windows(traces, profile.window_samples, profile.hop_samples)
    report(0.54, "正在加载模型并生成 LFP feature。")

    def embedding_progress(progress: float, message: str) -> None:
        report(0.54 + progress * 0.4, message)

    embeddings, profile, device = extract_embeddings(
        windows,
        batch_size=batch_size,
        package_dir=directory,
        device_choice=device_choice,
        progress_callback=embedding_progress,
    )

    report(0.96, "正在整理 LFP feature 可视化结果。")
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
    report(1, "LFP feature 生成完成。")
    return result
