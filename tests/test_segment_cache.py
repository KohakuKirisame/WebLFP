from pathlib import Path

import numpy as np
import pytest

from weblfp.recording import RecordingMetadata, SourceConfig
from weblfp.segment_cache import TraceSegmentCache


class FakeRecording:
    def __init__(self, path: Path) -> None:
        self.values = np.arange(4 * 1000, dtype=np.float32).reshape(4, 1000)
        self.read_count = 0
        self.metadata = RecordingMetadata(
            path=str(path),
            format="npy",
            sampling_rate_hz=1000,
            channel_ids=["0", "1", "2", "3"],
            num_channels=4,
            num_samples=1000,
            duration_sec=1,
            dtype="float32",
        )

    def get_traces(
        self,
        start_frame: int,
        end_frame: int,
        channel_ids: list[str] | None = None,
    ) -> np.ndarray:
        self.read_count += 1
        selected = [int(value) for value in channel_ids or self.metadata.channel_ids]
        return self.values[selected, start_frame:end_frame]


def test_trace_segment_cache_reuses_one_in_memory_copy(tmp_path: Path) -> None:
    path = tmp_path / "recording.npy"
    path.touch()
    recording = FakeRecording(path)
    open_count = 0

    def open_fake(_: SourceConfig) -> FakeRecording:
        nonlocal open_count
        open_count += 1
        return recording

    cache = TraceSegmentCache(opener=open_fake)
    source = SourceConfig(
        path=str(path),
        format="npy",
        sampling_rate_hz=1000,
        channel_axis="first",
    )
    first, first_hit = cache.get(source, 0.1, 0.5, ["0", "2"], 4)
    second, second_hit = cache.get(source, 0.1, 0.5, ["0", "2"], 4)

    assert first_hit is False
    assert second_hit is True
    assert first is second
    assert open_count == 1
    assert recording.read_count == 1
    assert first.traces.shape == (2, 400)
    assert first.traces.flags.writeable is False
    assert not np.shares_memory(first.traces, recording.values)


def test_trace_segment_cache_replaces_segment_when_selection_changes(tmp_path: Path) -> None:
    path = tmp_path / "recording.npy"
    path.touch()
    recording = FakeRecording(path)
    cache = TraceSegmentCache(opener=lambda _: recording)
    source = SourceConfig(
        path=str(path),
        format="npy",
        sampling_rate_hz=1000,
        channel_axis="first",
    )

    cache.get(source, 0, 0.4, ["0"], 4)
    _, cache_hit = cache.get(source, 0, 0.4, ["1"], 4)

    assert cache_hit is False
    assert recording.read_count == 2


def test_preview_clamp_does_not_allow_inference_past_recording_end(tmp_path: Path) -> None:
    path = tmp_path / "recording.npy"
    path.touch()
    recording = FakeRecording(path)
    cache = TraceSegmentCache(opener=lambda _: recording)
    source = SourceConfig(
        path=str(path),
        format="npy",
        sampling_rate_hz=1000,
        channel_axis="first",
    )

    preview, _ = cache.get(source, 0.8, 1.2, ["0"], 4, clamp_end=True)

    assert preview.end_sec == 1
    with pytest.raises(ValueError, match="exceeds recording duration"):
        cache.get(source, 0.8, 1.2, ["0"], 4)
