from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .recording import RecordingAdapter, RecordingMetadata, SourceConfig, open_recording


@dataclass(frozen=True)
class TraceSegment:
    metadata: RecordingMetadata
    channel_ids: list[str]
    start_sec: float
    end_sec: float
    traces: np.ndarray


OpenRecording = Callable[[SourceConfig], RecordingAdapter]


class TraceSegmentCache:
    """Keep the most recently requested raw trace segment in memory."""

    def __init__(self, opener: OpenRecording = open_recording) -> None:
        self._opener = opener
        self._key: tuple[object, ...] | None = None
        self._segment: TraceSegment | None = None
        self._lock = threading.RLock()

    def get(
        self,
        source: SourceConfig,
        start_sec: float,
        end_sec: float,
        channel_ids: list[str] | None,
        default_channel_count: int,
        clamp_end: bool = False,
    ) -> tuple[TraceSegment, bool]:
        key = self._cache_key(
            source,
            start_sec,
            end_sec,
            channel_ids,
            default_channel_count,
        )
        with self._lock:
            if key == self._key and self._segment is not None:
                if not clamp_end and end_sec > self._segment.metadata.duration_sec + 1e-9:
                    raise ValueError(
                        f"end_sec={end_sec} exceeds recording duration "
                        f"{self._segment.metadata.duration_sec:.6f} s."
                    )
                return self._segment, True

            recording = self._opener(source)
            metadata = recording.metadata
            resolved_end_sec = min(end_sec, metadata.duration_sec) if clamp_end else end_sec
            if start_sec < 0 or resolved_end_sec <= start_sec:
                raise ValueError("Segment range must satisfy 0 <= start_sec < end_sec.")
            if resolved_end_sec > metadata.duration_sec + 1e-9:
                raise ValueError(
                    f"end_sec={resolved_end_sec} exceeds recording duration "
                    f"{metadata.duration_sec:.6f} s."
                )

            selected = list(channel_ids or metadata.channel_ids[:default_channel_count])
            start_frame = round(start_sec * metadata.sampling_rate_hz)
            end_frame = round(resolved_end_sec * metadata.sampling_rate_hz)
            traces = np.array(
                recording.get_traces(start_frame, end_frame, selected),
                dtype=np.float32,
                order="C",
                copy=True,
            )
            traces.setflags(write=False)
            segment = TraceSegment(
                metadata=metadata,
                channel_ids=selected,
                start_sec=start_sec,
                end_sec=resolved_end_sec,
                traces=traces,
            )
            self._key = key
            self._segment = segment
            return segment, False

    def clear(self) -> None:
        with self._lock:
            self._key = None
            self._segment = None

    @staticmethod
    def _cache_key(
        source: SourceConfig,
        start_sec: float,
        end_sec: float,
        channel_ids: list[str] | None,
        default_channel_count: int,
    ) -> tuple[object, ...]:
        path = Path(source.path).expanduser().resolve()
        stat = path.stat()
        source_data = source.model_dump(mode="json")
        source_data["path"] = str(path)
        return (
            json.dumps(source_data, sort_keys=True, separators=(",", ":")),
            stat.st_mtime_ns,
            stat.st_size,
            round(start_sec, 9),
            round(end_sec, 9),
            tuple(channel_ids) if channel_ids is not None else None,
            default_channel_count,
        )
