from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import scipy.io
import spikeinterface.extractors as se
from spikeinterface.extractors.extractor_classes import (
    IntanRecordingExtractor,
    OpenEphysBinaryRecordingExtractor,
    Plexon2RecordingExtractor,
    PlexonRecordingExtractor,
    SpikeGLXRecordingExtractor,
)
from pydantic import BaseModel, Field, model_validator


RecordingFormat = Literal[
    "auto",
    "npy",
    "npz",
    "mat",
    "binary",
    "spikeglx",
    "openephys",
    "intan",
    "plexon",
    "plexon2",
    "alphaomega",
    "nwb",
]


class SourceConfig(BaseModel):
    path: str
    format: RecordingFormat = "auto"
    sampling_rate_hz: float | None = Field(default=None, gt=0)
    data_key: str | None = None
    channel_axis: Literal["auto", "first", "last"] = "auto"
    stream_id: str | None = None
    electrical_series_path: str | None = None
    segment_index: int = Field(default=0, ge=0)
    dtype: str | None = None
    num_channels: int | None = Field(default=None, gt=0)
    time_axis: Literal[0, 1] = 0
    file_offset: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_required_metadata(self) -> "SourceConfig":
        if self.format in {"npy", "npz", "mat"} and self.sampling_rate_hz is None:
            raise ValueError("sampling_rate_hz is required for NumPy and MATLAB inputs.")
        if self.format == "binary":
            missing = [
                name
                for name, value in (
                    ("sampling_rate_hz", self.sampling_rate_hz),
                    ("dtype", self.dtype),
                    ("num_channels", self.num_channels),
                )
                if value is None
            ]
            if missing:
                raise ValueError(f"Raw binary input requires: {', '.join(missing)}.")
        return self


class RecordingMetadata(BaseModel):
    path: str
    format: str
    sampling_rate_hz: float
    channel_ids: list[str]
    num_channels: int
    num_samples: int
    duration_sec: float
    dtype: str
    segment_index: int = 0
    num_segments: int = 1
    stream_id: str | None = None


class RecordingStream(BaseModel):
    stream_id: str
    stream_name: str


class RecordingStreamOptions(BaseModel):
    format: str
    streams: list[RecordingStream]


_STREAM_EXTRACTORS: dict[str, type[Any]] = {
    "spikeglx": SpikeGLXRecordingExtractor,
    "openephys": OpenEphysBinaryRecordingExtractor,
    "intan": IntanRecordingExtractor,
    "plexon": PlexonRecordingExtractor,
    "plexon2": Plexon2RecordingExtractor,
}


class RecordingAdapter(ABC):
    metadata: RecordingMetadata

    @abstractmethod
    def get_traces(
        self,
        start_frame: int,
        end_frame: int,
        channel_ids: list[str] | None = None,
    ) -> np.ndarray:
        """Return float32 traces in [channels, time] order."""


class ArrayRecording(RecordingAdapter):
    def __init__(
        self,
        array: np.ndarray,
        path: Path,
        recording_format: str,
        sampling_rate_hz: float,
        channel_axis: Literal["auto", "first", "last"],
    ) -> None:
        self._array = _to_channel_time(array, channel_axis)
        channel_count, sample_count = self._array.shape
        self.metadata = RecordingMetadata(
            path=str(path),
            format=recording_format,
            sampling_rate_hz=float(sampling_rate_hz),
            channel_ids=[str(index) for index in range(channel_count)],
            num_channels=channel_count,
            num_samples=sample_count,
            duration_sec=sample_count / sampling_rate_hz,
            dtype=str(self._array.dtype),
        )

    def get_traces(
        self,
        start_frame: int,
        end_frame: int,
        channel_ids: list[str] | None = None,
    ) -> np.ndarray:
        indices = _channel_indices(self.metadata.channel_ids, channel_ids)
        return np.asarray(self._array[indices, start_frame:end_frame], dtype=np.float32)


class SpikeInterfaceRecording(RecordingAdapter):
    def __init__(self, recording: Any, config: SourceConfig, recording_format: str) -> None:
        if config.segment_index >= recording.get_num_segments():
            raise ValueError(
                f"segment_index={config.segment_index} exceeds available segments "
                f"({recording.get_num_segments()})."
            )
        self._recording = recording
        self._segment_index = config.segment_index
        channel_ids = [str(channel_id) for channel_id in recording.get_channel_ids()]
        sample_count = int(recording.get_num_frames(segment_index=config.segment_index))
        sampling_rate = float(recording.get_sampling_frequency())
        dtype = str(recording.get_dtype())
        self.metadata = RecordingMetadata(
            path=config.path,
            format=recording_format,
            sampling_rate_hz=sampling_rate,
            channel_ids=channel_ids,
            num_channels=len(channel_ids),
            num_samples=sample_count,
            duration_sec=sample_count / sampling_rate,
            dtype=dtype,
            segment_index=config.segment_index,
            num_segments=recording.get_num_segments(),
            stream_id=config.stream_id,
        )

    def get_traces(
        self,
        start_frame: int,
        end_frame: int,
        channel_ids: list[str] | None = None,
    ) -> np.ndarray:
        selected = self._recording.get_channel_ids()
        if channel_ids is not None:
            by_string = {str(channel_id): channel_id for channel_id in selected}
            missing = sorted(set(channel_ids) - set(by_string))
            if missing:
                raise ValueError(f"Unknown channel IDs: {missing}")
            selected = [by_string[channel_id] for channel_id in channel_ids]
        traces = self._recording.get_traces(
            start_frame=start_frame,
            end_frame=end_frame,
            channel_ids=selected,
            segment_index=self._segment_index,
            return_scaled=False,
        )
        return np.asarray(traces.T, dtype=np.float32)


def _channel_indices(all_ids: list[str], selected_ids: list[str] | None) -> list[int]:
    if selected_ids is None:
        return list(range(len(all_ids)))
    index_by_id = {channel_id: index for index, channel_id in enumerate(all_ids)}
    missing = sorted(set(selected_ids) - set(index_by_id))
    if missing:
        raise ValueError(f"Unknown channel IDs: {missing}")
    return [index_by_id[channel_id] for channel_id in selected_ids]


def _to_channel_time(
    array: np.ndarray,
    channel_axis: Literal["auto", "first", "last"],
) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D recording array, got shape={array.shape}.")
    if channel_axis == "first":
        return array
    if channel_axis == "last":
        return array.T

    rows, columns = array.shape
    row_is_channel = rows <= 256 and rows < columns
    column_is_channel = columns <= 256 and columns < rows
    if row_is_channel == column_is_channel:
        raise ValueError(
            f"Cannot infer channel axis from shape={array.shape}; choose 'first' or 'last'."
        )
    return array if row_is_channel else array.T


def _load_npz(path: Path, key: str | None) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        keys = list(archive.files)
        selected = key or (keys[0] if len(keys) == 1 else None)
        if selected is None:
            raise ValueError(f"NPZ contains multiple arrays; choose data_key from {keys}.")
        if selected not in archive:
            raise ValueError(f"data_key={selected!r} not found; available keys: {keys}.")
        return np.asarray(archive[selected])


def _load_mat(path: Path, key: str | None) -> np.ndarray:
    try:
        values = scipy.io.loadmat(path)
        arrays = {
            name: value
            for name, value in values.items()
            if not name.startswith("__") and isinstance(value, np.ndarray)
        }
    except NotImplementedError:
        with h5py.File(path, "r") as file:
            arrays = {
                name: np.asarray(value)
                for name, value in file.items()
                if isinstance(value, h5py.Dataset)
            }
    selected = key or (next(iter(arrays)) if len(arrays) == 1 else None)
    if selected is None:
        raise ValueError(f"MAT contains multiple arrays; choose data_key from {sorted(arrays)}.")
    if selected not in arrays:
        raise ValueError(f"data_key={selected!r} not found; available keys: {sorted(arrays)}.")
    return arrays[selected]


def detect_format(path: Path) -> RecordingFormat:
    suffix = path.suffix.lower()
    by_suffix: dict[str, RecordingFormat] = {
        ".npy": "npy",
        ".npz": "npz",
        ".mat": "mat",
        ".rhd": "intan",
        ".rhs": "intan",
        ".plx": "plexon",
        ".pl2": "plexon2",
        ".mpx": "alphaomega",
        ".nwb": "nwb",
        ".bin": "binary",
        ".dat": "binary",
    }
    if suffix in by_suffix:
        return by_suffix[suffix]
    if path.is_dir():
        if any(path.rglob("*.meta")):
            return "spikeglx"
        if any(path.rglob("structure.oebin")):
            return "openephys"
    raise ValueError("Format could not be detected. Select a reader explicitly.")


def list_recording_streams(config: SourceConfig) -> RecordingStreamOptions:
    path = Path(config.path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    recording_format = detect_format(path) if config.format == "auto" else config.format
    extractor = _STREAM_EXTRACTORS.get(recording_format)
    if extractor is None:
        return RecordingStreamOptions(format=recording_format, streams=[])

    stream_names, stream_ids = extractor.get_streams(path)
    streams = [
        RecordingStream(stream_id=str(stream_id), stream_name=str(stream_name))
        for stream_name, stream_id in zip(stream_names, stream_ids, strict=True)
    ]
    return RecordingStreamOptions(format=recording_format, streams=streams)


def open_recording(config: SourceConfig) -> RecordingAdapter:
    path = Path(config.path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    recording_format = detect_format(path) if config.format == "auto" else config.format

    if recording_format == "npy":
        if config.sampling_rate_hz is None:
            raise ValueError("sampling_rate_hz is required for NPY input.")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        return ArrayRecording(array, path, recording_format, config.sampling_rate_hz, config.channel_axis)
    if recording_format == "npz":
        if config.sampling_rate_hz is None:
            raise ValueError("sampling_rate_hz is required for NPZ input.")
        array = _load_npz(path, config.data_key)
        return ArrayRecording(array, path, recording_format, config.sampling_rate_hz, config.channel_axis)
    if recording_format == "mat":
        if config.sampling_rate_hz is None:
            raise ValueError("sampling_rate_hz is required for MAT input.")
        array = _load_mat(path, config.data_key)
        return ArrayRecording(array, path, recording_format, config.sampling_rate_hz, config.channel_axis)

    if recording_format == "binary":
        missing = [
            name
            for name, value in (
                ("sampling_rate_hz", config.sampling_rate_hz),
                ("dtype", config.dtype),
                ("num_channels", config.num_channels),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"Raw binary input requires: {', '.join(missing)}.")
        recording = se.read_binary(
            file_paths=path,
            sampling_frequency=config.sampling_rate_hz,
            dtype=config.dtype,
            num_channels=config.num_channels,
            time_axis=config.time_axis,
            file_offset=config.file_offset,
        )
    elif recording_format == "spikeglx":
        recording = se.read_spikeglx(path, stream_id=config.stream_id)
    elif recording_format == "openephys":
        kwargs = {"stream_id": config.stream_id} if config.stream_id else {}
        recording = se.read_openephys(path, **kwargs)
    elif recording_format == "intan":
        recording = se.read_intan(path, stream_id=config.stream_id)
    elif recording_format == "plexon":
        recording = se.read_plexon(path, stream_id=config.stream_id)
    elif recording_format == "plexon2":
        recording = se.read_plexon2(path, stream_id=config.stream_id)
    elif recording_format == "alphaomega":
        folder_path = path.parent if path.is_file() else path
        lsx_files = [path.name] if path.is_file() else None
        recording = se.read_alphaomega(
            folder_path,
            lsx_files=lsx_files,
            stream_id=config.stream_id or "RAW",
        )
    elif recording_format == "nwb":
        recording = se.read_nwb_recording(
            path,
            electrical_series_path=config.electrical_series_path,
        )
    else:
        raise ValueError(f"Unsupported format: {recording_format}")

    return SpikeInterfaceRecording(recording, config, recording_format)
