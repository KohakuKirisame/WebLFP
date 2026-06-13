from pathlib import Path

import numpy as np
import pytest

from weblfp.recording import SourceConfig, list_recording_streams, open_recording


def test_npy_recording_preserves_channel_time_order(tmp_path: Path) -> None:
    path = tmp_path / "recording.npy"
    source = np.arange(4 * 100, dtype=np.float32).reshape(4, 100)
    np.save(path, source)

    recording = open_recording(
        SourceConfig(
            path=str(path),
            format="npy",
            sampling_rate_hz=1000,
            channel_axis="first",
        )
    )

    assert recording.metadata.num_channels == 4
    assert recording.metadata.num_samples == 100
    np.testing.assert_array_equal(recording.get_traces(10, 20, ["1", "3"]), source[[1, 3], 10:20])


def test_npy_object_array_is_converted_to_numeric_matrix(tmp_path: Path) -> None:
    path = tmp_path / "object-recording.npy"
    source = np.arange(4 * 100, dtype=np.float32).reshape(4, 100)
    np.save(path, source.astype(object))

    recording = open_recording(
        SourceConfig(
            path=str(path),
            format="npy",
            sampling_rate_hz=1000,
            channel_axis="first",
        )
    )

    assert recording.metadata.dtype == "float32"
    np.testing.assert_array_equal(recording.get_traces(10, 20, ["1", "3"]), source[[1, 3], 10:20])


def test_npy_object_array_rejects_non_numeric_content(tmp_path: Path) -> None:
    path = tmp_path / "object-recording.npy"
    np.save(path, np.array([[{"value": 1}]], dtype=object))

    with pytest.raises(ValueError, match="rectangular numeric matrix"):
        open_recording(
            SourceConfig(
                path=str(path),
                format="npy",
                sampling_rate_hz=1000,
                channel_axis="first",
            )
        )


def test_auto_axis_rejects_ambiguous_array(tmp_path: Path) -> None:
    path = tmp_path / "ambiguous.npy"
    np.save(path, np.zeros((32, 32), dtype=np.float32))

    with pytest.raises(ValueError, match="Cannot infer channel axis"):
        open_recording(
            SourceConfig(
                path=str(path),
                format="npy",
                sampling_rate_hz=1000,
                channel_axis="auto",
            )
        )


def test_plexon_stream_discovery_preserves_exact_stream_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "recording.plx"
    path.touch()

    class FakePlexonExtractor:
        @classmethod
        def get_streams(cls, file_path: Path) -> tuple[list[str], list[str]]:
            assert file_path == path.resolve()
            return ["LFP ", "SPK "], ["LFP ", "SPK "]

    monkeypatch.setitem(
        __import__("weblfp.recording", fromlist=["_STREAM_EXTRACTORS"])._STREAM_EXTRACTORS,
        "plexon",
        FakePlexonExtractor,
    )

    options = list_recording_streams(SourceConfig(path=str(path), format="auto"))

    assert options.format == "plexon"
    assert [item.stream_id for item in options.streams] == ["LFP ", "SPK "]
    assert [item.stream_name for item in options.streams] == ["LFP ", "SPK "]
