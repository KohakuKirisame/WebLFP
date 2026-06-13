from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from weblfp.api import app
from weblfp.run_history import RunStore


client = TestClient(app)


def _source(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "format": "npy",
        "sampling_rate_hz": 1875,
        "channel_axis": "first",
    }


def test_health_and_model_profile_are_available() -> None:
    assert client.get("/api/health").json() == {"status": "ok"}

    response = client.get("/api/model")

    assert response.status_code == 200
    payload = response.json()
    assert payload["checkpoint_available"] is True
    assert payload["embedding_dim"] == 256
    assert payload["epoch"] == 9


def test_pytorch_install_requires_explicit_confirmation() -> None:
    response = client.post(
        "/api/settings/pytorch-install",
        json={"option_id": "torch-2.12.0-cpu", "confirmed": False},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Explicit installation confirmation is required."


def test_recording_file_dialog_returns_selected_path(monkeypatch, tmp_path: Path) -> None:
    selected = (tmp_path / "recording.npy").resolve()
    monkeypatch.setattr("weblfp.api.select_recording_file", lambda: selected)

    response = client.post("/api/dialogs/recording-file")

    assert response.status_code == 200
    assert response.json() == {"path": str(selected)}


def test_inspect_and_preview_npy_recording(tmp_path: Path) -> None:
    path = tmp_path / "recording.npy"
    time = np.arange(1875, dtype=np.float32) / 1875
    values = np.stack([np.sin(2 * np.pi * frequency * time) for frequency in (4, 8, 12, 20)])
    np.save(path, values)

    inspect_response = client.post("/api/inspect", json={"source": _source(path)})
    preview_response = client.post(
        "/api/preview",
        json={"source": _source(path), "start_sec": 0, "duration_sec": 0.5},
    )

    assert inspect_response.status_code == 200
    assert inspect_response.json()["num_channels"] == 4
    assert preview_response.status_code == 200
    preview = preview_response.json()
    assert preview["channel_ids"] == ["0", "1", "2", "3"]
    assert len(preview["times_sec"]) == len(preview["normalized_traces"][0])


def test_stream_endpoint_returns_empty_options_for_array_recording(tmp_path: Path) -> None:
    path = tmp_path / "recording.npy"
    np.save(path, np.zeros((2, 100), dtype=np.float32))

    response = client.post("/api/streams", json={"source": _source(path)})

    assert response.status_code == 200
    assert response.json() == {"format": "npy", "streams": []}


def test_preview_covers_the_full_requested_time_range(tmp_path: Path) -> None:
    path = tmp_path / "long-recording.npy"
    time = np.arange(50 * 1875, dtype=np.float32) / 1875
    np.save(path, np.stack([np.sin(2 * np.pi * 8 * time), np.cos(2 * np.pi * 12 * time)]))

    response = client.post(
        "/api/preview",
        json={
            "source": _source(path),
            "start_sec": 10,
            "duration_sec": 30,
            "max_points": 1200,
        },
    )

    assert response.status_code == 200
    times = response.json()["times_sec"]
    assert times[0] == 10
    assert times[-1] > 39.9


def test_result_history_endpoint_can_be_empty(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("weblfp.api.results", RunStore(tmp_path / "runs"))

    response = client.get("/api/results")

    assert response.status_code == 200
    assert response.json() == []
