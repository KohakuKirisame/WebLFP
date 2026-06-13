from pathlib import Path
from types import SimpleNamespace

import pytest

from weblfp import dialogs


def test_windows_file_dialog_returns_selected_path(monkeypatch, tmp_path: Path) -> None:
    selected = tmp_path / "recording.npy"

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=f"{selected}\n", stderr="")

    monkeypatch.setattr(dialogs.subprocess, "run", fake_run)

    assert dialogs._select_recording_file_windows() == selected.resolve()


def test_windows_file_dialog_returns_none_when_cancelled(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(dialogs.subprocess, "run", fake_run)

    assert dialogs._select_recording_file_windows() is None


def test_windows_file_dialog_reports_launch_failure(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="STA initialization failed")

    monkeypatch.setattr(dialogs.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="STA initialization failed"):
        dialogs._select_recording_file_windows()
