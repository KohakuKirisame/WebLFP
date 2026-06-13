import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from weblfp import settings as settings_module
from weblfp.settings import performance_assessment, pytorch_options


_HELPER_SPEC = importlib.util.spec_from_file_location(
    "install_pytorch",
    Path(__file__).parents[1] / "scripts" / "install_pytorch.py",
)
assert _HELPER_SPEC and _HELPER_SPEC.loader
_HELPER = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(_HELPER)


def _system(platform: str, *, cuda: str | None = None, rocm: bool = False) -> dict:
    return {
        "platform": {"system": platform},
        "nvidia": {
            "available": cuda is not None,
            "driver_cuda_version": cuda,
        },
        "rocm": {"detected": rocm},
    }


def test_cuda_options_enforce_13_minimum_and_driver_compatibility() -> None:
    options = pytorch_options(_system("Windows", cuda="13.3"))
    cuda_options = [option for option in options if option.backend == "cuda"]

    assert cuda_options
    assert all(float(option.runtime_version or 0) >= 13.0 for option in cuda_options)
    assert all(option.compatible for option in cuda_options)
    assert any(option.runtime_version == "13.2" and option.recommended for option in cuda_options)


def test_cuda_and_rocm_options_are_disabled_when_platform_is_incompatible() -> None:
    windows_options = pytorch_options(_system("Windows"))
    linux_options = pytorch_options(_system("Linux", rocm=True))

    assert all(not option.compatible for option in windows_options if option.backend == "cuda")
    assert all(not option.compatible for option in windows_options if option.backend == "rocm")
    assert all(option.compatible for option in windows_options if option.backend == "cpu")
    assert all(option.compatible for option in linux_options if option.backend == "rocm")


def test_install_helper_can_check_current_process_without_terminating_it() -> None:
    assert _HELPER.process_exists(os.getpid()) is True


def test_performance_assessment_reports_all_cuda_threshold_failures() -> None:
    result = performance_assessment(
        memory={"installed_gib": 8.0, "configured_speed_mt_s": 3200},
        nvidia={"gpus": [{"memory_mib": 8192}]},
        pytorch={"backend": "cuda", "device_memory_mib": [8192]},
        cuda_bf16={"passes": False, "measured_tflops": 40.0},
    )

    assert result["acceptable"] is False
    assert {warning["code"] for warning in result["warnings"]} == {
        "system-memory",
        "memory-speed",
        "gpu-memory",
        "cuda-bf16",
    }


def test_cpu_mode_ignores_gpu_memory_and_cuda_bf16_thresholds() -> None:
    result = performance_assessment(
        memory={"installed_gib": 32.0, "configured_speed_mt_s": 6000},
        nvidia={"gpus": [{"memory_mib": 4096}]},
        pytorch={"backend": "cpu", "device_memory_mib": []},
        cuda_bf16={"passes": False, "measured_tflops": 10.0},
    )

    assert result["acceptable"] is True
    assert result["warnings"] == []


def test_stale_install_status_is_marked_failed(monkeypatch, tmp_path: Path) -> None:
    status_path = tmp_path / "pytorch-install.json"
    status_path.write_text(
        json.dumps({"state": "installing", "message": "Installing."}),
        encoding="utf-8",
    )
    old_time = time.time() - 30
    os.utime(status_path, (old_time, old_time))
    monkeypatch.setattr(settings_module, "settings_dir", lambda: tmp_path)

    status = settings_module.install_status()

    assert status is not None
    assert status["state"] == "failed"
    assert "stopped unexpectedly" in status["message"]


def test_install_status_includes_elapsed_time_and_log_tail(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "pytorch-install.json").write_text(
        json.dumps(
            {
                "state": "completed",
                "message": "Done.",
                "started_at": 100.0,
                "finished_at": 112.4,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "pytorch-install.log").write_text("first\nsecond\n", encoding="utf-8")
    monkeypatch.setattr(settings_module, "settings_dir", lambda: tmp_path)

    status = settings_module.install_status()

    assert status is not None
    assert status["elapsed_sec"] == 12
    assert status["log_tail"] == ["first", "second"]


def test_install_helper_records_completion_and_restarts(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if "-c" in command:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {"version": "2.12.0+cu132", "cuda": "13.2", "hip": None, "cudnn": 92101}
                ),
            )
        return SimpleNamespace(returncode=0)

    popen_calls = []

    def fake_popen(command, **kwargs):
        popen_calls.append((command, kwargs))
        return SimpleNamespace(pid=12345)

    monkeypatch.setattr(_HELPER.subprocess, "run", fake_run)
    monkeypatch.setattr(_HELPER.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "install_pytorch.py",
            "--project-root",
            str(tmp_path),
            "--uv-path",
            "uv.exe",
            "--python-path",
            "python.exe",
            "--torch-version",
            "2.12.0",
            "--index-url",
            "https://download.pytorch.org/whl/cu132",
            "--backend-label",
            "PyTorch 2.12.0 CUDA 13.2",
            "--server-pid",
            "2147483647",
            "--job-id",
            "test-job",
        ],
    )

    return_code = _HELPER.main()
    status = json.loads((tmp_path / ".weblfp" / "pytorch-install.json").read_text())

    assert return_code == 0
    assert status["state"] == "completed"
    assert status["verification"]["cuda"] == "13.2"
    assert len(calls) == 2
    assert calls[0][0][calls[0][0].index("--link-mode") + 1] == "copy"
    assert popen_calls[0][0] == ["uv.exe", "run", "--no-sync", "weblfp"]
