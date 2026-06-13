from __future__ import annotations

import ctypes
import json
import os
import platform
import re
import subprocess
import sys
import time
from ctypes import wintypes
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import torch
from pydantic import BaseModel

from .profile import project_root


Backend = Literal["cpu", "cuda", "rocm"]

MIN_SYSTEM_MEMORY_GIB = 16.0
MIN_MEMORY_SPEED_MT_S = 4800
MIN_GPU_MEMORY_MIB = 12_000
RTX_4070_BF16_FLOOR_TFLOPS = 60.0


class PyTorchOption(BaseModel):
    id: str
    torch_version: str
    backend: Backend
    runtime_version: str | None
    index_url: str
    platforms: list[str]
    label: str
    recommended: bool = False
    compatible: bool = False
    compatibility_reason: str


PYTORCH_MATRIX: tuple[dict[str, Any], ...] = (
    {
        "version": "2.12.0",
        "recommended": True,
        "builds": (
            ("cpu", None, "https://download.pytorch.org/whl/cpu", ("Windows", "Linux", "Darwin")),
            ("cuda", "13.0", "https://download.pytorch.org/whl/cu130", ("Windows", "Linux")),
            ("cuda", "13.2", "https://download.pytorch.org/whl/cu132", ("Windows", "Linux")),
            ("rocm", "7.1", "https://download.pytorch.org/whl/rocm7.1", ("Linux",)),
            ("rocm", "7.2", "https://download.pytorch.org/whl/rocm7.2", ("Linux",)),
        ),
    },
    {
        "version": "2.11.0",
        "recommended": False,
        "builds": (
            ("cpu", None, "https://download.pytorch.org/whl/cpu", ("Windows", "Linux", "Darwin")),
            ("cuda", "13.0", "https://download.pytorch.org/whl/cu130", ("Windows", "Linux")),
            ("rocm", "7.1", "https://download.pytorch.org/whl/rocm7.1", ("Linux",)),
        ),
    },
    {
        "version": "2.10.0",
        "recommended": False,
        "builds": (
            ("cpu", None, "https://download.pytorch.org/whl/cpu", ("Windows", "Linux", "Darwin")),
            ("cuda", "13.0", "https://download.pytorch.org/whl/cu130", ("Windows", "Linux")),
            ("rocm", "7.0", "https://download.pytorch.org/whl/rocm7.0", ("Linux",)),
            ("rocm", "7.1", "https://download.pytorch.org/whl/rocm7.1", ("Linux",)),
        ),
    },
)


def _run(command: list[str], timeout: int = 10) -> str | None:
    try:
        flags = subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            creationflags=flags,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    return output or None


def _parse_version(output: str | None, pattern: str) -> str | None:
    if not output:
        return None
    match = re.search(pattern, output, flags=re.IGNORECASE)
    return match.group(1) if match else None


def _format_cudnn(version: int | None) -> str | None:
    if not version:
        return None
    major = version // 10_000
    minor = (version % 10_000) // 100
    patch = version % 100
    return f"{major}.{minor}.{patch}"


def _system_cudnn() -> tuple[str | None, list[str]]:
    candidates: list[Path] = []
    headers: list[Path] = []
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidates.extend(Path(cuda_path).glob("bin/cudnn*.dll"))
        candidates.extend(Path(cuda_path).glob("lib*/libcudnn.so*"))
        headers.append(Path(cuda_path) / "include" / "cudnn_version.h")
    if platform.system() == "Windows":
        root = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA" / "CUDNN"
        candidates.extend(root.glob("v*/bin/*/cudnn*.dll"))
        headers.extend(root.glob("v*/include/*/cudnn_version.h"))
    else:
        for root in (Path("/usr/lib"), Path("/usr/local/lib"), Path("/opt/rocm/lib")):
            candidates.extend(root.glob("**/libcudnn.so*"))
        headers.extend(
            [
                Path("/usr/include/cudnn_version.h"),
                Path("/usr/local/cuda/include/cudnn_version.h"),
            ]
        )

    version = None
    for header in headers:
        if not header.is_file():
            continue
        try:
            content = header.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        parts = []
        for name in ("CUDNN_MAJOR", "CUDNN_MINOR", "CUDNN_PATCHLEVEL"):
            match = re.search(rf"#define\s+{name}\s+(\d+)", content)
            if not match:
                parts = []
                break
            parts.append(match.group(1))
        if parts:
            version = ".".join(parts)
            break

    files = sorted({str(path) for path in candidates if path.is_file()})[:20]
    if version is None:
        for path in files:
            match = re.search(r"[\\/]v(\d+(?:\.\d+){0,2})[\\/]", path, flags=re.IGNORECASE)
            if match:
                version = match.group(1)
                break
    return version, files


def _total_physical_memory_bytes() -> int | None:
    if platform.system() == "Windows":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
        return None

    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def _windows_memory_modules() -> list[dict[str, Any]]:
    if platform.system() != "Windows":
        return []
    script = (
        "Get-CimInstance Win32_PhysicalMemory | "
        "Select-Object Manufacturer,PartNumber,Capacity,Speed,ConfiguredClockSpeed | "
        "ConvertTo-Json -Compress"
    )
    output = _run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script])
    if not output:
        return []
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return []
    rows = payload if isinstance(payload, list) else [payload]
    modules = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        modules.append(
            {
                "manufacturer": str(row.get("Manufacturer") or "").strip(),
                "part_number": str(row.get("PartNumber") or "").strip(),
                "capacity_bytes": int(row.get("Capacity") or 0),
                "rated_speed_mt_s": int(row.get("Speed") or 0) or None,
                "configured_speed_mt_s": int(row.get("ConfiguredClockSpeed") or 0) or None,
            }
        )
    return modules


def _linux_memory_speed_mt_s() -> int | None:
    if platform.system() != "Linux":
        return None
    output = _run(["dmidecode", "--type", "17"])
    if not output:
        return None
    speeds = [
        int(value)
        for value in re.findall(
            r"Configured (?:Memory )?Speed:\s*(\d+)\s*(?:MT/s|MHz)",
            output,
            flags=re.IGNORECASE,
        )
        if int(value) > 0
    ]
    return min(speeds) if speeds else None


def _memory_info() -> dict[str, Any]:
    total_bytes = _total_physical_memory_bytes()
    modules = _windows_memory_modules()
    installed_bytes = sum(module["capacity_bytes"] for module in modules) or total_bytes
    configured_speeds = [
        module["configured_speed_mt_s"]
        for module in modules
        if module["configured_speed_mt_s"] is not None
    ]
    configured_speed = min(configured_speeds) if configured_speeds else _linux_memory_speed_mt_s()
    return {
        "total_bytes": total_bytes,
        "total_gib": round(total_bytes / (1024**3), 2) if total_bytes else None,
        "installed_gib": round(installed_bytes / (1024**3), 2) if installed_bytes else None,
        "configured_speed_mt_s": configured_speed,
        "modules": modules,
    }


def _benchmark_cuda_bf16(torch_backend: Backend) -> dict[str, Any]:
    result: dict[str, Any] = {
        "evaluated": False,
        "supported": None,
        "measured_tflops": None,
        "reference": "GeForce RTX 4070-class",
        "reference_floor_tflops": RTX_4070_BF16_FLOOR_TFLOPS,
        "passes": None,
        "reason": "CPU mode does not require a CUDA BF16 benchmark.",
    }
    if torch_backend != "cuda":
        return result
    if not torch.cuda.is_available():
        result["reason"] = "The CUDA build is installed, but CUDA is not available."
        return result

    result["evaluated"] = True
    try:
        supported = bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:
        supported = bool(torch.cuda.is_bf16_supported())
    result["supported"] = supported
    if not supported:
        result["passes"] = False
        result["reason"] = "The active CUDA device does not support native BF16 operations."
        return result

    size = 4096
    iterations = 8
    left = None
    right = None
    output = None
    try:
        left = torch.randn((size, size), device="cuda", dtype=torch.bfloat16)
        right = torch.randn((size, size), device="cuda", dtype=torch.bfloat16)
        output = torch.empty_like(left)
        for _ in range(3):
            torch.mm(left, right, out=output)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            torch.mm(left, right, out=output)
        end.record()
        end.synchronize()
        elapsed_seconds = start.elapsed_time(end) / 1000
        if elapsed_seconds <= 0:
            raise RuntimeError("CUDA event timing returned zero elapsed time.")
        measured = (2 * size**3 * iterations) / elapsed_seconds / 1e12
        result["measured_tflops"] = round(measured, 1)
        result["passes"] = measured >= RTX_4070_BF16_FLOOR_TFLOPS
        result["reason"] = "Measured with a local dense BF16 matrix multiplication benchmark."
    except RuntimeError as error:
        result["reason"] = f"The BF16 benchmark failed: {error}"
    finally:
        del left, right, output
    return result


def performance_assessment(
    memory: dict[str, Any],
    nvidia: dict[str, Any],
    pytorch: dict[str, Any],
    cuda_bf16: dict[str, Any],
) -> dict[str, Any]:
    warnings = []
    installed_gib = memory.get("installed_gib")
    memory_speed = memory.get("configured_speed_mt_s")
    backend = pytorch["backend"]

    if installed_gib is not None and installed_gib < MIN_SYSTEM_MEMORY_GIB:
        warnings.append(
            {
                "code": "system-memory",
                "message": (
                    f"系统内存仅 {installed_gib:.1f} GiB，低于要求的 "
                    f"{MIN_SYSTEM_MEMORY_GIB:.0f} GiB。"
                ),
            }
        )
    if memory_speed is not None and memory_speed < MIN_MEMORY_SPEED_MT_S:
        warnings.append(
            {
                "code": "memory-speed",
                "message": (
                    f"内存有效传输率为 {memory_speed} MT/s，低于要求的 "
                    f"{MIN_MEMORY_SPEED_MT_S} MT/s。"
                ),
            }
        )

    if backend != "cpu":
        device_memory = pytorch.get("device_memory_mib") or [
            gpu["memory_mib"] for gpu in nvidia.get("gpus", [])
        ]
        if device_memory and device_memory[0] < MIN_GPU_MEMORY_MIB:
            warnings.append(
                {
                    "code": "gpu-memory",
                    "message": (
                        f"当前 GPU 显存为 {device_memory[0] / 1024:.1f} GiB，"
                        "低于要求的 12 GB。"
                    ),
                }
            )

    if backend == "cuda" and cuda_bf16.get("passes") is False:
        measured = cuda_bf16.get("measured_tflops")
        detail = (
            f"实测 {measured:.1f} TFLOP/s"
            if measured is not None
            else "当前设备不支持原生 BF16"
        )
        warnings.append(
            {
                "code": "cuda-bf16",
                "message": (
                    f"CUDA BF16 性能不足（{detail}），低于 RTX 4070 级参考下限 "
                    f"{RTX_4070_BF16_FLOOR_TFLOPS:.0f} TFLOP/s。"
                ),
            }
        )

    return {
        "acceptable": not warnings,
        "backend": backend,
        "warnings": warnings,
        "thresholds": {
            "system_memory_gib": MIN_SYSTEM_MEMORY_GIB,
            "memory_speed_mt_s": MIN_MEMORY_SPEED_MT_S,
            "gpu_memory_gb": 12,
            "cuda_bf16_tflops": RTX_4070_BF16_FLOOR_TFLOPS,
        },
    }


@lru_cache(maxsize=1)
def _detect_system_cached() -> dict[str, Any]:
    system = platform.system()
    memory = _memory_info()
    nvidia_output = _run(["nvidia-smi"])
    nvidia_query = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    driver_cuda = _parse_version(
        nvidia_output,
        r"CUDA(?:\s+UMD)?\s+Version\s*:\s*([0-9.]+)",
    )
    nvcc_output = _run(["nvcc", "--version"])
    toolkit_cuda = _parse_version(nvcc_output, r"release\s+([0-9.]+)")
    gpus: list[dict[str, Any]] = []
    if nvidia_query:
        for line in nvidia_query.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 3:
                gpus.append(
                    {
                        "name": parts[0],
                        "driver_version": parts[1],
                        "memory_mib": int(float(parts[2])),
                    }
                )

    hip_output = _run(["hipcc", "--version"])
    rocm_output = _run(["rocminfo"], timeout=20)
    rocm_version = _parse_version(hip_output, r"HIP version:\s*([0-9.]+)")
    system_cudnn_version, system_cudnn_files = _system_cudnn()

    torch_backend: Backend = "cpu"
    if torch.version.cuda:
        torch_backend = "cuda"
    elif getattr(torch.version, "hip", None):
        torch_backend = "rocm"

    device_count = torch.cuda.device_count()
    device_properties = [torch.cuda.get_device_properties(i) for i in range(device_count)]
    pytorch = {
        "version": torch.__version__,
        "backend": torch_backend,
        "cuda_build_version": torch.version.cuda,
        "hip_build_version": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(),
        "device_count": device_count,
        "devices": [properties.name for properties in device_properties],
        "device_memory_mib": [
            round(properties.total_memory / (1024**2)) for properties in device_properties
        ],
        "cudnn_available": torch.backends.cudnn.is_available(),
        "cudnn_version": _format_cudnn(torch.backends.cudnn.version()),
    }
    nvidia = {
        "available": bool(gpus),
        "gpus": gpus,
        "driver_cuda_version": driver_cuda,
        "toolkit_cuda_version": toolkit_cuda,
        "minimum_cuda_version": "13.0",
    }
    cuda_bf16 = _benchmark_cuda_bf16(torch_backend)

    return {
        "platform": {
            "system": system,
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
        },
        "memory": memory,
        "nvidia": nvidia,
        "rocm": {
            "platform_supported": system == "Linux",
            "detected": bool(hip_output or rocm_output),
            "version": rocm_version,
        },
        "pytorch": pytorch,
        "cuda_bf16": cuda_bf16,
        "performance": performance_assessment(memory, nvidia, pytorch, cuda_bf16),
        "system_cudnn_version": system_cudnn_version,
        "system_cudnn_files": system_cudnn_files,
    }


def detect_system(*, refresh: bool = False) -> dict[str, Any]:
    if refresh:
        _detect_system_cached.cache_clear()
    return _detect_system_cached()


def _version_at_least(actual: str | None, required: str) -> bool:
    if actual is None:
        return False
    actual_parts = tuple(int(part) for part in actual.split(".")[:2])
    required_parts = tuple(int(part) for part in required.split(".")[:2])
    return actual_parts >= required_parts


def pytorch_options(system_info: dict[str, Any] | None = None) -> list[PyTorchOption]:
    info = system_info or detect_system()
    current_platform = info["platform"]["system"]
    options: list[PyTorchOption] = []
    for release in PYTORCH_MATRIX:
        for backend, runtime, index_url, platforms in release["builds"]:
            compatible = current_platform in platforms
            reason = "Compatible with this system."
            if not compatible:
                reason = f"{backend.upper()} build is not available on {current_platform}."
            elif backend == "cuda":
                compatible = bool(info["nvidia"]["available"]) and _version_at_least(
                    info["nvidia"]["driver_cuda_version"],
                    runtime,
                )
                if not info["nvidia"]["available"]:
                    reason = "No NVIDIA GPU was detected."
                elif not compatible:
                    reason = (
                        f"NVIDIA driver reports CUDA {info['nvidia']['driver_cuda_version'] or 'unknown'}; "
                        f"CUDA {runtime} or newer is required."
                    )
            elif backend == "rocm":
                compatible = current_platform == "Linux" and bool(info["rocm"]["detected"])
                if current_platform != "Linux":
                    reason = "Official ROCm wheels are Linux-only."
                elif not info["rocm"]["detected"]:
                    reason = "ROCm/HIP was not detected."

            suffix = "CPU" if runtime is None else f"{backend.upper()} {runtime}"
            option_id = f"torch-{release['version']}-{backend}{(runtime or '').replace('.', '')}"
            options.append(
                PyTorchOption(
                    id=option_id,
                    torch_version=release["version"],
                    backend=backend,
                    runtime_version=runtime,
                    index_url=index_url,
                    platforms=list(platforms),
                    label=f"PyTorch {release['version']} · {suffix}",
                    recommended=bool(release["recommended"]),
                    compatible=compatible,
                    compatibility_reason=reason,
                )
            )
    return options


def get_pytorch_option(option_id: str) -> PyTorchOption:
    options = {option.id: option for option in pytorch_options()}
    try:
        option = options[option_id]
    except KeyError as error:
        raise ValueError("Unknown PyTorch installation option.") from error
    if not option.compatible:
        raise ValueError(option.compatibility_reason)
    if option.backend == "cuda" and not _version_at_least(option.runtime_version, "13.0"):
        raise ValueError("CUDA builds below 13.0 are not permitted.")
    return option


def settings_dir() -> Path:
    directory = project_root() / ".weblfp"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _process_exists(pid: int) -> bool:
    if platform.system() == "Windows":
        synchronize = 0x00100000
        wait_timeout = 0x00000102
        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(synchronize, False, pid)
        if not handle:
            return False
        try:
            return kernel32.WaitForSingleObject(handle, 0) == wait_timeout
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def install_status() -> dict[str, Any] | None:
    path = settings_dir() / "pytorch-install.json"
    if not path.is_file():
        return None
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"state": "unknown", "message": "Installation status could not be read."}
    if status.get("state") in {"waiting", "installing"}:
        installer_pid = status.get("installer_pid")
        is_stale_legacy_status = installer_pid is None and time.time() - path.stat().st_mtime > 10
        if is_stale_legacy_status or (
            isinstance(installer_pid, int) and not _process_exists(installer_pid)
        ):
            status.update(
                state="failed",
                message="Installer process stopped unexpectedly. Review the log and retry.",
            )
            path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    started_at = status.get("started_at")
    if isinstance(started_at, int | float):
        finished_at = status.get("finished_at")
        end_time = float(finished_at) if isinstance(finished_at, int | float) else time.time()
        status["elapsed_sec"] = max(0, round(end_time - float(started_at)))
    log_path = settings_dir() / "pytorch-install.log"
    status["log_tail"] = _install_log_tail(log_path)
    return status


def _install_log_tail(path: Path, max_bytes: int = 16_384, max_lines: int = 18) -> list[str]:
    if not path.is_file():
        return []
    try:
        with path.open("rb") as file:
            file.seek(0, os.SEEK_END)
            size = file.tell()
            file.seek(max(0, size - max_bytes))
            content = file.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    return [line for line in content.replace("\r", "\n").splitlines() if line.strip()][-max_lines:]


def start_pytorch_install(option: PyTorchOption) -> dict[str, Any]:
    current = install_status()
    if current and current.get("state") in {"waiting", "installing"}:
        raise ValueError("Another PyTorch installation is already running.")

    root = project_root()
    helper = root / "scripts" / "install_pytorch.py"
    monitor = root / "scripts" / "watch_pytorch_install.py"
    uv_path = Path(os.environ.get("APPDATA", "")) / "Python" / "Scripts" / "uv.exe"
    if not uv_path.is_file():
        resolved = _run(["where" if platform.system() == "Windows" else "which", "uv"])
        if not resolved:
            raise FileNotFoundError("uv executable was not found.")
        uv_path = Path(resolved.splitlines()[0].strip())

    job_id = uuid4().hex[:12]
    status_path = settings_dir() / "pytorch-install.json"
    initial = {
        "job_id": job_id,
        "state": "waiting",
        "message": "Waiting for WebLFP to stop before replacing PyTorch.",
        "option": option.model_dump(),
        "started_at": time.time(),
    }
    status_path.write_text(json.dumps(initial, indent=2), encoding="utf-8")
    log_path = settings_dir() / "pytorch-install.log"
    log_path.write_bytes(b"")
    helper_python = Path(sys.executable)
    if platform.system() == "Windows":
        pythonw = helper_python.with_name("pythonw.exe")
        if pythonw.is_file():
            helper_python = pythonw

    command = [
        str(helper_python),
        str(helper),
        "--project-root",
        str(root),
        "--uv-path",
        str(uv_path),
        "--python-path",
        sys.executable,
        "--torch-version",
        option.torch_version,
        "--index-url",
        option.index_url,
        "--backend-label",
        option.label,
        "--server-pid",
        str(os.getpid()),
        "--job-id",
        job_id,
    ]
    kwargs: dict[str, Any] = {"cwd": root, "stdin": subprocess.DEVNULL}
    if platform.system() == "Windows":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    installer = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **kwargs,
    )

    launched = {**initial, "installer_pid": installer.pid}
    if platform.system() == "Windows":
        subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(monitor),
                "--status-path",
                str(status_path),
                "--log-path",
                str(log_path),
            ],
            cwd=root,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            creationflags=subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    return launched
