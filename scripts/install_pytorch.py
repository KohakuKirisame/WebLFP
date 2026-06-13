from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import subprocess
import time
from ctypes import wintypes
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--uv-path", required=True)
    parser.add_argument("--python-path", required=True)
    parser.add_argument("--torch-version", required=True)
    parser.add_argument("--index-url", required=True)
    parser.add_argument("--backend-label", required=True)
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--job-id", required=True)
    return parser.parse_args()


def process_exists(pid: int) -> bool:
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


def write_status(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def emit(log_path: Path, message: str) -> None:
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"{message}\n")


def detached_kwargs(log_file: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": log_file,
        "stderr": log_file,
    }
    if platform.system() == "Windows":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return kwargs


def main() -> int:
    args = parse_args()
    root = Path(args.project_root)
    settings_dir = root / ".weblfp"
    settings_dir.mkdir(parents=True, exist_ok=True)
    status_path = settings_dir / "pytorch-install.json"
    log_path = settings_dir / "pytorch-install.log"
    base = {
        "job_id": args.job_id,
        "installer_pid": os.getpid(),
        "started_at": time.time(),
        "option": {
            "torch_version": args.torch_version,
            "label": args.backend_label,
            "index_url": args.index_url,
        },
    }

    return_code = 1
    try:
        emit(log_path, "WebLFP PyTorch installer started.")
        emit(log_path, f"Selected runtime: {args.backend_label}")
        emit(log_path, "Waiting for the WebLFP service to stop...")
        write_status(
            status_path,
            {**base, "state": "waiting", "message": "Waiting for WebLFP to stop."},
        )
        while process_exists(args.server_pid):
            time.sleep(0.25)

        emit(log_path, "WebLFP stopped. Starting the package installation.")
        write_status(
            status_path,
            {**base, "state": "installing", "message": f"Installing {args.backend_label}."},
        )
        command = [
            args.uv_path,
            "pip",
            "install",
            "--python",
            args.python_path,
            "--reinstall-package",
            "torch",
            "--link-mode",
            "copy",
            f"torch=={args.torch_version}",
            "--default-index",
            args.index_url,
        ]
        emit(log_path, f"Running: {' '.join(command)}")
        with log_path.open("ab") as log_file:
            result = subprocess.run(
                command,
                cwd=root,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return_code = result.returncode
        if return_code != 0:
            raise RuntimeError(f"uv exited with code {return_code}.")

        emit(log_path, "Package installation completed. Verifying the PyTorch runtime...")
        verify_code = (
            "import json, torch; "
            "print(json.dumps({'version': torch.__version__, 'cuda': torch.version.cuda, "
            "'hip': getattr(torch.version, 'hip', None), 'cudnn': torch.backends.cudnn.version()}))"
        )
        verified = subprocess.run(
            [args.python_path, "-c", verify_code],
            capture_output=True,
            text=True,
            check=False,
        )
        if verified.returncode != 0:
            raise RuntimeError("PyTorch import verification failed after installation.")
        verification = json.loads(verified.stdout)
        write_status(
            status_path,
            {
                **base,
                "state": "completed",
                "message": f"Installed {args.backend_label}. WebLFP is restarting.",
                "return_code": return_code,
                "finished_at": time.time(),
                "verification": verification,
            },
        )
        emit(log_path, f"Verified PyTorch {verification['version']}.")
    except Exception as error:
        emit(log_path, f"Installation failed: {error}")
        write_status(
            status_path,
            {
                **base,
                "state": "failed",
                "message": f"Installation failed: {error}",
                "return_code": return_code,
                "finished_at": time.time(),
            },
        )
    finally:
        emit(log_path, "Restarting WebLFP...")
        try:
            server_log_path = settings_dir / "weblfp-server.log"
            with server_log_path.open("ab") as server_log:
                subprocess.Popen(
                    [args.uv_path, "run", "--no-sync", "weblfp"],
                    cwd=root,
                    **detached_kwargs(server_log),
                )
        except OSError as error:
            emit(log_path, f"WebLFP restart failed: {error}")
            write_status(
                status_path,
                {
                    **base,
                    "state": "failed",
                    "message": f"WebLFP restart failed: {error}",
                    "return_code": return_code,
                    "finished_at": time.time(),
                },
            )
            return_code = 1
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
