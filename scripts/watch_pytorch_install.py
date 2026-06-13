from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-path", required=True)
    parser.add_argument("--log-path", required=True)
    return parser.parse_args()


def read_status(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def configure_console() -> None:
    if os.name != "nt":
        return
    kernel32 = ctypes.windll.kernel32
    if not kernel32.GetConsoleWindow():
        kernel32.AllocConsole()
    sys.stdout = open("CONOUT$", "w", encoding="utf-8", errors="replace", buffering=1)
    sys.stderr = open("CONOUT$", "w", encoding="utf-8", errors="replace", buffering=1)


def main() -> int:
    args = parse_args()
    status_path = Path(args.status_path)
    log_path = Path(args.log_path)
    configure_console()
    if os.name == "nt":
        ctypes.windll.kernel32.SetConsoleTitleW("WebLFP PyTorch Installer")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("WebLFP PyTorch 安装进度", flush=True)
    print("窗口会持续显示状态和安装日志；关闭它不会中断后台安装。\n", flush=True)
    position = 0
    last_state = None
    last_heartbeat = time.monotonic()
    started = time.monotonic()

    while True:
        if log_path.is_file():
            with log_path.open("rb") as log_file:
                log_file.seek(position)
                chunk = log_file.read()
                position = log_file.tell()
            if chunk:
                print(chunk.decode("utf-8", errors="replace"), end="", flush=True)

        status = read_status(status_path)
        state = status.get("state") if status else None
        if state and state != last_state:
            print(f"\n状态：{state} - {status.get('message', '')}", flush=True)
            last_state = state
        if state in {"completed", "failed"}:
            print("\nWebLFP 正在恢复服务，本窗口将在 8 秒后关闭。", flush=True)
            time.sleep(8)
            return 0 if state == "completed" else 1

        now = time.monotonic()
        if now - last_heartbeat >= 3:
            elapsed = round(now - started)
            phase = "等待 WebLFP 停止" if state == "waiting" else "下载并安装 PyTorch"
            print(f"[{elapsed:>4} 秒] {phase}，任务仍在运行…", flush=True)
            last_heartbeat = now
        time.sleep(0.5)


if __name__ == "__main__":
    raise SystemExit(main())
