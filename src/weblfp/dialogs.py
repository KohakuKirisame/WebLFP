from __future__ import annotations

import platform
import subprocess
from pathlib import Path


WINDOWS_RECORDING_FILTER = (
    "LFP recordings (*.npy;*.npz;*.mat;*.bin;*.dat;*.rhd;*.rhs;*.plx;*.pl2;*.mpx;*.nwb)|"
    "*.npy;*.npz;*.mat;*.bin;*.dat;*.rhd;*.rhs;*.plx;*.pl2;*.mpx;*.nwb|"
    "All files (*.*)|*.*"
)


def _select_recording_file_windows() -> Path | None:
    script = rf"""
Add-Type -AssemblyName System.Windows.Forms
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[System.Windows.Forms.Application]::EnableVisualStyles()
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = '选择 LFP 记录'
$dialog.Filter = '{WINDOWS_RECORDING_FILTER}'
$dialog.Multiselect = $false
$dialog.CheckFileExists = $true
$dialog.RestoreDirectory = $true
$result = $dialog.ShowDialog()
if ($result -eq [System.Windows.Forms.DialogResult]::OK) {{
    Write-Output $dialog.FileName
}}
"""
    try:
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-STA",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                script,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except OSError as error:
        raise RuntimeError("Windows file dialog could not be launched.") from error
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        message = "Windows file dialog failed."
        if detail:
            message = f"{message} {detail}"
        raise RuntimeError(message)

    selected = result.stdout.strip()
    return Path(selected).resolve() if selected else None


def _select_recording_file_tk() -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as error:
        raise RuntimeError("The Python tkinter module is not available.") from error

    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.update()
        try:
            selected = filedialog.askopenfilename(
                parent=root,
                title="选择 LFP 记录",
                filetypes=(
                    (
                        "LFP recordings",
                        "*.npy *.npz *.mat *.bin *.dat *.rhd *.rhs *.plx *.pl2 *.mpx *.nwb",
                    ),
                    ("All files", "*.*"),
                ),
            )
        finally:
            root.destroy()
    except tk.TclError as error:
        raise RuntimeError("The native file dialog could not be opened.") from error

    return Path(selected).resolve() if selected else None


def select_recording_file() -> Path | None:
    if platform.system() == "Windows":
        return _select_recording_file_windows()
    return _select_recording_file_tk()
