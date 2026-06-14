from __future__ import annotations

import os
import platform
import subprocess
from typing import Literal

from pydantic import BaseModel

from .profile import project_root


REPOSITORY_URL = "https://github.com/KohakuKirisame/WebLFP"
DEFAULT_BRANCH = "main"


class UpdateStatus(BaseModel):
    repository_url: str = REPOSITORY_URL
    branch: str = DEFAULT_BRANCH
    status: Literal[
        "up_to_date",
        "update_available",
        "local_ahead",
        "diverged",
        "unavailable",
    ]
    update_available: bool | None
    local_commit: str | None = None
    remote_commit: str | None = None
    latest_commit_url: str | None = None
    detail: str


def _git(arguments: list[str], timeout: int = 15) -> subprocess.CompletedProcess[str]:
    flags = subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
    return subprocess.run(
        ["git", "-C", str(project_root()), *arguments],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        creationflags=flags,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )


def _git_output(arguments: list[str], timeout: int = 15) -> str | None:
    try:
        result = _git(arguments, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    try:
        return _git(["merge-base", "--is-ancestor", ancestor, descendant]).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _has_commit(commit: str) -> bool:
    try:
        return _git(["cat-file", "-e", f"{commit}^{{commit}}"]).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def check_for_updates() -> UpdateStatus:
    local_commit = _git_output(["rev-parse", "HEAD"])
    remote_line = _git_output(
        ["ls-remote", REPOSITORY_URL, f"refs/heads/{DEFAULT_BRANCH}"],
        timeout=20,
    )
    if local_commit is None or remote_line is None:
        return UpdateStatus(
            status="unavailable",
            update_available=None,
            local_commit=local_commit,
            detail="The local Git commit or the GitHub branch could not be read.",
        )

    remote_commit = remote_line.split()[0]
    latest_commit_url = f"{REPOSITORY_URL}/commit/{remote_commit}"
    common = {
        "local_commit": local_commit,
        "remote_commit": remote_commit,
        "latest_commit_url": latest_commit_url,
    }
    if local_commit == remote_commit:
        return UpdateStatus(
            **common,
            status="up_to_date",
            update_available=False,
            detail="The local commit matches GitHub main.",
        )

    if _has_commit(remote_commit):
        if _is_ancestor(remote_commit, local_commit):
            return UpdateStatus(
                **common,
                status="local_ahead",
                update_available=False,
                detail="The local checkout contains commits newer than GitHub main.",
            )
        if _is_ancestor(local_commit, remote_commit):
            return UpdateStatus(
                **common,
                status="update_available",
                update_available=True,
                detail="GitHub main contains commits newer than the local checkout.",
            )
        return UpdateStatus(
            **common,
            status="diverged",
            update_available=True,
            detail="The local checkout and GitHub main have diverged.",
        )

    tracking_commit = _git_output(
        ["rev-parse", "--verify", f"refs/remotes/origin/{DEFAULT_BRANCH}"]
    )
    if tracking_commit == remote_commit and _is_ancestor(tracking_commit, local_commit):
        return UpdateStatus(
            **common,
            status="local_ahead",
            update_available=False,
            detail="The local checkout contains commits newer than GitHub main.",
        )
    return UpdateStatus(
        **common,
        status="update_available",
        update_available=True,
        detail="GitHub main points to a different, newer commit.",
    )
