from __future__ import annotations

import threading
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class InferenceCancelled(RuntimeError):
    pass


class InferenceJobStatus(BaseModel):
    job_id: str
    state: Literal["queued", "running", "completed", "failed", "cancelled"]
    progress: float = Field(ge=0, le=100)
    message: str
    result: dict[str, Any] | None = None
    error: str | None = None


class InferenceJobStore:
    def __init__(self, capacity: int = 16) -> None:
        self.capacity = capacity
        self._items: dict[str, InferenceJobStatus] = {}
        self._cancelled: set[str] = set()
        self._lock = threading.RLock()

    def create(self) -> InferenceJobStatus:
        with self._lock:
            if any(item.state in {"queued", "running"} for item in self._items.values()):
                raise RuntimeError("Another inference task is already running.")
            job = InferenceJobStatus(
                job_id=uuid4().hex[:12],
                state="queued",
                progress=0,
                message="推理任务已创建。",
            )
            self._items[job.job_id] = job
            self._trim()
            return job.model_copy(deep=True)

    def get(self, job_id: str) -> InferenceJobStatus:
        with self._lock:
            try:
                return self._items[job_id].model_copy(deep=True)
            except KeyError as error:
                raise KeyError("Inference task was not found.") from error

    def report(self, job_id: str, progress: float, message: str) -> None:
        with self._lock:
            if job_id in self._cancelled:
                raise InferenceCancelled("Inference was cancelled by the user.")
            job = self._items[job_id]
            job.state = "running"
            job.progress = max(job.progress, min(100, progress))
            job.message = message

    def complete(self, job_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            job = self._items[job_id]
            job.state = "completed"
            job.progress = 100
            job.message = "隐空间生成完成。"
            job.result = result

    def fail(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._items[job_id]
            job.state = "failed"
            job.message = "隐空间生成失败。"
            job.error = error

    def cancel(self, job_id: str) -> InferenceJobStatus:
        with self._lock:
            job = self._items[job_id]
            if job.state in {"queued", "running"}:
                self._cancelled.add(job_id)
                job.message = "正在取消推理任务…"
            return job.model_copy(deep=True)

    def mark_cancelled(self, job_id: str) -> None:
        with self._lock:
            job = self._items[job_id]
            job.state = "cancelled"
            job.message = "推理任务已取消。"
            self._cancelled.discard(job_id)

    def _trim(self) -> None:
        removable = [
            job_id
            for job_id, item in self._items.items()
            if item.state not in {"queued", "running"}
        ]
        while len(self._items) > self.capacity and removable:
            self._items.pop(removable.pop(0), None)
