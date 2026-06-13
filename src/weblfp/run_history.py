from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from pydantic import BaseModel

from .inference import InferenceResult
from .profile import project_root


class RunSummary(BaseModel):
    run_id: str
    created_at: str
    source_name: str
    source_path: str
    start_sec: float
    end_sec: float
    window_count: int
    embedding_dim: int
    device: str


class StoredRunData(BaseModel):
    run_id: str
    window_count: int
    embedding_dim: int
    device: str
    source_sample_rate_hz: float
    model_sample_rate_hz: float
    selected_channel_ids: list[str]
    window_start_sec: list[float]
    pca_2d: list[list[float]]
    adjacent_cosine_similarity: list[float]
    embedding_norm_min: float
    embedding_norm_max: float
    downstream: dict[str, Any] | None = None


class RunStore:
    def __init__(self, directory: Path | None = None) -> None:
        self.directory = directory or project_root() / ".weblfp" / "runs"
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def add(self, result: InferenceResult, request: dict[str, Any]) -> StoredRunData:
        run_id = uuid4().hex[:12]
        created_at = datetime.now(timezone.utc).isoformat()
        source = request["source"]
        source_path = str(source["path"])
        norms = np.linalg.norm(result.embeddings, axis=1)
        summary = RunSummary(
            run_id=run_id,
            created_at=created_at,
            source_name=Path(source_path).name or source_path,
            source_path=source_path,
            start_sec=float(request["start_sec"]),
            end_sec=float(request["end_sec"]),
            window_count=len(result.embeddings),
            embedding_dim=result.embeddings.shape[1],
            device=result.device,
        )
        resolved = {
            "run_id": run_id,
            "created_at": created_at,
            "source": source,
            "start_sec": request["start_sec"],
            "end_sec": request["end_sec"],
            "selected_channel_ids": result.selected_channel_ids,
            "batch_size": request["batch_size"],
            "device": result.device,
            "model_id": result.profile.id,
            "model_checkpoint_sha256": result.profile.checkpoint_sha256,
            "source_sample_rate_hz": result.source_sample_rate_hz,
            "target_sample_rate_hz": result.profile.target_sample_rate_hz,
            "window_sec": result.profile.window_sec,
            "hop_sec": result.profile.hop_sec,
            "normalization": result.profile.normalization,
            "embedding_dim": result.profile.embedding_dim,
            "window_count": len(result.embeddings),
            "embedding_norm_min": float(norms.min()),
            "embedding_norm_max": float(norms.max()),
            "summary": summary.model_dump(),
        }

        arrays_path = self._arrays_path(run_id)
        temporary_arrays = arrays_path.with_suffix(".tmp")
        with self._lock:
            with temporary_arrays.open("wb") as file:
                np.savez_compressed(
                    file,
                    embeddings=result.embeddings,
                    window_start_sec=result.window_start_sec,
                    window_end_sec=result.window_end_sec,
                    pca_2d=result.pca_2d,
                    adjacent_cosine_similarity=result.adjacent_cosine_similarity,
                )
            temporary_arrays.replace(arrays_path)
            self._write_json(self._metadata_path(run_id), resolved)
        return self.get(run_id)

    def list(self) -> list[RunSummary]:
        summaries: list[RunSummary] = []
        with self._lock:
            for path in self.directory.glob("*.json"):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    summaries.append(RunSummary.model_validate(payload["summary"]))
                except (OSError, KeyError, json.JSONDecodeError, ValueError):
                    continue
        return sorted(summaries, key=lambda item: item.created_at, reverse=True)

    def get(self, run_id: str) -> StoredRunData:
        metadata = self.metadata(run_id)
        with self._lock, np.load(self.arrays_path(run_id), allow_pickle=False) as arrays:
            embeddings = np.asarray(arrays["embeddings"])
            return StoredRunData(
                run_id=run_id,
                window_count=len(embeddings),
                embedding_dim=embeddings.shape[1],
                device=str(metadata["device"]),
                source_sample_rate_hz=float(metadata["source_sample_rate_hz"]),
                model_sample_rate_hz=float(metadata["target_sample_rate_hz"]),
                selected_channel_ids=[str(value) for value in metadata["selected_channel_ids"]],
                window_start_sec=np.asarray(arrays["window_start_sec"]).tolist(),
                pca_2d=np.asarray(arrays["pca_2d"]).tolist(),
                adjacent_cosine_similarity=np.asarray(
                    arrays["adjacent_cosine_similarity"]
                ).tolist(),
                embedding_norm_min=float(metadata["embedding_norm_min"]),
                embedding_norm_max=float(metadata["embedding_norm_max"]),
                downstream=metadata.get("downstream"),
            )

    def metadata(self, run_id: str) -> dict[str, Any]:
        path = self._metadata_path(run_id)
        if not path.is_file():
            raise FileNotFoundError(f"Run {run_id!r} was not found.")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Run {run_id!r} metadata is unreadable.") from error

    def arrays_path(self, run_id: str) -> Path:
        path = self._arrays_path(run_id)
        if not path.is_file():
            raise FileNotFoundError(f"Run {run_id!r} arrays were not found.")
        return path

    def metadata_path(self, run_id: str) -> Path:
        path = self._metadata_path(run_id)
        if not path.is_file():
            raise FileNotFoundError(f"Run {run_id!r} metadata was not found.")
        return path

    def embedding_arrays(self, run_id: str) -> tuple[np.ndarray, np.ndarray]:
        with self._lock, np.load(self.arrays_path(run_id), allow_pickle=False) as arrays:
            return (
                np.asarray(arrays["embeddings"], dtype=np.float32),
                np.asarray(arrays["window_start_sec"], dtype=np.float64),
            )

    def save_downstream(self, run_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            metadata = self.metadata(run_id)
            metadata["downstream"] = result
            self._write_json(self._metadata_path(run_id), metadata)

    def delete(self, run_id: str) -> None:
        metadata_path = self._metadata_path(run_id)
        arrays_path = self._arrays_path(run_id)
        with self._lock:
            existing = [path for path in (arrays_path, metadata_path) if path.is_file()]
            if not existing:
                raise FileNotFoundError(f"Run {run_id!r} was not found.")
            for path in existing:
                path.unlink()

    def _metadata_path(self, run_id: str) -> Path:
        self._validate_run_id(run_id)
        return self.directory / f"{run_id}.json"

    def _arrays_path(self, run_id: str) -> Path:
        self._validate_run_id(run_id)
        return self.directory / f"{run_id}.npz"

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        if len(run_id) != 12 or any(character not in "0123456789abcdef" for character in run_id):
            raise ValueError("Invalid run ID.")

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(path)
