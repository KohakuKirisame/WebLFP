from __future__ import annotations

import os
import threading
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .dialogs import select_recording_file
from .downstream_schema import SpikeTypeDecodeResult
from .inference_jobs import InferenceCancelled, InferenceJobStatus, InferenceJobStore
from .preprocessing import downsample_preview, robust_zscore_channels
from .profile import default_model_dir, load_model_profile, project_root, verify_checkpoint
from .recording import (
    RecordingMetadata,
    RecordingStreamOptions,
    SourceConfig,
    list_recording_streams,
    open_recording,
)
from .run_history import RunStore, RunSummary, StoredRunData
from .segment_cache import TraceSegmentCache
from .settings import (
    detect_system,
    get_pytorch_option,
    install_status,
    pytorch_installed,
    pytorch_options,
    start_pytorch_install,
)


class InspectRequest(BaseModel):
    source: SourceConfig


class PreviewRequest(BaseModel):
    source: SourceConfig
    start_sec: float = Field(default=0, ge=0)
    duration_sec: float = Field(default=2, gt=0)
    channel_ids: list[str] | None = None
    max_points: int = Field(default=1200, ge=100, le=5000)


class InferenceRequest(BaseModel):
    source: SourceConfig
    start_sec: float = Field(default=0, ge=0)
    end_sec: float = Field(gt=0)
    channel_ids: list[str] | None = None
    batch_size: int = Field(default=32, ge=1, le=512)
    device: Literal["auto", "cpu", "cuda"] = "auto"


class PyTorchInstallRequest(BaseModel):
    option_id: str
    confirmed: bool


class SpikeTypeDecodeRequest(BaseModel):
    batch_size: int = Field(default=32, ge=1, le=512)
    device: Literal["auto", "cpu", "cuda"] = "auto"


class DeleteRunResponse(BaseModel):
    run_id: str
    deleted: bool


class PreviewResponse(BaseModel):
    metadata: RecordingMetadata
    channel_ids: list[str]
    times_sec: list[float]
    raw_traces: list[list[float]]
    normalized_traces: list[list[float]]


class InferenceResponse(BaseModel):
    run_id: str
    window_count: int
    embedding_dim: int
    device: str
    source_sample_rate_hz: float
    model_sample_rate_hz: float
    selected_channel_ids: list[str]
    window_start_sec: list[float]
    umap_3d: list[list[float]]
    umap_window_start_sec: list[float]
    adjacent_cosine_similarity: list[float]
    embedding_norm_min: float
    embedding_norm_max: float
    downstream: SpikeTypeDecodeResult | None = None
    embeddings_download_url: str
    run_download_url: str


results = RunStore()
inference_jobs = InferenceJobStore()
trace_segments = TraceSegmentCache()
app = FastAPI(title="WebLFP", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _bad_request(error: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(error))


def _inference_response(data: StoredRunData) -> InferenceResponse:
    return InferenceResponse(
        **data.model_dump(),
        embeddings_download_url=f"/api/results/{data.run_id}/embeddings.npz",
        run_download_url=f"/api/results/{data.run_id}/run.json",
    )


def _run_and_store(
    request: InferenceRequest,
    progress_callback: Any = None,
) -> InferenceResponse:
    try:
        from .inference import run_inference
    except ModuleNotFoundError as error:
        if error.name != "torch" and not (error.name or "").startswith("torch."):
            raise
        raise RuntimeError(
            "PyTorch is not installed. Configure a runtime on the Settings page first."
        ) from error

    result = run_inference(
        source=request.source,
        start_sec=request.start_sec,
        end_sec=request.end_sec,
        channel_ids=request.channel_ids,
        batch_size=request.batch_size,
        device_choice=request.device,
        progress_callback=progress_callback,
        segment_cache=trace_segments,
    )
    stored = results.add(result, request.model_dump())
    return _inference_response(stored)


def _execute_inference_job(job_id: str, request: InferenceRequest) -> None:
    try:
        response = _run_and_store(
            request,
            progress_callback=lambda progress, message: inference_jobs.report(
                job_id,
                progress * 100,
                message,
            ),
        )
        inference_jobs.complete(job_id, response.model_dump())
    except InferenceCancelled:
        inference_jobs.mark_cancelled(job_id)
    except (OSError, ValueError, RuntimeError) as error:
        inference_jobs.fail(job_id, str(error))


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/formats")
def formats() -> dict[str, list[str]]:
    return {
        "formats": [
            "auto",
            "npy",
            "npz",
            "mat",
            "binary",
            "spikeglx",
            "openephys",
            "intan",
            "plexon",
            "plexon2",
            "alphaomega",
            "nwb",
        ]
    }


@app.post("/api/dialogs/recording-file")
def recording_file_dialog() -> dict[str, str | None]:
    try:
        selected = select_recording_file()
    except RuntimeError as error:
        raise _bad_request(error) from error
    return {"path": str(selected) if selected else None}


@app.get("/api/model")
def model_info() -> dict[str, Any]:
    try:
        profile = load_model_profile(default_model_dir())
        verify_checkpoint(default_model_dir())
        return {**profile.model_dump(), "checkpoint_available": True}
    except (OSError, ValueError) as error:
        raise _bad_request(error) from error


@app.get("/api/settings/system")
def system_info(refresh: bool = False) -> dict[str, Any]:
    return detect_system(refresh=refresh)


@app.get("/api/settings/pytorch-status")
def pytorch_status() -> dict[str, bool]:
    return {"installed": pytorch_installed()}


@app.get("/api/settings/pytorch-options")
def available_pytorch_options() -> dict[str, Any]:
    system = detect_system()
    return {
        "minimum_cuda_version": "13.0",
        "options": [option.model_dump() for option in pytorch_options(system)],
    }


@app.get("/api/settings/pytorch-install")
def pytorch_install_status() -> dict[str, Any]:
    return install_status() or {
        "state": "idle",
        "message": "No PyTorch installation has been started.",
    }


@app.post("/api/settings/pytorch-install")
def install_pytorch(request: PyTorchInstallRequest) -> dict[str, Any]:
    if not request.confirmed:
        raise HTTPException(status_code=400, detail="Explicit installation confirmation is required.")
    try:
        option = get_pytorch_option(request.option_id)
        status = start_pytorch_install(option)
    except (OSError, ValueError, RuntimeError) as error:
        raise _bad_request(error) from error

    threading.Timer(1.0, lambda: os._exit(0)).start()
    return {
        **status,
        "restart_expected": True,
        "message": "Installation scheduled. WebLFP will stop, install PyTorch, and restart.",
    }


@app.post("/api/inspect", response_model=RecordingMetadata)
def inspect_recording(request: InspectRequest) -> RecordingMetadata:
    try:
        return open_recording(request.source).metadata
    except (OSError, ValueError, RuntimeError) as error:
        raise _bad_request(error) from error


@app.post("/api/streams", response_model=RecordingStreamOptions)
def recording_streams(request: InspectRequest) -> RecordingStreamOptions:
    try:
        return list_recording_streams(request.source)
    except (OSError, ValueError, RuntimeError) as error:
        raise _bad_request(error) from error


@app.post("/api/preview", response_model=PreviewResponse)
def preview_recording(request: PreviewRequest) -> PreviewResponse:
    try:
        segment, _ = trace_segments.get(
            source=request.source,
            start_sec=request.start_sec,
            end_sec=request.start_sec + request.duration_sec,
            channel_ids=request.channel_ids,
            default_channel_count=4,
            clamp_end=True,
        )
        metadata = segment.metadata
        selected = segment.channel_ids
        traces = segment.traces
        normalized = robust_zscore_channels(traces)
        times, raw_preview = downsample_preview(
            traces,
            request.start_sec,
            metadata.sampling_rate_hz,
            request.max_points,
        )
        _, normalized_preview = downsample_preview(
            normalized,
            request.start_sec,
            metadata.sampling_rate_hz,
            request.max_points,
        )
        return PreviewResponse(
            metadata=metadata,
            channel_ids=selected,
            times_sec=times.tolist(),
            raw_traces=raw_preview.tolist(),
            normalized_traces=normalized_preview.tolist(),
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise _bad_request(error) from error


@app.post("/api/infer", response_model=InferenceResponse)
def infer(request: InferenceRequest) -> InferenceResponse:
    try:
        return _run_and_store(request)
    except (OSError, ValueError, RuntimeError) as error:
        raise _bad_request(error) from error


@app.post("/api/inference-jobs", response_model=InferenceJobStatus)
def create_inference_job(request: InferenceRequest) -> InferenceJobStatus:
    try:
        job = inference_jobs.create()
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    threading.Thread(
        target=_execute_inference_job,
        args=(job.job_id, request),
        daemon=True,
    ).start()
    return job


@app.get("/api/inference-jobs/{job_id}", response_model=InferenceJobStatus)
def inference_job_status(job_id: str) -> InferenceJobStatus:
    try:
        return inference_jobs.get(job_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.delete("/api/inference-jobs/{job_id}", response_model=InferenceJobStatus)
def cancel_inference_job(job_id: str) -> InferenceJobStatus:
    try:
        return inference_jobs.cancel(job_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/results", response_model=list[RunSummary])
def result_history() -> list[RunSummary]:
    return results.list()


@app.delete("/api/results/{run_id}", response_model=DeleteRunResponse)
def delete_result(run_id: str) -> DeleteRunResponse:
    try:
        results.delete(run_id)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except OSError as error:
        raise _bad_request(error) from error
    return DeleteRunResponse(run_id=run_id, deleted=True)


@app.get("/api/results/{run_id}", response_model=InferenceResponse)
def result_detail(run_id: str) -> InferenceResponse:
    try:
        return _inference_response(results.get(run_id))
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/api/results/{run_id}/type-decode", response_model=SpikeTypeDecodeResult)
def decode_result(run_id: str, request: SpikeTypeDecodeRequest) -> SpikeTypeDecodeResult:
    try:
        try:
            from .type_decoder import decode_spike_types
        except ModuleNotFoundError as error:
            if error.name != "torch" and not (error.name or "").startswith("torch."):
                raise
            raise RuntimeError(
                "PyTorch is not installed. Configure a runtime on the Settings page first."
            ) from error

        features, window_start_sec = results.embedding_arrays(run_id)
        result = decode_spike_types(
            features=features,
            window_start_sec=window_start_sec,
            batch_size=request.batch_size,
            device_choice=request.device,
        )
        results.save_downstream(run_id, result.model_dump())
        return result
    except (OSError, ValueError, RuntimeError) as error:
        raise _bad_request(error) from error


@app.get("/api/results/{run_id}/embeddings.npz")
def download_embeddings(run_id: str) -> FileResponse:
    try:
        path = results.arrays_path(run_id)
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return FileResponse(path, filename=f"{run_id}-embeddings.npz")


@app.get("/api/results/{run_id}/run.json")
def download_run(run_id: str) -> FileResponse:
    try:
        path = results.metadata_path(run_id)
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return FileResponse(path, filename=f"{run_id}-run.json", media_type="application/json")


frontend_dist = project_root() / "frontend" / "dist"
if frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")


def main() -> None:
    uvicorn.run("weblfp.api:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
