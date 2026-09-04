"""Long-running Colab worker: load models once, serve remote jobs over localhost HTTP."""
from __future__ import annotations

import logging
import os
import threading
import uuid
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from video_retrieval.config import get_settings
from video_retrieval.remote.models import RemoteJobRequest, RemoteJobResponse
from video_retrieval.remote.worker import run_request, warm_runtime
from video_retrieval.runtime import AppRuntime

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = "/content/data"
DEFAULT_PORT = 8765
MAX_STORED_JOBS = 64

JobStatus = Literal["queued", "running", "done", "error"]


class HealthResponse(BaseModel):
    ok: bool
    pid: int
    data_dir: str
    runtime_ready: bool


class AsyncJobAccepted(BaseModel):
    job_id: str
    status: Literal["queued"] = "queued"


class AsyncJobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    ok: bool | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_run_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        force=True,
    )
    data_dir = os.environ.get("COLAB_REMOTE_DATA_DIR", DEFAULT_DATA_DIR)
    settings = get_settings(data_dir=data_dir, colab=True)
    logger.info("Warming Colab worker runtime (data_dir=%s)...", data_dir)
    runtime = warm_runtime(settings)
    app.state.settings = settings
    app.state.runtime = runtime
    logger.info("Colab worker ready on pid=%s", os.getpid())
    yield


app = FastAPI(title="video-retrieval Colab worker", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    runtime: AppRuntime | None = getattr(app.state, "runtime", None)
    settings = getattr(app.state, "settings", None)
    return HealthResponse(
        ok=True,
        pid=os.getpid(),
        data_dir=str(settings.data_dir) if settings else DEFAULT_DATA_DIR,
        runtime_ready=runtime is not None and runtime.search is not None,
    )


@app.post("/job", response_model=RemoteJobResponse)
def job(body: dict[str, Any]) -> RemoteJobResponse:
    """Synchronous job (localhost / Colab CLI proxy). Prefer /jobs for tunnels."""
    try:
        request_model = RemoteJobRequest.model_validate(body)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    with _run_lock:
        response = run_request(request_model)
    if not response.ok:
        logger.error("Worker job failed: %s", response.error)
    return response


@app.post("/jobs", response_model=AsyncJobAccepted)
def submit_job(body: dict[str, Any]) -> AsyncJobAccepted:
    """Accept a job and return immediately (avoids Cloudflare ~100s proxy timeout)."""
    try:
        request_model = RemoteJobRequest.model_validate(body)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {"status": "queued", "response": None}
        _prune_jobs_locked()

    thread = threading.Thread(
        target=_execute_job,
        args=(job_id, request_model),
        name=f"worker-job-{job_id[:8]}",
        daemon=True,
    )
    thread.start()
    logger.info("Queued async job %s type=%s", job_id, request_model.job)
    return AsyncJobAccepted(job_id=job_id)


@app.get("/jobs/{job_id}", response_model=AsyncJobStatusResponse)
def get_job(job_id: str) -> AsyncJobStatusResponse:
    with _jobs_lock:
        record = _jobs.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Unknown job_id={job_id}")
        status: JobStatus = record["status"]
        response: RemoteJobResponse | None = record.get("response")

    if response is None:
        return AsyncJobStatusResponse(job_id=job_id, status=status)
    return AsyncJobStatusResponse(
        job_id=job_id,
        status=status,
        ok=response.ok,
        result=response.result,
        error=response.error,
    )


def _execute_job(job_id: str, request: RemoteJobRequest) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "running"
    logger.info("Running async job %s type=%s", job_id, request.job)
    try:
        with _run_lock:
            response = run_request(request)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Async job %s crashed", job_id)
        response = RemoteJobResponse(ok=False, error=str(exc))
    status: JobStatus = "done" if response.ok else "error"
    if not response.ok:
        logger.error("Async job %s failed: %s", job_id, response.error)
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = status
            _jobs[job_id]["response"] = response


def _prune_jobs_locked() -> None:
    """Drop oldest finished jobs when the store grows too large."""
    if len(_jobs) <= MAX_STORED_JOBS:
        return
    finished = [
        jid
        for jid, rec in _jobs.items()
        if rec.get("status") in {"done", "error"}
    ]
    overflow = len(_jobs) - MAX_STORED_JOBS
    for jid in finished[: max(0, overflow)]:
        _jobs.pop(jid, None)
