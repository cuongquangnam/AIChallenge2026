"""Long-running Colab worker: load models once, serve remote jobs over localhost HTTP."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from video_retrieval.config import get_settings
from video_retrieval.remote.models import RemoteJobRequest, RemoteJobResponse
from video_retrieval.remote.worker import run_request, warm_runtime
from video_retrieval.runtime import AppRuntime

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = "/content/data"
DEFAULT_PORT = 8765


class HealthResponse(BaseModel):
    ok: bool
    pid: int
    data_dir: str
    runtime_ready: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    try:
        request_model = RemoteJobRequest.model_validate(body)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    response = run_request(request_model)
    if not response.ok:
        logger.error("Worker job failed: %s", response.error)
    return response
