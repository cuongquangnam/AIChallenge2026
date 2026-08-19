from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from video_retrieval.config import get_settings
from video_retrieval.pipeline.indexer import VideoIndexer
from video_retrieval.search.service import SearchService

settings = get_settings()
settings.ensure_dirs()

app = FastAPI(
    title="Video Retrieval API",
    description="Offline indexing + visual/textual search over video keyframes",
    version="0.1.0",
)


class IndexRequest(BaseModel):
    path: str
    video_id: str | None = None


class SearchRequest(BaseModel):
    query: str
    mode: Literal["text", "visual", "hybrid"] = "hybrid"
    limit: int = Field(default=10, ge=1, le=100)
    vector_name: Literal["siglip", "beit3"] = "siglip"
    source: Literal["ocr", "asr"] | None = None
    video_id: str | None = None


class Task2CandidatesRequest(BaseModel):
    """Controls retrieval of VLM evidence for the award-acceptance question."""

    video_id: str | None = None
    candidates_per_query: int = Field(default=20, ge=1, le=100)
    group_limit: int = Field(default=10, ge=1, le=20)
    max_gap_sec: float = Field(default=10.0, ge=0.0, le=120.0)
    max_gap_frames: int = Field(default=10, ge=0, le=1000)
    context_radius_frames: int = Field(default=5, ge=0, le=100)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/index")
def index_video(body: IndexRequest):
    indexer = VideoIndexer(settings)
    try:
        result = indexer.index_video(Path(body.path), video_id=body.video_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result.model_dump(mode="json")


@app.post("/index/upload")
async def index_upload(file: UploadFile = File(...), video_id: str | None = None):
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    dest = settings.videos_dir / f"{video_id or Path(file.filename or 'upload').stem}{suffix}"
    content = await file.read()
    dest.write_bytes(content)
    indexer = VideoIndexer(settings)
    result = indexer.index_video(dest, video_id=video_id)
    return result.model_dump(mode="json")


@app.post("/search")
def search(body: SearchRequest):
    service = SearchService(settings)
    try:
        if body.mode == "text":
            if body.source or body.video_id:
                response = service.search_text_filtered(
                    body.query,
                    limit=body.limit,
                    source=body.source,
                    video_id=body.video_id,
                )
            else:
                response = service.search_text(body.query, limit=body.limit)
        elif body.mode == "visual":
            response = service.search_visual_text(
                body.query,
                limit=body.limit,
                vector_name=body.vector_name,
                video_id=body.video_id,
            )
        else:
            if body.source or body.video_id:
                response = service.search_hybrid_filtered(
                    body.query,
                    limit=body.limit,
                    source=body.source,
                    video_id=body.video_id,
                )
            else:
                response = service.search_hybrid(body.query, limit=body.limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return response.model_dump(mode="json")


@app.post("/task2/candidates")
def task2_candidates(body: Task2CandidatesRequest):
    """Return ranked frame windows to send to a VLM for Task 2 reasoning."""
    service = SearchService(settings)
    try:
        response = service.retrieve_task2_candidates(
            video_id=body.video_id,
            candidates_per_query=body.candidates_per_query,
            group_limit=body.group_limit,
            max_gap_sec=body.max_gap_sec,
            max_gap_frames=body.max_gap_frames,
            context_radius_frames=body.context_radius_frames,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return response.model_dump(mode="json")
