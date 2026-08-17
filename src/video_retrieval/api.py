from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from video_retrieval.config import Settings, get_settings
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
    data_dir: str | None = None
    stages: list[Literal["visual", "ocr", "asr"]] | None = None
    only: Literal["visual", "ocr", "asr"] | None = None
    reuse_extract: bool = True


class SearchRequest(BaseModel):
    query: str
    mode: Literal["visual", "asr", "ocr", "mixed"] = "mixed"
    limit: int = Field(default=10, ge=1, le=100)
    vector_name: Literal["siglip", "beit3"] = "siglip"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _index_settings(data_dir: str | None = None) -> Settings:
    return get_settings(data_dir=data_dir) if data_dir else settings


def _selected_stages(
    only: str | None = None,
    stages: list[str] | None = None,
) -> list[str] | None:
    if only and stages:
        raise HTTPException(status_code=400, detail="Use only or stages, not both.")
    if only:
        return [only]
    return stages


@app.post("/index")
def index_video(body: IndexRequest):
    indexer = VideoIndexer(_index_settings(body.data_dir))
    try:
        result = indexer.index_video(
            Path(body.path),
            video_id=body.video_id,
            stages=_selected_stages(body.only, body.stages),
            reuse_extract=body.reuse_extract,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result.model_dump(mode="json")


@app.post("/index/upload")
async def index_upload(
    file: UploadFile = File(...),
    video_id: str | None = None,
    data_dir: str | None = None,
    only: Literal["visual", "ocr", "asr"] | None = None,
    stages: str | None = None,
    reuse_extract: bool = True,
):
    index_settings = _index_settings(data_dir)
    index_settings.ensure_dirs()
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    dest = index_settings.videos_dir / f"{video_id or Path(file.filename or 'upload').stem}{suffix}"
    content = await file.read()
    dest.write_bytes(content)
    selected = None
    if stages:
        selected = [part.strip() for part in stages.split(",") if part.strip()]
    indexer = VideoIndexer(index_settings)
    result = indexer.index_video(
        dest,
        video_id=video_id,
        stages=_selected_stages(only, selected),
        reuse_extract=reuse_extract,
    )
    index_settings = _index_settings(data_dir)
    index_settings.ensure_dirs()
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    dest = index_settings.videos_dir / f"{video_id or Path(file.filename or 'upload').stem}{suffix}"
    content = await file.read()
    dest.write_bytes(content)
    indexer = VideoIndexer(index_settings)
    result = indexer.index_video(dest, video_id=video_id)
    return result.model_dump(mode="json")


@app.post("/search")
def search(body: SearchRequest):
    service = SearchService(settings)
    if body.mode == "ocr":
        response = service.search_ocr(body.query, limit=body.limit)
    elif body.mode == "asr":
        response = service.search_asr(body.query, limit=body.limit)
    elif body.mode == "visual":
        response = service.search_visual(
            body.query, limit=body.limit, vector_name=body.vector_name
        )
    else:
        response = service.search_mixed(
            body.query, limit=body.limit, vector_name=body.vector_name
        )
    return response.model_dump(mode="json")
