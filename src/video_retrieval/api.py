from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from video_retrieval.config import Settings, get_settings
from video_retrieval.pipeline.indexer import VideoIndexer
from video_retrieval.search.service import SearchService

settings = get_settings()
settings.ensure_dirs()

WEB_DIR = Path(__file__).resolve().parent / "web"

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
    limit: int = Field(default=24, ge=1, le=100)
    vector_name: Literal["siglip"] = "siglip"


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


def _safe_video_id(video_id: str) -> str:
    if "/" in video_id or "\\" in video_id or ".." in video_id or not video_id.strip():
        raise HTTPException(status_code=400, detail="Invalid video_id")
    return video_id.strip()


def _safe_keyframe_file(video_id: str, filename: str) -> Path:
    video_id = _safe_video_id(video_id)
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        raise HTTPException(status_code=400, detail="Unsupported media type")
    root = settings.keyframes_dir.resolve()
    path = (root / video_id / filename).resolve()
    if not path.is_relative_to(root):
        raise HTTPException(status_code=400, detail="Path escapes keyframe root")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Keyframe not found")
    return path


def _safe_video_file(video_id: str) -> Path:
    video_id = _safe_video_id(video_id)
    root = settings.videos_dir.resolve()
    for suffix in (".mp4", ".mov", ".mkv", ".webm", ".avi"):
        path = (root / f"{video_id}{suffix}").resolve()
        if not path.is_relative_to(root):
            raise HTTPException(status_code=400, detail="Path escapes video root")
        if path.is_file():
            return path
    raise HTTPException(status_code=404, detail="Video not found")


def _image_url_for_hit(hit: dict[str, Any]) -> str | None:
    keyframe_path = hit.get("keyframe_path")
    video_id = hit.get("video_id")
    if not keyframe_path or not video_id:
        return None
    name = Path(str(keyframe_path)).name
    if not name:
        return None
    return f"/media/keyframes/{quote(str(video_id))}/{quote(name)}"


def _video_url_for_hit(hit: dict[str, Any], *, available: dict[str, bool] | None = None) -> str | None:
    video_id = hit.get("video_id")
    if not video_id:
        return None
    key = str(video_id)
    if available is not None and key in available:
        return f"/media/videos/{quote(key)}" if available[key] else None
    try:
        _safe_video_file(key)
        exists = True
    except HTTPException:
        exists = False
    if available is not None:
        available[key] = exists
    return f"/media/videos/{quote(key)}" if exists else None


def _enrich_search_payload(payload: dict[str, Any]) -> dict[str, Any]:
    hits = []
    video_available: dict[str, bool] = {}
    for hit in payload.get("hits") or []:
        item = dict(hit)
        item["image_url"] = _image_url_for_hit(item)
        item["video_url"] = _video_url_for_hit(item, available=video_available)
        hits.append(item)
    payload["hits"] = hits
    return payload


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
    return _enrich_search_payload(response.model_dump(mode="json"))


class CaptureRequest(BaseModel):
    t: float = Field(default=0.0, ge=0.0, description="Timestamp in seconds")


@app.get("/media/keyframes/{video_id}/{filename}")
def get_keyframe(video_id: str, filename: str):
    path = _safe_keyframe_file(video_id, filename)
    return FileResponse(path, media_type="image/jpeg")


@app.get("/media/videos/{video_id}")
def get_video(video_id: str):
    path = _safe_video_file(video_id)
    media_types = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".avi": "video/x-msvideo",
    }
    return FileResponse(
        path,
        media_type=media_types.get(path.suffix.lower(), "application/octet-stream"),
        filename=path.name,
    )


@app.post("/media/videos/{video_id}/capture")
def capture_video_frame(video_id: str, body: CaptureRequest):
    """Extract a JPEG at timestamp ``t`` and return frame index + data URL."""
    import base64

    import cv2

    path = _safe_video_file(video_id)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open video")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 1e-3:
            fps = 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_index = int(round(body.t * fps))
        if frame_count > 0:
            frame_index = max(0, min(frame_index, frame_count - 1))
        else:
            frame_index = max(0, frame_index)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise HTTPException(status_code=404, detail="Could not read frame at timestamp")
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            raise HTTPException(status_code=500, detail="Could not encode frame")
        timestamp_sec = frame_index / fps
        b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
        return {
            "video_id": video_id,
            "timestamp_sec": timestamp_sec,
            "frame_index": frame_index,
            "fps": fps,
            "image_data_url": f"data:image/jpeg;base64,{b64}",
            "video_url": f"/media/videos/{quote(video_id)}",
            "source": "user_capture",
            "score": 1.0,
        }
    finally:
        cap.release()


@app.get("/")
def ui_index():
    index = WEB_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(index)


if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
