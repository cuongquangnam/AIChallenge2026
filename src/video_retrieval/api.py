from __future__ import annotations

import csv
import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from video_retrieval.config import Settings, get_settings
from video_retrieval.events.export import chains_to_csv_lines
from video_retrieval.pipeline.indexer import VideoIndexer
from video_retrieval.qa.frames import QAFrameExtractionError, VideoNotFoundForQAError
from video_retrieval.qa.llm import (
    InvalidQAModelResponseError,
    QAModelNotConfiguredError,
)
from video_retrieval.runtime import get_runtime, init_runtime

logger = logging.getLogger(__name__)

settings = get_settings()
settings.ensure_dirs()

WEB_DIR = Path(__file__).resolve().parent / "web"


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    from video_retrieval.runtime import _runtime

    if _runtime is None:
        init_runtime(settings)
    yield


app = FastAPI(
    title="Video Retrieval API",
    description="Search UI + KIS / QA / TRAKE task pages for video keyframes",
    version="0.2.0",
    lifespan=_lifespan,
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
    vector_name: Literal["siglip", "beit3"] = "siglip"


class KisRequest(BaseModel):
    query: str
    mode: Literal["visual", "asr", "ocr", "mixed"] = "mixed"
    limit: int = Field(default=100, ge=1, le=100)
    vector_name: Literal["siglip", "beit3"] = "siglip"
    query_id: str = ""


class QARequest(BaseModel):
    question: str
    limit: int = Field(default=24, ge=1, le=100)
    group_count: int | None = Field(default=None, ge=1, le=20)
    frame_radius: int | None = Field(default=None, ge=0, le=30)


class TrakeRequest(BaseModel):
    query: str
    top_chains: int = Field(default=24, ge=1, le=100)


class CaptureRequest(BaseModel):
    t: float = Field(default=0.0, ge=0.0, description="Timestamp in seconds")


class FrameBatchRequest(BaseModel):
    start_frame: int = Field(default=0, ge=0, description="First frame index (inclusive)")
    count: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of consecutive frames to return (max 100)",
    )


class SubmissionRow(BaseModel):
    video_id: str
    frame_index: int = Field(ge=0)


class ResolveSubmissionRequest(BaseModel):
    """Resolve KIS CSV rows (``video_id,frame_idx``) into viewable frames."""

    rows: list[SubmissionRow] | None = None
    csv_text: str | None = Field(
        default=None,
        description="Raw CSV text with video_id,frame_idx (no header required)",
    )
    query_id: str = ""


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
    return result.model_dump(mode="json")


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
    if keyframe_path and video_id:
        name = Path(str(keyframe_path)).name
        if name:
            return f"/media/keyframes/{quote(str(video_id))}/{quote(name)}"
    return None


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


def _enrich_chains(payload: dict[str, Any]) -> dict[str, Any]:
    video_available: dict[str, bool] = {}
    for chain in payload.get("chains") or []:
        video_id = chain.get("video_id")
        chain["video_url"] = _video_url_for_hit(
            {"video_id": video_id}, available=video_available
        )
        for event in chain.get("events") or []:
            event["video_id"] = video_id
            event["image_url"] = _image_url_for_hit(
                {
                    "video_id": video_id,
                    "keyframe_path": event.get("keyframe_path"),
                }
            )
            event["video_url"] = chain.get("video_url")
    return payload


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


def _rows_to_csv(rows: list[tuple[str, int]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    for video_id, frame_idx in rows:
        writer.writerow([video_id, frame_idx])
    return buf.getvalue()


@app.post("/search")
def search(body: SearchRequest):
    service = get_runtime().search
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


@app.post("/kis")
def kis_search(body: KisRequest):
    try:
        result = get_runtime().kis.run(body.query, limit=body.limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("KIS backend error: %s", exc)
        raise HTTPException(status_code=502, detail=f"KIS backend error: {exc}") from exc

    payload = result.model_dump(mode="json")
    payload = _enrich_chains(payload)
    payload["hits"] = _enrich_search_payload({"hits": payload.get("hits") or []})["hits"]
    payload["query_id"] = (body.query_id or "").strip()
    rows = result.submission_rows
    payload["submission_rows"] = [
        {"video_id": video_id, "frame_index": frame_idx} for video_id, frame_idx in rows
    ]
    payload["csv_text"] = _rows_to_csv(rows)
    return payload


@app.post("/qa")
def answer_question(body: QARequest):
    try:
        result = get_runtime().qa.answer(
            body.question,
            limit=body.limit,
            frame_radius=body.frame_radius,
        )
    except QAModelNotConfiguredError as exc:
        logger.error("QA model not configured: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except VideoNotFoundForQAError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (QAFrameExtractionError, InvalidQAModelResponseError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("QA backend error: %s", exc)
        raise HTTPException(status_code=502, detail=f"QA backend error: {exc}") from exc

    payload = result.model_dump(mode="json")
    video_available: dict[str, bool] = {}
    enriched_results = []
    for item in payload.get("results") or []:
        chain = dict(item.get("chain") or {})
        video_id = chain.get("video_id")
        video_url = _video_url_for_hit({"video_id": video_id}, available=video_available)
        chain["video_url"] = video_url
        for event in chain.get("events") or []:
            event["video_id"] = video_id
            event["image_url"] = _image_url_for_hit(
                {
                    "video_id": video_id,
                    "keyframe_path": event.get("keyframe_path"),
                }
            )
            event["video_url"] = video_url
            if event.get("event_id") == item.get("questioned_event_id"):
                event["is_question_target"] = True
        item = dict(item)
        item["chain"] = chain
        enriched_results.append(item)
    payload["results"] = enriched_results

    enriched_hits = []
    for hit in payload.get("hits") or []:
        item = dict(hit)
        item["video_url"] = _video_url_for_hit(
            {"video_id": item.get("video_id")}, available=video_available
        )
        item["image_url"] = _image_url_for_hit(
            {
                "video_id": item.get("video_id"),
                "keyframe_path": _keyframe_for_frame(
                    enriched_results,
                    item.get("video_id"),
                    item.get("frame_id"),
                ),
            }
        )
        enriched_hits.append(item)
    payload["hits"] = enriched_hits
    payload["video_url"] = _video_url_for_hit(
        {"video_id": result.video_id}, available=video_available
    )
    payload["csv_text"] = _qa_rows_to_csv(
        [(h.video_id, h.frame_id, h.answer) for h in result.hits]
    )
    payload["evidence_hit"] = enriched_hits[0] if enriched_hits else {
        "video_id": result.video_id,
        "frame_index": result.frame_id,
        "frame_id": result.frame_id,
        "score": 1.0,
        "source": "qa",
        "video_url": payload["video_url"],
        "answer": result.answer,
    }
    return payload


def _keyframe_for_frame(
    results: list[dict[str, Any]],
    video_id: str | None,
    frame_id: int | None,
) -> str | None:
    if not video_id or frame_id is None:
        return None
    for item in results:
        chain = item.get("chain") or {}
        if chain.get("video_id") != video_id:
            continue
        for event in chain.get("events") or []:
            if event.get("frame_index") == frame_id:
                return event.get("keyframe_path")
    return None


def _qa_rows_to_csv(rows: list[tuple[str, int, str]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    for video_id, frame_id, answer in rows:
        writer.writerow([video_id, frame_id, answer])
    return buf.getvalue()


@app.post("/trake")
def trake_search(body: TrakeRequest):
    try:
        result = get_runtime().trake.run(body.query, top_chains=body.top_chains)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("TRAKE backend error: %s", exc)
        raise HTTPException(status_code=502, detail=f"TRAKE backend error: {exc}") from exc

    payload = result.model_dump(mode="json")
    payload = _enrich_chains(payload)
    csv_lines = chains_to_csv_lines(result.chains)
    if csv_lines:
        payload["csv_text"] = "\n".join(csv_lines) + "\n"
        payload["csv_row"] = csv_lines[0]
    return payload


@app.get("/media/qa-frames")
def get_qa_frame(path: str):
    """Serve a temporary QA evidence JPEG under data/qa_frames only."""
    root = (settings.data_dir / "qa_frames").resolve()
    try:
        resolved = Path(path).expanduser().resolve()
    except OSError as exc:
        raise HTTPException(status_code=400, detail="Invalid path") from exc
    if not resolved.is_relative_to(root):
        raise HTTPException(status_code=400, detail="Path escapes qa_frames root")
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="QA frame not found")
    if not resolved.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(status_code=400, detail="Unsupported media type")
    return FileResponse(resolved, media_type="image/jpeg")


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


_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def _series_of(video_id: str) -> str:
    stem = video_id.strip()
    if "_V" in stem:
        return stem.split("_V", 1)[0] or stem
    if "_" in stem:
        return stem.split("_", 1)[0] or stem
    return stem or "Other"


def _poster_url(video_id: str) -> str | None:
    folder = (settings.keyframes_dir / video_id).resolve()
    root = settings.keyframes_dir.resolve()
    if not folder.is_dir() or not folder.is_relative_to(root):
        return None
    for name in (
        "shot_0000_middle.jpg",
        "shot_0001_middle.jpg",
        "shot_0000_start.jpg",
        "shot_0000_end.jpg",
    ):
        path = folder / name
        if path.is_file():
            return f"/media/keyframes/{quote(video_id)}/{quote(name)}"
    return None


def _list_video_entries(
    *,
    q: str | None = None,
    series: str | None = None,
) -> list[dict[str, Any]]:
    root = settings.videos_dir.resolve()
    if not root.is_dir():
        return []
    needle = (q or "").strip().lower()
    series_filter = (series or "").strip()
    entries: list[dict[str, Any]] = []
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in _VIDEO_SUFFIXES:
            continue
        video_id = path.stem
        series_id = _series_of(video_id)
        if series_filter and series_id != series_filter:
            continue
        if needle and needle not in video_id.lower() and needle not in path.name.lower():
            continue
        entries.append(
            {
                "video_id": video_id,
                "series": series_id,
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "video_url": f"/media/videos/{quote(video_id)}",
            }
        )
    return entries


def _group_video_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for item in entries:
        buckets.setdefault(str(item["series"]), []).append(item)
    groups: list[dict[str, Any]] = []
    for series_id in sorted(buckets.keys()):
        items = buckets[series_id]
        poster = None
        for item in items:
            poster = _poster_url(str(item["video_id"]))
            if poster:
                break
        groups.append(
            {
                "series": series_id,
                "count": len(items),
                "poster_url": poster,
            }
        )
    return groups


@app.get("/api/videos/groups")
def list_video_groups(q: str | None = None):
    entries = _list_video_entries(q=q)
    groups = _group_video_entries(entries)
    return {
        "query": q or "",
        "total_videos": len(entries),
        "total_groups": len(groups),
        "groups": groups,
    }


@app.get("/api/videos")
def list_videos(
    q: str | None = None,
    series: str | None = None,
    offset: int = 0,
    limit: int = 48,
):
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be 1..200")
    entries = _list_video_entries(q=q, series=series)
    total = len(entries)
    page = []
    for item in entries[offset : offset + limit]:
        enriched = dict(item)
        enriched["poster_url"] = _poster_url(item["video_id"])
        page.append(enriched)
    return {
        "query": q or "",
        "series": series or "",
        "total": total,
        "offset": offset,
        "limit": limit,
        "videos": page,
    }


def _open_video_capture(video_id: str):
    import cv2

    path = _safe_video_file(video_id)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open video")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-3:
        fps = 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return cap, fps, frame_count


def _encode_frame_jpeg(frame) -> str:
    import base64

    import cv2

    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode frame")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


@app.post("/media/videos/{video_id}/capture")
def capture_video_frame(video_id: str, body: CaptureRequest):
    import cv2

    cap, fps, frame_count = _open_video_capture(video_id)
    try:
        frame_index = int(round(body.t * fps))
        if frame_count > 0:
            frame_index = max(0, min(frame_index, frame_count - 1))
        else:
            frame_index = max(0, frame_index)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise HTTPException(status_code=404, detail="Could not read frame at timestamp")
        b64 = _encode_frame_jpeg(frame)
        timestamp_sec = frame_index / fps
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


def _consecutive_frames_payload(
    video_id: str, *, start_frame: int, count: int
) -> dict[str, Any]:
    import cv2

    if start_frame < 0:
        raise HTTPException(status_code=400, detail="start_frame must be >= 0")
    if count < 1 or count > 100:
        raise HTTPException(status_code=400, detail="count must be 1..100")

    cap, fps, frame_count = _open_video_capture(video_id)
    try:
        empty = {
            "video_id": video_id,
            "start_frame": start_frame,
            "requested_count": count,
            "fps": fps,
            "frame_count": frame_count,
            "video_url": f"/media/videos/{quote(video_id)}",
            "frames": [],
        }
        if frame_count > 0 and start_frame >= frame_count:
            return empty

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames: list[dict[str, Any]] = []
        for offset in range(count):
            frame_index = start_frame + offset
            if frame_count > 0 and frame_index >= frame_count:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            b64 = _encode_frame_jpeg(frame)
            frames.append(
                {
                    "frame_index": frame_index,
                    "timestamp_sec": frame_index / fps,
                    "image_data_url": f"data:image/jpeg;base64,{b64}",
                }
            )
        empty["frames"] = frames
        return empty
    finally:
        cap.release()


@app.get("/media/videos/{video_id}/frames")
def get_consecutive_frames(video_id: str, start_frame: int = 0, count: int = 10):
    return _consecutive_frames_payload(video_id, start_frame=start_frame, count=count)


@app.post("/media/videos/{video_id}/frames")
def post_consecutive_frames(video_id: str, body: FrameBatchRequest):
    return _consecutive_frames_payload(
        video_id, start_frame=body.start_frame, count=body.count
    )


def _parse_submission_csv(csv_text: str) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    reader = csv.reader(io.StringIO(csv_text))
    for line_no, cells in enumerate(reader, start=1):
        if not cells or all(not str(cell).strip() for cell in cells):
            continue
        if len(cells) < 2:
            raise HTTPException(
                status_code=400,
                detail=f"CSV line {line_no}: expected video_id,frame_idx",
            )
        video_id = str(cells[0]).strip()
        frame_raw = str(cells[1]).strip()
        if line_no == 1 and video_id.lower() in {"video_id", "video"} and frame_raw.lower() in {
            "frame_idx",
            "frame_index",
            "frame",
        }:
            continue
        try:
            frame_index = int(float(frame_raw))
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"CSV line {line_no}: invalid frame_idx {frame_raw!r}",
            ) from exc
        if frame_index < 0:
            raise HTTPException(
                status_code=400,
                detail=f"CSV line {line_no}: frame_idx must be >= 0",
            )
        if not video_id or "/" in video_id or "\\" in video_id or ".." in video_id:
            raise HTTPException(
                status_code=400,
                detail=f"CSV line {line_no}: invalid video_id",
            )
        rows.append((video_id, frame_index))
    return rows


def _resolve_submission_rows(
    rows: list[tuple[str, int]], *, query_id: str = ""
) -> dict[str, Any]:
    import cv2

    if not rows:
        raise HTTPException(status_code=400, detail="No submission rows to resolve")
    if len(rows) > 100:
        raise HTTPException(status_code=400, detail="At most 100 rows allowed")

    decoded: dict[tuple[str, int], dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    by_video: dict[str, list[int]] = {}
    for video_id, frame_index in rows:
        by_video.setdefault(video_id, []).append(frame_index)

    for video_id, indices in by_video.items():
        unique = sorted(set(indices))
        try:
            cap, fps, frame_count = _open_video_capture(video_id)
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else "video unavailable"
            for frame_index in unique:
                errors.append(
                    {
                        "video_id": video_id,
                        "frame_index": frame_index,
                        "detail": detail,
                    }
                )
            continue
        try:
            video_url = f"/media/videos/{quote(video_id)}"
            for frame_index in unique:
                if frame_count > 0 and frame_index >= frame_count:
                    errors.append(
                        {
                            "video_id": video_id,
                            "frame_index": frame_index,
                            "detail": f"frame_index >= frame_count ({frame_count})",
                        }
                    )
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if not ok or frame is None:
                    errors.append(
                        {
                            "video_id": video_id,
                            "frame_index": frame_index,
                            "detail": "could not read frame",
                        }
                    )
                    continue
                b64 = _encode_frame_jpeg(frame)
                decoded[(video_id, frame_index)] = {
                    "video_id": video_id,
                    "frame_index": frame_index,
                    "timestamp_sec": frame_index / fps,
                    "fps": fps,
                    "image_data_url": f"data:image/jpeg;base64,{b64}",
                    "image_url": f"data:image/jpeg;base64,{b64}",
                    "video_url": video_url,
                    "source": "csv_import",
                    "score": 1.0,
                    "role": "csv",
                }
        finally:
            cap.release()

    hits: list[dict[str, Any]] = []
    for video_id, frame_index in rows:
        hit = decoded.get((video_id, frame_index))
        if hit is not None:
            hits.append(dict(hit))

    return {
        "query": query_id or "csv_import",
        "mode": "csv",
        "query_id": query_id,
        "total_rows": len(rows),
        "resolved": len(hits),
        "errors": errors,
        "hits": hits,
        "plan": None,
    }


@app.post("/api/submission/frames")
def resolve_submission_frames(body: ResolveSubmissionRequest):
    if body.rows is not None:
        rows = [(item.video_id.strip(), int(item.frame_index)) for item in body.rows]
        for video_id, frame_index in rows:
            if not video_id or "/" in video_id or "\\" in video_id or ".." in video_id:
                raise HTTPException(status_code=400, detail=f"Invalid video_id: {video_id!r}")
            if frame_index < 0:
                raise HTTPException(status_code=400, detail="frame_index must be >= 0")
    elif body.csv_text is not None:
        rows = _parse_submission_csv(body.csv_text)
    else:
        raise HTTPException(status_code=400, detail="Provide rows or csv_text")
    return _resolve_submission_rows(rows, query_id=(body.query_id or "").strip())


def _ui_page(name: str):
    page = WEB_DIR / name
    if not page.is_file():
        raise HTTPException(status_code=404, detail=f"UI not found: {name}")
    return FileResponse(page)


@app.get("/")
def ui_index():
    return _ui_page("index.html")


@app.get("/videos")
def ui_videos():
    return _ui_page("videos.html")


@app.get("/kis")
def ui_kis():
    return _ui_page("kis.html")


@app.get("/qa")
def ui_qa():
    return _ui_page("qa.html")


@app.get("/trake")
def ui_trake():
    return _ui_page("trake.html")


if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
