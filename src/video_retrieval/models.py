from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class FrameRole(str, Enum):
    START = "start"
    MIDDLE = "middle"
    END = "end"


class KeyFrame(BaseModel):
    video_id: str
    shot_index: int
    role: FrameRole
    frame_index: int
    timestamp_sec: float
    path: Path


class Shot(BaseModel):
    video_id: str
    shot_index: int
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    keyframes: list[KeyFrame] = Field(default_factory=list)


class AudioTrack(BaseModel):
    video_id: str
    path: Path
    sample_rate: int = 16000
    duration_sec: float | None = None


class VisualEmbedding(BaseModel):
    keyframe: KeyFrame
    siglip: list[float]
    beit3: list[float]


class TextDocument(BaseModel):
    doc_id: str
    video_id: str
    source: str  # ocr | asr
    text: str
    shot_index: int | None = None
    frame_index: int | None = None
    role: FrameRole | None = None
    start_sec: float | None = None
    end_sec: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IndexResult(BaseModel):
    video_id: str
    video_path: Path
    num_shots: int
    num_keyframes: int
    num_visual_points: int
    num_text_docs: int
    audio_path: Path | None = None


class SearchHit(BaseModel):
    video_id: str
    score: float
    source: str
    shot_index: int | None = None
    frame_index: int | None = None
    role: FrameRole | None = None
    timestamp_sec: float | None = None
    text: str | None = None
    keyframe_path: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    mode: str
    hits: list[SearchHit]
