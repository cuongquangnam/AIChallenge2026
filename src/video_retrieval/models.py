from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

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


class ObjectDetection(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox_xyxy: tuple[float, float, float, float]


class FrameObjectDetections(BaseModel):
    keyframe: KeyFrame
    detections: list[ObjectDetection] = Field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for detection in self.detections:
            counts[detection.label] = counts.get(detection.label, 0) + 1
        return counts


class ObjectRequirement(BaseModel):
    label: str
    min_count: int = Field(default=1, ge=1)


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
    num_object_detections: int = 0
    audio_path: Path | None = None
    stages: list[str] = Field(default_factory=list)


class QueryPlan(BaseModel):
    """LLM-extracted search strings for each retrieval channel."""

    ocr: str = ""
    asr: str = ""
    visual: str = ""
    required_objects: list[ObjectRequirement] = Field(default_factory=list)
    weights: dict[str, float] = Field(
        default_factory=lambda: {"ocr": 1.0, "asr": 1.0, "visual": 1.0}
    )


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
    channel_scores: dict[str, float] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    mode: str
    hits: list[SearchHit]
    plan: QueryPlan | None = None


class QAFrame(BaseModel):
    frame_id: int
    timestamp_sec: float
    path: Path


class QAFrameGroup(BaseModel):
    video_id: str
    center_frame_id: int
    retrieval_score: float
    frames: list[QAFrame]
    sources: list[str] = Field(default_factory=list)


class QAAnswerHit(BaseModel):
    """One ranked QA submission row: video_id, frame_id, answer."""

    video_id: str
    frame_id: int
    answer: str
    score: float = 0.0
    timestamp_sec: float | None = None
    image_url: str | None = None
    video_url: str | None = None
    source: str = "qa"


class EventSpec(BaseModel):
    """One ordered event in a KIS / QA / TRAKE query."""

    event_id: str
    description: str = ""
    ocr: str = ""
    asr: str = ""
    visual: str = ""
    required_objects: list[ObjectRequirement] = Field(default_factory=list)
    is_question_target: bool = False
    # Seconds from the previous event in a typical clip. E1 is always null.
    gap_from_prev_sec: float | None = None
    gap_min_sec: float | None = None
    gap_max_sec: float | None = None


class EventChainPlan(BaseModel):
    task: Literal["kis", "qa", "trake"] = "kis"
    context: str = ""
    events: list[EventSpec] = Field(default_factory=list)
    question_event_id: str | None = None


class EventHit(BaseModel):
    event_id: str
    frame_index: int
    score: float = 0.0
    timestamp_sec: float | None = None
    keyframe_path: str | None = None
    text: str | None = None
    source: str = "event"
    description: str | None = None
    image_url: str | None = None
    video_url: str | None = None


class EventChain(BaseModel):
    video_id: str
    score: float
    events: list[EventHit] = Field(default_factory=list)


# TRAKE aliases (backward compatible API field names).
TrakeEventPlan = EventSpec
TrakePlan = EventChainPlan
TrakeEventHit = EventHit
TrakeChain = EventChain


class TrakeResult(BaseModel):
    query: str
    plan: EventChainPlan | None = None
    chains: list[EventChain] = Field(default_factory=list)
    csv_row: str = ""


class KisResult(BaseModel):
    query: str
    plan: EventChainPlan | None = None
    chains: list[EventChain] = Field(default_factory=list)
    hits: list[SearchHit] = Field(default_factory=list)
    submission_rows: list[tuple[str, int]] = Field(default_factory=list)


class QAResultItem(BaseModel):
    chain: EventChain
    answer: str
    questioned_event_id: str
    questioned_frame_id: int


class QAResult(BaseModel):
    question: str
    video_id: str
    frame_id: int
    answer: str
    plan: EventChainPlan | None = None
    results: list[QAResultItem] = Field(default_factory=list)
    hits: list["QAAnswerHit"] = Field(default_factory=list)
    # Legacy fields kept for gradual UI migration.
    descriptions: list[str] = Field(default_factory=list)
    frame_groups: list[QAFrameGroup] = Field(default_factory=list)
