from __future__ import annotations

import json
import shutil
from pathlib import Path

from video_retrieval.config import Settings, get_settings
from video_retrieval.detection.objects import ObjectDetector
from video_retrieval.encoders.visual import VisualEncoder
from video_retrieval.extraction.audio import extract_audio
from video_retrieval.extraction.keyframes import extract_keyframes
from video_retrieval.models import (
    AudioTrack,
    FrameObjectDetections,
    IndexResult,
    KeyFrame,
    Shot,
    TextDocument,
)
from video_retrieval.storage.elasticsearch_store import ElasticsearchStore
from video_retrieval.storage.qdrant_store import QdrantStore
from video_retrieval.text.asr import ASREngine
from video_retrieval.text.ocr import OCREngine

INDEX_STAGES = ("visual", "ocr", "asr", "objects")
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


class VideoIndexer:
    """Offline indexing pipeline matching the architecture diagram."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        visual: VisualEncoder | None = None,
        ocr: OCREngine | None = None,
        asr: ASREngine | None = None,
        objects: ObjectDetector | None = None,
        qdrant: QdrantStore | None = None,
        es: ElasticsearchStore | None = None,
    ):
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        self._visual = visual
        self._ocr = ocr
        self._asr = asr
        self._objects = objects
        self._qdrant = qdrant
        self._es = es

    @property
    def visual(self) -> VisualEncoder:
        if self._visual is None:
            self._visual = VisualEncoder(self.settings)
        return self._visual

    @property
    def ocr(self) -> OCREngine:
        if self._ocr is None:
            self._ocr = OCREngine(self.settings)
        return self._ocr

    @property
    def asr(self) -> ASREngine:
        if self._asr is None:
            self._asr = ASREngine(self.settings)
        return self._asr

    @property
    def qdrant(self) -> QdrantStore:
        if self._qdrant is None:
            self._qdrant = QdrantStore(self.settings)
        return self._qdrant

    @property
    def objects(self) -> ObjectDetector:
        if self._objects is None:
            self._objects = ObjectDetector(self.settings)
        return self._objects

    @property
    def es(self) -> ElasticsearchStore:
        if self._es is None:
            self._es = ElasticsearchStore(self.settings)
        return self._es

    def index_video(
        self,
        video_path: Path,
        video_id: str | None = None,
        *,
        stages: list[str] | set[str] | None = None,
        reuse_extract: bool = True,
    ) -> IndexResult:
        video_path = Path(video_path).resolve()
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        selected = normalize_stages(stages)
        video_id = video_id or video_path.stem
        stored_video = self._store_video(video_path, video_id)
        shots, audio = self._ensure_extracted(
            stored_video,
            video_id,
            selected=selected,
            reuse_extract=reuse_extract,
        )
        keyframes: list[KeyFrame] = [kf for shot in shots for kf in shot.keyframes]

        existing_manifest = self._read_manifest(video_id)
        object_detections = _load_object_detections(existing_manifest)
        if "objects" in selected:
            object_detections = self.objects.detect_keyframes(keyframes)
        n_objects = sum(len(frame.detections) for frame in object_detections)

        n_visual = 0
        if "visual" in selected:
            embeddings = self.visual.encode_keyframes(keyframes)
            n_visual = self.qdrant.upsert_embeddings(
                embeddings,
                object_detections=object_detections,
            )
        elif "objects" in selected:
            self.qdrant.set_object_payload(object_detections)

        text_docs: list[TextDocument] = []
        existing_text = existing_manifest.get("text_docs") if existing_manifest else []
        if not isinstance(existing_text, list):
            existing_text = []

        if "ocr" in selected:
            ocr_docs = self.ocr.extract_from_keyframes(keyframes)
            text_docs.extend(ocr_docs)
            existing_text = _replace_text_source(existing_text, "ocr", ocr_docs)
        if "asr" in selected:
            if audio is None:
                raise FileNotFoundError(f"No audio track for {video_id}")
            asr_docs = self.asr.transcribe(audio, shots=shots)
            text_docs.extend(asr_docs)
            existing_text = _replace_text_source(existing_text, "asr", asr_docs)

        n_text = 0
        if text_docs:
            n_text = self.es.index_documents(text_docs)

        self._write_manifest(
            video_id=video_id,
            stored_video=stored_video,
            audio=audio,
            shots=shots,
            keyframes=keyframes,
            text_docs=existing_text,
            object_detections=object_detections,
        )
        return IndexResult(
            video_id=video_id,
            video_path=stored_video,
            num_shots=len(shots),
            num_keyframes=len(keyframes),
            num_visual_points=n_visual,
            num_text_docs=n_text,
            num_object_detections=n_objects,
            audio_path=audio.path if audio else None,
            stages=sorted(selected),
        )

    def index_directory(
        self,
        directory: Path,
        *,
        stages: list[str] | set[str] | None = None,
        reuse_extract: bool = True,
    ) -> list[IndexResult]:
        directory = Path(directory)
        results: list[IndexResult] = []
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() in VIDEO_EXTS:
                results.append(
                    self.index_video(path, stages=stages, reuse_extract=reuse_extract)
                )
        return results

    def _store_video(self, video_path: Path, video_id: str) -> Path:
        stored_video = self.settings.videos_dir / f"{video_id}{video_path.suffix.lower()}"
        if video_path != stored_video.resolve():
            shutil.copy2(video_path, stored_video)
        return stored_video

    def _ensure_extracted(
        self,
        stored_video: Path,
        video_id: str,
        *,
        selected: set[str],
        reuse_extract: bool,
    ) -> tuple[list[Shot], AudioTrack | None]:
        need_shots = bool(selected & {"visual", "ocr", "asr", "objects"})
        need_audio = "asr" in selected
        shots: list[Shot] = []
        audio: AudioTrack | None = None

        if reuse_extract:
            shots, audio = self._load_extracted(video_id)

        if need_shots and not shots:
            shots = extract_keyframes(
                stored_video,
                self.settings.keyframes_dir,
                video_id,
                shot_backend=self.settings.shot_backend,
            )
        if need_audio and (audio is None or not audio.path.exists()):
            audio = extract_audio(stored_video, self.settings.audio_dir, video_id)
        if need_audio and audio is None:
            raise FileNotFoundError(f"No audio track for {video_id}")
        if need_shots and not shots:
            raise FileNotFoundError(f"No keyframes for {video_id}")
        return shots, audio

    def _load_extracted(self, video_id: str) -> tuple[list[Shot], AudioTrack | None]:
        manifest = self._read_manifest(video_id)
        shots: list[Shot] = []
        audio: AudioTrack | None = None
        if manifest:
            raw_shots = manifest.get("shots") or []
            if isinstance(raw_shots, list):
                shots = [Shot.model_validate(item) for item in raw_shots]
            audio_path = manifest.get("audio_path")
            if audio_path:
                path = Path(audio_path)
                if path.exists():
                    audio = AudioTrack(video_id=video_id, path=path)

        wav = self.settings.audio_dir / f"{video_id}.wav"
        if audio is None and wav.exists():
            audio = AudioTrack(video_id=video_id, path=wav)
        return shots, audio

    def _read_manifest(self, video_id: str) -> dict:
        manifest_path = self.settings.manifests_dir / f"{video_id}.json"
        if not manifest_path.exists():
            return {}
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write_manifest(
        self,
        *,
        video_id: str,
        stored_video: Path,
        audio: AudioTrack | None,
        shots: list[Shot],
        keyframes: list[KeyFrame],
        text_docs: list,
        object_detections: list[FrameObjectDetections],
    ) -> None:
        existing = self._read_manifest(video_id)
        serialized_docs = [
            doc.model_dump(mode="json") if isinstance(doc, TextDocument) else doc
            for doc in text_docs
        ]
        manifest = {
            **existing,
            "video_id": video_id,
            "video_path": str(stored_video),
            "audio_path": str(audio.path) if audio else existing.get("audio_path"),
            "num_shots": len(shots) or existing.get("num_shots", 0),
            "num_keyframes": len(keyframes) or existing.get("num_keyframes", 0),
            "shots": [shot.model_dump(mode="json") for shot in shots]
            or existing.get("shots", []),
            "text_docs": serialized_docs,
            "object_detections": [
                frame.model_dump(mode="json") for frame in object_detections
            ],
        }
        manifest_path = self.settings.manifests_dir / f"{video_id}.json"
        self.settings.manifests_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def normalize_stages(stages: list[str] | set[str] | None) -> set[str]:
    if not stages:
        return set(INDEX_STAGES)
    selected = {str(item).strip().lower() for item in stages if str(item).strip()}
    unknown = selected - set(INDEX_STAGES)
    if unknown:
        raise ValueError(
            f"Unknown index stage(s) {sorted(unknown)}; expected one of {list(INDEX_STAGES)}"
        )
    if not selected:
        raise ValueError("At least one index stage is required: visual, ocr, asr, objects")
    return selected


def _replace_text_source(existing: list, source: str, docs: list[TextDocument]) -> list:
    kept = [
        item
        for item in existing
        if not (isinstance(item, dict) and item.get("source") == source)
        and not (isinstance(item, TextDocument) and item.source == source)
    ]
    return kept + list(docs)


def _load_object_detections(manifest: dict) -> list[FrameObjectDetections]:
    raw = manifest.get("object_detections") or []
    if not isinstance(raw, list):
        return []
    frames: list[FrameObjectDetections] = []
    for item in raw:
        try:
            frames.append(FrameObjectDetections.model_validate(item))
        except (TypeError, ValueError):
            continue
    return frames
