from __future__ import annotations

import json
import shutil
from pathlib import Path

from video_retrieval.config import Settings, get_settings
from video_retrieval.encoders.visual import VisualEncoder
from video_retrieval.extraction.audio import extract_audio
from video_retrieval.extraction.keyframes import (
    extract_keyframes,
    load_existing_shots,
    video_timing,
)
from video_retrieval.models import AudioTrack, IndexResult, KeyFrame, Shot, TextDocument
from video_retrieval.storage.elasticsearch_store import ElasticsearchStore
from video_retrieval.storage.qdrant_store import QdrantStore
from video_retrieval.text.asr import ASREngine
from video_retrieval.text.ocr import OCREngine

INDEX_STAGES = ("visual", "ocr", "asr")
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
        qdrant: QdrantStore | None = None,
        es: ElasticsearchStore | None = None,
    ):
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        self._visual = visual
        self._ocr = ocr
        self._asr = asr
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
        resume: bool = True,
    ) -> IndexResult:
        video_path = Path(video_path).resolve()
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        selected = normalize_stages(stages)
        video_id = video_id or video_path.stem
        stored_video = self._store_video(video_path, video_id)
        requested = set(selected)
        if resume:
            already = self._completed_stages(video_id)
            selected = requested - already
            if not selected:
                print(f"Skipping {video_id}: already indexed {sorted(requested)}")
                return self._result_from_manifest(video_id, stored_video, requested)
            if already:
                print(
                    f"Resuming {video_id}: {sorted(selected)} "
                    f"(already {sorted(already & requested)})"
                )

        print(f"Indexing {video_id}: stages={sorted(selected)}")
        shots, audio = self._ensure_extracted(
            stored_video,
            video_id,
            selected=selected,
            reuse_extract=reuse_extract,
        )
        keyframes: list[KeyFrame] = [kf for shot in shots for kf in shot.keyframes]

        n_visual = 0
        if "visual" in selected:
            embeddings = self.visual.encode_keyframes(keyframes)
            n_visual = self.qdrant.upsert_embeddings(embeddings)

        text_docs: list[TextDocument] = []
        existing_manifest = self._read_manifest(video_id)
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
            indexed_stages=selected,
        )
        return IndexResult(
            video_id=video_id,
            video_path=stored_video,
            num_shots=len(shots),
            num_keyframes=len(keyframes),
            num_visual_points=n_visual,
            num_text_docs=n_text,
            audio_path=audio.path if audio else None,
            stages=sorted(requested),
        )

    def index_directory(
        self,
        directory: Path,
        *,
        stages: list[str] | set[str] | None = None,
        reuse_extract: bool = True,
        resume: bool = True,
    ) -> list[IndexResult]:
        directory = Path(directory)
        results: list[IndexResult] = []
        videos = [
            path
            for path in sorted(directory.iterdir())
            if path.is_file() and path.suffix.lower() in VIDEO_EXTS
        ]
        if videos:
            for path in videos:
                results.append(
                    self.index_video(
                        path,
                        stages=stages,
                        reuse_extract=reuse_extract,
                        resume=resume,
                    )
                )
            return results

        for folder in sorted(directory.iterdir()):
            if not folder.is_dir():
                continue
            video = self._video_for_id(folder.name)
            if video is None:
                continue
            results.append(
                self.index_video(
                    video,
                    video_id=folder.name,
                    stages=stages,
                    reuse_extract=reuse_extract,
                    resume=resume,
                )
            )
        return results

    def _video_for_id(self, video_id: str) -> Path | None:
        for ext in sorted(VIDEO_EXTS):
            candidate = self.settings.videos_dir / f"{video_id}{ext}"
            if candidate.exists():
                return candidate
        return None

    def _store_video(self, video_path: Path, video_id: str) -> Path:
        stored_video = (self.settings.videos_dir / f"{video_id}{video_path.suffix.lower()}").resolve()
        source = video_path.resolve()
        if source == stored_video or stored_video.exists():
            return stored_video
        stored_video.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, stored_video)
        return stored_video

    def _completed_stages(self, video_id: str) -> set[str]:
        manifest = self._read_manifest(video_id)
        if not manifest:
            return set()
        done = {str(item).strip().lower() for item in (manifest.get("indexed_stages") or [])}
        sources = {
            str(item.get("source") or "")
            for item in (manifest.get("text_docs") or [])
            if isinstance(item, dict)
        }
        if "ocr" in sources:
            done.add("ocr")
        if "asr" in sources:
            done.add("asr")
        if "visual" not in done:
            expected = int(manifest.get("num_keyframes") or 0)
            stored = self.qdrant.count_for_video(video_id)
            if expected and stored >= expected:
                done.add("visual")
            elif expected == 0 and stored > 0:
                done.add("visual")
        return done & set(INDEX_STAGES)

    def _result_from_manifest(
        self,
        video_id: str,
        stored_video: Path,
        selected: set[str],
    ) -> IndexResult:
        manifest = self._read_manifest(video_id)
        audio_path = manifest.get("audio_path")
        path = Path(audio_path) if audio_path else None
        text_docs = manifest.get("text_docs") or []
        return IndexResult(
            video_id=video_id,
            video_path=stored_video,
            num_shots=int(manifest.get("num_shots") or 0),
            num_keyframes=int(manifest.get("num_keyframes") or 0),
            num_visual_points=0,
            num_text_docs=len(text_docs) if isinstance(text_docs, list) else 0,
            audio_path=path if path and path.exists() else None,
            stages=sorted(selected),
        )

    def _ensure_extracted(
        self,
        stored_video: Path,
        video_id: str,
        *,
        selected: set[str],
        reuse_extract: bool,
    ) -> tuple[list[Shot], AudioTrack | None]:
        need_shots = bool(selected & {"visual", "ocr", "asr"})
        need_audio = "asr" in selected
        shots: list[Shot] = []
        audio: AudioTrack | None = None

        if reuse_extract:
            shots, audio = self._load_extracted(video_id, stored_video)

        if need_shots and not shots:
            if not reuse_extract:
                self._clear_keyframe_dir(video_id)
            shots = extract_keyframes(
                stored_video,
                self.settings.keyframes_dir,
                video_id,
                shot_backend=self.settings.shot_backend,
                max_shot_sec=self.settings.max_shot_sec,
                opencv_threshold=self.settings.opencv_shot_threshold,
                opencv_min_shot_len=self.settings.opencv_min_shot_len,
                transnet_threshold=self.settings.transnet_threshold,
                transnet_device=self.settings.transnet_device,
            )
        if need_audio and (audio is None or not audio.path.exists()):
            audio = extract_audio(stored_video, self.settings.audio_dir, video_id)
        if need_audio and audio is None:
            raise FileNotFoundError(f"No audio track for {video_id}")
        if need_shots and not shots:
            raise FileNotFoundError(f"No keyframes for {video_id}")
        return shots, audio

    def _clear_keyframe_dir(self, video_id: str) -> None:
        folder = self.settings.keyframes_dir / video_id
        if folder.is_dir():
            shutil.rmtree(folder)
            print(f"Cleared keyframes for {video_id}", flush=True)

    def _load_extracted(
        self,
        video_id: str,
        video_path: Path | None = None,
    ) -> tuple[list[Shot], AudioTrack | None]:
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
            audio = AudioTrack(
                video_id=video_id,
                path=wav,
                duration_sec=_wav_duration(wav),
            )
        elif audio is not None and audio.duration_sec is None:
            audio = audio.model_copy(update={"duration_sec": _wav_duration(audio.path)})

        if not shots:
            fps, duration = 25.0, audio.duration_sec if audio else None
            if video_path and video_path.exists():
                fps, video_duration = video_timing(video_path)
                duration = duration or video_duration
            shots = load_existing_shots(
                self.settings.keyframes_dir,
                video_id,
                fps=fps,
                duration_sec=duration,
            )
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
        indexed_stages: set[str] | list[str] | None = None,
    ) -> None:
        existing = self._read_manifest(video_id)
        serialized_docs = [
            doc.model_dump(mode="json") if isinstance(doc, TextDocument) else doc
            for doc in text_docs
        ]
        done = {str(item).strip().lower() for item in (existing.get("indexed_stages") or [])}
        if indexed_stages:
            done |= {str(item).strip().lower() for item in indexed_stages}
        for doc in serialized_docs:
            if isinstance(doc, dict) and doc.get("source") in INDEX_STAGES:
                done.add(str(doc["source"]))
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
            "indexed_stages": sorted(done),
        }
        manifest_path = self.settings.manifests_dir / f"{video_id}.json"
        self.settings.manifests_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


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
        raise ValueError("At least one index stage is required: visual, ocr, asr")
    return selected


def _replace_text_source(existing: list, source: str, docs: list[TextDocument]) -> list:
    kept = [
        item
        for item in existing
        if not (isinstance(item, dict) and item.get("source") == source)
        and not (isinstance(item, TextDocument) and item.source == source)
    ]
    return kept + list(docs)


def _wav_duration(path: Path) -> float | None:
    try:
        import wave

        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            if rate <= 0:
                return None
            return handle.getnframes() / float(rate)
    except Exception:
        return None
