from __future__ import annotations

import json
import shutil
from pathlib import Path

from video_retrieval.config import Settings, get_settings
from video_retrieval.encoders.visual import VisualEncoder
from video_retrieval.extraction.audio import extract_audio
from video_retrieval.extraction.keyframes import extract_keyframes
from video_retrieval.models import IndexResult, KeyFrame
from video_retrieval.storage.elasticsearch_store import ElasticsearchStore
from video_retrieval.storage.qdrant_store import QdrantStore
from video_retrieval.text.asr import ASREngine
from video_retrieval.text.ocr import OCREngine


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
        self.visual = visual or VisualEncoder(self.settings)
        self.ocr = ocr or OCREngine(self.settings)
        self.asr = asr or ASREngine(self.settings)
        self.qdrant = qdrant or QdrantStore(self.settings)
        self.es = es or ElasticsearchStore(self.settings)

    def index_video(self, video_path: Path, video_id: str | None = None) -> IndexResult:
        video_path = Path(video_path).resolve()
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        video_id = video_id or video_path.stem
        stored_video = self.settings.videos_dir / f"{video_id}{video_path.suffix.lower()}"
        if video_path != stored_video.resolve():
            shutil.copy2(video_path, stored_video)

        # 1) Parallel conceptual branches: keyframes + audio
        shots = extract_keyframes(
            stored_video,
            self.settings.keyframes_dir,
            video_id,
            shot_backend=self.settings.shot_backend,
        )
        audio = extract_audio(stored_video, self.settings.audio_dir, video_id)

        keyframes: list[KeyFrame] = [kf for shot in shots for kf in shot.keyframes]

        # 2) Visual feature indexing → Qdrant (SigLIP + BEiT3)
        embeddings = self.visual.encode_keyframes(keyframes)
        n_visual = self.qdrant.upsert_embeddings(embeddings)

        # 3) Textual indexing → Elasticsearch (Gemini OCR + Whisper ASR)
        ocr_docs = self.ocr.extract_from_keyframes(keyframes)
        asr_docs = self.asr.transcribe(audio)
        text_docs = ocr_docs + asr_docs
        n_text = self.es.index_documents(text_docs)

        manifest = {
            "video_id": video_id,
            "video_path": str(stored_video),
            "audio_path": str(audio.path),
            "num_shots": len(shots),
            "num_keyframes": len(keyframes),
            "shots": [shot.model_dump(mode="json") for shot in shots],
            "text_docs": [doc.model_dump(mode="json") for doc in text_docs],
        }
        manifest_path = self.settings.data_dir / "manifests" / f"{video_id}.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return IndexResult(
            video_id=video_id,
            video_path=stored_video,
            num_shots=len(shots),
            num_keyframes=len(keyframes),
            num_visual_points=n_visual,
            num_text_docs=n_text,
            audio_path=audio.path,
        )

    def index_directory(self, directory: Path) -> list[IndexResult]:
        directory = Path(directory)
        results: list[IndexResult] = []
        exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() in exts:
                results.append(self.index_video(path))
        return results
