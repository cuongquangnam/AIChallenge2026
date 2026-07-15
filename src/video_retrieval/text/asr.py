from __future__ import annotations

from pathlib import Path

from video_retrieval.config import Settings
from video_retrieval.models import AudioTrack, TextDocument


class ASREngine:
    """Whisper ASR with a mock backend for local development."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.backend = settings.asr_backend
        self._model = None
        if self.backend == "whisper":
            self._load_whisper()

    def _load_whisper(self) -> None:
        try:
            import whisper
        except ImportError as exc:
            raise ImportError("Install ML extras: pip install '.[ml]'") from exc
        self._model = whisper.load_model(self.settings.whisper_model)

    def transcribe(self, audio: AudioTrack) -> list[TextDocument]:
        if self.backend == "whisper":
            return self._transcribe_whisper(audio)
        return [
            TextDocument(
                doc_id=f"{audio.video_id}:asr:0",
                video_id=audio.video_id,
                source="asr",
                text=f"mock transcription for {audio.video_id}",
                start_sec=0.0,
                end_sec=audio.duration_sec,
                metadata={"audio_path": str(audio.path)},
            )
        ]

    def _transcribe_whisper(self, audio: AudioTrack) -> list[TextDocument]:
        result = self._model.transcribe(str(audio.path))
        docs: list[TextDocument] = []
        segments = result.get("segments") or []
        if not segments:
            text = (result.get("text") or "").strip()
            if text:
                docs.append(
                    TextDocument(
                        doc_id=f"{audio.video_id}:asr:full",
                        video_id=audio.video_id,
                        source="asr",
                        text=text,
                        start_sec=0.0,
                        end_sec=audio.duration_sec,
                        metadata={"audio_path": str(audio.path)},
                    )
                )
            return docs

        for idx, seg in enumerate(segments):
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            docs.append(
                TextDocument(
                    doc_id=f"{audio.video_id}:asr:{idx}",
                    video_id=audio.video_id,
                    source="asr",
                    text=text,
                    start_sec=float(seg.get("start", 0.0)),
                    end_sec=float(seg.get("end", 0.0)),
                    metadata={"audio_path": str(audio.path)},
                )
            )
        return docs
