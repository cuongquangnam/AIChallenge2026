from __future__ import annotations

import os

from video_retrieval.config import Settings
from video_retrieval.models import AudioTrack, FrameRole, KeyFrame, Shot, TextDocument


class ASREngine:
    """Whisper ASR with a mock backend for local development."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.backend = (settings.asr_backend or "mock").strip().lower()
        self._model = None
        if self.backend == "whisper":
            self._load_whisper()
        elif self.backend == "faster_whisper":
            self._load_faster_whisper()
        elif self.backend != "mock":
            raise ValueError(
                f"Unknown ASR_BACKEND={settings.asr_backend!r}; "
                "expected mock, whisper, or faster_whisper"
            )

    def _load_whisper(self) -> None:
        try:
            import whisper
        except ImportError as exc:
            raise ImportError("Install ML extras: pip install '.[ml]'") from exc
        self._model = whisper.load_model(self.settings.whisper_model)

    def _load_faster_whisper(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError(
                "Install faster-whisper: pip install '.[ml]' "
                "(adds faster-whisper for Mac-efficient ASR)."
            ) from exc

        threads = int(self.settings.faster_whisper_cpu_threads or 0)
        if threads <= 0:
            threads = max(os.cpu_count() or 4, 1)
        # CPU + int8 is the most reliable Mac throughput path.
        self._model = WhisperModel(
            self.settings.whisper_model,
            device="cpu",
            compute_type=self.settings.faster_whisper_compute_type or "int8",
            cpu_threads=threads,
        )
        print(
            f"faster-whisper model={self.settings.whisper_model} "
            f"compute_type={self.settings.faster_whisper_compute_type} "
            f"cpu_threads={threads}",
            flush=True,
        )

    def transcribe(
        self,
        audio: AudioTrack,
        shots: list[Shot] | None = None,
    ) -> list[TextDocument]:
        if self.backend == "whisper":
            docs = self._transcribe_whisper(audio)
        elif self.backend == "faster_whisper":
            docs = self._transcribe_faster_whisper(audio)
        else:
            docs = [
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
        return attach_asr_docs_to_shots(docs, shots or [])

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

    def _transcribe_faster_whisper(self, audio: AudioTrack) -> list[TextDocument]:
        segments, _info = self._model.transcribe(str(audio.path), vad_filter=True)
        docs: list[TextDocument] = []
        for idx, seg in enumerate(segments):
            text = (seg.text or "").strip()
            if not text:
                continue
            docs.append(
                TextDocument(
                    doc_id=f"{audio.video_id}:asr:{idx}",
                    video_id=audio.video_id,
                    source="asr",
                    text=text,
                    start_sec=float(seg.start or 0.0),
                    end_sec=float(seg.end or 0.0),
                    metadata={"audio_path": str(audio.path)},
                )
            )
        return docs


def attach_asr_docs_to_shots(
    docs: list[TextDocument],
    shots: list[Shot],
) -> list[TextDocument]:
    """Bind each ASR segment to the overlapping shot's representative keyframe."""
    if not docs or not shots:
        return docs

    attached: list[TextDocument] = []
    for doc in docs:
        shot = _shot_for_time(shots, _segment_midpoint(doc))
        if shot is None:
            attached.append(doc)
            continue
        keyframe = _representative_keyframe(shot)
        metadata = dict(doc.metadata)
        if keyframe is not None:
            metadata["keyframe_path"] = str(keyframe.path)
            attached.append(
                doc.model_copy(
                    update={
                        "shot_index": shot.shot_index,
                        "frame_index": keyframe.frame_index,
                        "role": keyframe.role,
                        "metadata": metadata,
                    }
                )
            )
        else:
            attached.append(
                doc.model_copy(
                    update={
                        "shot_index": shot.shot_index,
                        "metadata": metadata,
                    }
                )
            )
    return attached


def _segment_midpoint(doc: TextDocument) -> float:
    start = float(doc.start_sec or 0.0)
    end = float(doc.end_sec if doc.end_sec is not None else start)
    return (start + end) / 2.0


def _shot_for_time(shots: list[Shot], t: float) -> Shot | None:
    for shot in shots:
        if shot.start_sec <= t <= shot.end_sec:
            return shot
    if not shots:
        return None
    # Clamp to nearest shot when ASR spills slightly past boundaries.
    return min(
        shots,
        key=lambda shot: min(abs(t - shot.start_sec), abs(t - shot.end_sec)),
    )


def _representative_keyframe(shot: Shot) -> KeyFrame | None:
    if not shot.keyframes:
        return None
    for keyframe in shot.keyframes:
        if keyframe.role == FrameRole.MIDDLE:
            return keyframe
    return shot.keyframes[0]
