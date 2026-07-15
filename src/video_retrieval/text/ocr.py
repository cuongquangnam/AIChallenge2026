from __future__ import annotations

from pathlib import Path

from video_retrieval.config import Settings
from video_retrieval.models import FrameRole, KeyFrame, TextDocument


class OCREngine:
    """Gemini OCR with a mock backend for local development."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.backend = settings.ocr_backend
        self._client = None
        if self.backend == "gemini":
            self._init_gemini()

    def _init_gemini(self) -> None:
        if not self.settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required for OCR_BACKEND=gemini")
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError("Install ML extras: pip install '.[ml]'") from exc

        genai.configure(api_key=self.settings.gemini_api_key)
        self._client = genai.GenerativeModel(self.settings.gemini_model)

    def extract_from_keyframes(self, keyframes: list[KeyFrame]) -> list[TextDocument]:
        docs: list[TextDocument] = []
        for kf in keyframes:
            text = self.extract_text(kf.path)
            if not text.strip():
                continue
            docs.append(
                TextDocument(
                    doc_id=f"{kf.video_id}:ocr:{kf.shot_index}:{kf.role.value}",
                    video_id=kf.video_id,
                    source="ocr",
                    text=text.strip(),
                    shot_index=kf.shot_index,
                    frame_index=kf.frame_index,
                    role=kf.role,
                    start_sec=kf.timestamp_sec,
                    end_sec=kf.timestamp_sec,
                    metadata={"keyframe_path": str(kf.path)},
                )
            )
        return docs

    def extract_text(self, image_path: Path) -> str:
        if self.backend == "gemini":
            return self._extract_gemini(image_path)
        # Mock: tag middle frames so textual search is exerciseable end-to-end.
        name = image_path.stem
        if FrameRole.MIDDLE.value in name:
            return f"mock ocr text from {image_path.name}"
        return ""

    def _extract_gemini(self, image_path: Path) -> str:
        from PIL import Image

        image = Image.open(image_path)
        prompt = (
            "Extract all visible text from this video frame. "
            "Return plain text only, preserve line breaks for distinct regions."
        )
        response = self._client.generate_content([prompt, image])
        return (response.text or "").strip()
