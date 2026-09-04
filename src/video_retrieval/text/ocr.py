from __future__ import annotations

import json
from pathlib import Path

from video_retrieval.config import Settings
from video_retrieval.models import FrameRole, KeyFrame, TextDocument
from video_retrieval.text.content_parts import image_part, text_part
from video_retrieval.text.llm import LLMClient
from video_retrieval.text.llm_factory import get_llm_client


class OCREngine:
    """OCR via Gemini / Qwen VL, with a mock backend for local development."""

    def __init__(self, settings: Settings, *, client: LLMClient | None = None):
        self.settings = settings
        self.backend = (settings.ocr_backend or "mock").strip().lower()
        if self.backend in {"qwen", "qwen2_5_vl", "qwen2.5-vl"}:
            self.backend = "qwen_vl"
        self._client = client
        if self.backend in {"gemini", "qwen_vl"} and self._client is None:
            self._client = get_llm_client(settings, backend=self.backend)
            if self._client is None:
                raise ValueError(
                    f"OCR_BACKEND={self.backend} requires a working LLM "
                    "(GEMINI_API_KEY for gemini, or local Qwen VL weights)"
                )

    def extract_from_keyframes(self, keyframes: list[KeyFrame]) -> list[TextDocument]:
        ocr_keyframes = [kf for kf in keyframes if kf.role == FrameRole.MIDDLE]
        if self.backend in {"gemini", "qwen_vl"}:
            return self._extract_llm_keyframes(ocr_keyframes)

        docs: list[TextDocument] = []
        for kf in ocr_keyframes:
            text = self.extract_text(kf.path)
            if not text.strip():
                continue
            docs.append(self._to_text_document(kf, text))
        return docs

    def _extract_llm_keyframes(self, keyframes: list[KeyFrame]) -> list[TextDocument]:
        docs: list[TextDocument] = []
        batch_size = max(self.settings.gemini_batch_size, 1)
        batches = list(_chunked(keyframes, batch_size))
        total_batches = len(batches)

        for batch_index, batch in enumerate(batches, start=1):
            print(
                f"OCR batch {batch_index}/{total_batches}: "
                f"{len(batch)} frame(s), "
                f"shots {batch[0].shot_index}-{batch[-1].shot_index}"
            )
            text_by_image = self._extract_llm_batch(batch)
            for kf in batch:
                text = text_by_image.get(kf.path.name, "")
                if not text.strip():
                    continue
                docs.append(self._to_text_document(kf, text))
        return docs

    def extract_text(self, image_path: Path) -> str:
        if self.backend in {"gemini", "qwen_vl"}:
            return self._extract_llm(image_path)
        if FrameRole.MIDDLE.value in image_path.name:
            return f"mock ocr text from {image_path.name}"
        return ""

    def _extract_llm_batch(self, keyframes: list[KeyFrame]) -> dict[str, str]:
        from PIL import Image

        assert self._client is not None
        image_ids = [kf.path.name for kf in keyframes]
        parts: list = [text_part(_batch_ocr_instructions(image_ids))]
        for kf in keyframes:
            image = Image.open(kf.path)
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
            parts.append(text_part(f"[IMAGE: {kf.path.name}]"))
            parts.append(image_part(image))

        raw = self._client.generate_parts(
            parts,
            json_response=True,
            component="OCR batch",
        )
        return _parse_batch_ocr_response(raw, image_ids)

    def _extract_llm(self, image_path: Path) -> str:
        from PIL import Image

        assert self._client is not None
        image = Image.open(image_path)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        prompt = (
            "Extract all visible text from this video frame. "
            "Return plain text only, preserve line breaks for distinct regions. "
            "If there is no visible text, return an empty response."
        )
        return self._client.generate_parts(
            [text_part(prompt), image_part(image)],
            json_response=False,
            component="OCR frame",
        )

    @staticmethod
    def _to_text_document(kf: KeyFrame, text: str) -> TextDocument:
        return TextDocument(
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


def _chunked(items: list[KeyFrame], size: int) -> list[list[KeyFrame]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _batch_ocr_instructions(image_ids: list[str]) -> str:
    ids_json = json.dumps(image_ids)
    return (
        "Extract all visible text from each labeled video frame below. "
        f"The image_id values are exactly: {ids_json}. "
        'Return JSON only with shape {"results":[{"image_id":"<filename>","text":"..."}]}. '
        "Include one entry per image_id, in the same order. "
        "Use an empty string when a frame has no visible text."
    )


def _parse_batch_ocr_response(raw: str, image_ids: list[str]) -> dict[str, str]:
    text_by_id = {image_id: "" for image_id in image_ids}
    if not raw.strip():
        return text_by_id

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return text_by_id

    results = payload.get("results", [])
    if not isinstance(results, list):
        return text_by_id

    for item in results:
        if not isinstance(item, dict):
            continue
        image_id = item.get("image_id")
        if image_id in text_by_id:
            text_by_id[image_id] = str(item.get("text") or "")
    return text_by_id
