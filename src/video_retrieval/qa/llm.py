from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Protocol

from video_retrieval.config import Settings
from video_retrieval.models import QAFrameGroup
from video_retrieval.qa.prompts import (
    ANSWER_SYSTEM_PROMPT,
    DECOMPOSITION_SYSTEM_PROMPT,
    SINGLE_FRAME_ANSWER_PROMPT,
    build_answer_prompt,
    build_decomposition_prompt,
    build_single_frame_answer_prompt,
)
from video_retrieval.text.gemini_config import gemini_generate_config
from video_retrieval.text.gemini_logging import log_gemini_failure

logger = logging.getLogger(__name__)


class QAError(RuntimeError):
    """Base error for the Q&A pipeline."""


class QAModelNotConfiguredError(QAError):
    pass


class InvalidQAModelResponseError(QAError):
    pass


class QAModel(Protocol):
    def decompose_question(self, question: str) -> list[str]: ...

    def answer_with_frames(
        self,
        *,
        question: str,
        descriptions: list[str],
        video_id: str,
        frame_groups: list[QAFrameGroup],
    ) -> dict[str, object]: ...

    def answer_single_frame(
        self,
        *,
        question: str,
        video_id: str,
        frame_id: int,
        image_path: Path | None,
        event_description: str = "",
    ) -> str: ...


class UnconfiguredQAModel:
    def _raise(self) -> None:
        raise QAModelNotConfiguredError(
            "Q&A LLM is not configured. Set QA_LLM_BACKEND=gemini and GEMINI_API_KEY."
        )

    def decompose_question(self, question: str) -> list[str]:
        self._raise()
        return []

    def answer_with_frames(
        self,
        *,
        question: str,
        descriptions: list[str],
        video_id: str,
        frame_groups: list[QAFrameGroup],
    ) -> dict[str, object]:
        self._raise()
        return {}

    def answer_single_frame(
        self,
        *,
        question: str,
        video_id: str,
        frame_id: int,
        image_path: Path | None,
        event_description: str = "",
    ) -> str:
        self._raise()
        return ""


class GeminiQAModel:
    """Multimodal Q&A via the project's Gemini client (PIL image parts)."""

    def __init__(self, settings: Settings):
        if not settings.gemini_api_key:
            raise QAModelNotConfiguredError("GEMINI_API_KEY is required for Q&A")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise QAModelNotConfiguredError(
                "google-genai is required for Gemini Q&A"
            ) from exc

        self.settings = settings
        self.model = settings.gemini_model
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._types = types

    def decompose_question(self, question: str) -> list[str]:
        prompt = (
            f"{DECOMPOSITION_SYSTEM_PROMPT}\n\n"
            f"{build_decomposition_prompt(question)}"
        )
        raw = self._generate_text(prompt)
        payload = _parse_json_object(raw)
        descriptions = payload.get("descriptions")
        if not isinstance(descriptions, list):
            raise InvalidQAModelResponseError("LLM response is missing descriptions[]")
        cleaned = [str(item).strip() for item in descriptions if str(item).strip()]
        if not cleaned:
            raise InvalidQAModelResponseError("LLM returned no retrieval descriptions")
        return list(dict.fromkeys(cleaned))

    def answer_single_frame(
        self,
        *,
        question: str,
        video_id: str,
        frame_id: int,
        image_path: Path | None,
        event_description: str = "",
    ) -> str:
        from PIL import Image

        parts: list = [
            self._types.Part.from_text(text=SINGLE_FRAME_ANSWER_PROMPT),
            self._types.Part.from_text(
                text=build_single_frame_answer_prompt(
                    question,
                    video_id,
                    frame_id,
                    event_description=event_description,
                )
            ),
        ]
        if image_path is not None and image_path.is_file():
            image = Image.open(image_path)
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
            parts.append(self._types.Part.from_text(text=f"frame_id={frame_id}"))
            parts.append(self._types.Part(image))
        raw = self._generate_parts(parts)
        payload = _parse_json_object(raw)
        return str(payload.get("answer") or "").strip()

    def answer_with_frames(
        self,
        *,
        question: str,
        descriptions: list[str],
        video_id: str,
        frame_groups: list[QAFrameGroup],
    ) -> dict[str, object]:
        from PIL import Image

        compact_groups: list[QAFrameGroup] = []
        for group in frame_groups:
            centers = [frame for frame in group.frames if frame.frame_id == group.center_frame_id]
            if not centers and group.frames:
                centers = [group.frames[len(group.frames) // 2]]
            if centers:
                compact_groups.append(group.model_copy(update={"frames": centers}))
        if not compact_groups:
            compact_groups = frame_groups

        parts: list = [
            self._types.Part.from_text(text=ANSWER_SYSTEM_PROMPT),
            self._types.Part.from_text(
                text=build_answer_prompt(question, descriptions, video_id, compact_groups)
            ),
        ]
        for group_index, group in enumerate(compact_groups, start=1):
            parts.append(
                self._types.Part.from_text(
                    text=f"Group {group_index}, center frame {group.center_frame_id}"
                )
            )
            for frame in group.frames:
                parts.append(
                    self._types.Part.from_text(text=f"frame_id={frame.frame_id}")
                )
                image = Image.open(frame.path)
                if image.mode not in ("RGB", "L"):
                    image = image.convert("RGB")
                parts.append(self._types.Part(image))

        raw = self._generate_parts(parts)
        return _parse_json_object(raw)

    def _generate_text(self, prompt: str) -> str:
        return self._generate_parts([self._types.Part.from_text(text=prompt)])

    def _generate_parts(self, parts: list) -> str:
        kwargs: dict = {
            "model": self.model,
            "contents": [self._types.Content(role="user", parts=parts)],
        }
        config = gemini_generate_config(self.model, json_response=True)
        if config is not None:
            kwargs["config"] = config

        max_retries = max(1, int(self.settings.gemini_max_retries))
        delay = 1.5
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = self._client.models.generate_content(**kwargs)
                return (response.text or "").strip()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                message = str(exc)
                retryable = any(
                    token in message
                    for token in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "high demand")
                )
                if not retryable or attempt >= max_retries - 1:
                    log_gemini_failure(
                        component="QA generate_content",
                        model=self.model,
                        exc=exc,
                        attempt=attempt + 1,
                        max_attempts=max_retries,
                        fallback="raise to caller",
                    )
                    raise
                logger.warning(
                    "Gemini QA call failed (model=%s, attempt=%s/%s, error_type=%s), retrying in %.1fs: %r",
                    self.model,
                    attempt + 1,
                    max_retries,
                    type(exc).__name__,
                    delay,
                    exc,
                )
                time.sleep(delay)
                delay = min(delay * 2, 12)
        assert last_exc is not None
        raise last_exc


def create_qa_model(settings: Settings) -> QAModel:
    backend = (settings.qa_llm_backend or "gemini").strip().lower()
    if backend == "gemini":
        if not settings.gemini_api_key:
            return UnconfiguredQAModel()
        try:
            return GeminiQAModel(settings)
        except QAModelNotConfiguredError:
            return UnconfiguredQAModel()
    if backend == "none":
        return UnconfiguredQAModel()
    raise QAModelNotConfiguredError(f"Unsupported QA_LLM_BACKEND: {backend!r}")


def _parse_json_object(raw: str) -> dict[str, object]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise InvalidQAModelResponseError("LLM did not return a JSON object")
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise InvalidQAModelResponseError("LLM returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise InvalidQAModelResponseError("LLM response must be a JSON object")
    return payload
