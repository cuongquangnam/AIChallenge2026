from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from video_retrieval.config import Settings
from video_retrieval.models import QAFrameGroup
from video_retrieval.qa.prompts import (
    ANSWER_SYSTEM_PROMPT,
    BATCH_FRAME_ANSWER_PROMPT,
    DECOMPOSITION_SYSTEM_PROMPT,
    build_answer_prompt,
    build_batch_frame_answer_prompt,
    build_decomposition_prompt,
)
from video_retrieval.text.gemini_client import get_gemini_client
from video_retrieval.text.llm import LLMClient


@dataclass(frozen=True)
class QASingleFrameRequest:
    chain_index: int
    video_id: str
    frame_id: int
    image_path: Path | None
    event_description: str = ""


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

    def answer_single_frames_batch(
        self,
        *,
        question: str,
        items: list[QASingleFrameRequest],
    ) -> dict[int, str]: ...


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

    def answer_single_frames_batch(
        self,
        *,
        question: str,
        items: list[QASingleFrameRequest],
    ) -> dict[int, str]:
        self._raise()
        return {}


class GeminiQAModel:
    """Multimodal Q&A via the shared Gemini client (PIL image parts)."""

    def __init__(self, settings: Settings, *, llm: LLMClient | None = None):
        if not settings.gemini_api_key:
            raise QAModelNotConfiguredError("GEMINI_API_KEY is required for Q&A")
        client = llm or get_gemini_client(settings)
        if client is None:
            raise QAModelNotConfiguredError("GEMINI_API_KEY is required for Q&A")
        try:
            from google.genai import types
        except ImportError as exc:
            raise QAModelNotConfiguredError(
                "google-genai is required for Gemini Q&A"
            ) from exc

        self.settings = settings
        self.model = client.model
        self._llm = client
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
        answers = self.answer_single_frames_batch(
            question=question,
            items=[
                QASingleFrameRequest(
                    chain_index=0,
                    video_id=video_id,
                    frame_id=frame_id,
                    image_path=image_path,
                    event_description=event_description,
                )
            ],
        )
        return answers.get(0, "")

    def answer_single_frames_batch(
        self,
        *,
        question: str,
        items: list[QASingleFrameRequest],
    ) -> dict[int, str]:
        if not items:
            return {}
        from PIL import Image

        prompt_items = [
            (item.chain_index, item.video_id, item.frame_id, item.event_description)
            for item in items
        ]
        parts: list = [
            self._types.Part.from_text(text=BATCH_FRAME_ANSWER_PROMPT),
            self._types.Part.from_text(
                text=build_batch_frame_answer_prompt(question, prompt_items)
            ),
        ]
        for item in items:
            parts.append(
                self._types.Part.from_text(
                    text=(
                        f"chain_index={item.chain_index} "
                        f"video_id={item.video_id} frame_id={item.frame_id}"
                    )
                )
            )
            if item.image_path is not None and item.image_path.is_file():
                image = Image.open(item.image_path)
                if image.mode not in ("RGB", "L"):
                    image = image.convert("RGB")
                parts.append(self._types.Part(image))

        raw = self._generate_parts(parts)
        payload = _parse_json_object(raw)
        answers_raw = payload.get("answers")
        if not isinstance(answers_raw, list):
            raise InvalidQAModelResponseError("LLM batch response is missing answers[]")

        out: dict[int, str] = {}
        for entry in answers_raw:
            if not isinstance(entry, dict):
                continue
            try:
                chain_index = int(entry.get("chain_index"))
            except (TypeError, ValueError):
                continue
            out[chain_index] = str(entry.get("answer") or "").strip()
        return out

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
        return self._llm.generate_parts(
            parts,
            json_response=True,
            component="QA generate_content",
        )


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
