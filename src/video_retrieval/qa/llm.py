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
from video_retrieval.text.content_parts import image_part, text_part
from video_retrieval.text.llm import LLMClient
from video_retrieval.text.llm_factory import get_llm_client, resolve_llm_backend


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
            "Q&A LLM is not configured. Set QA_LLM_BACKEND=gemini|qwen_vl "
            "(or LLM_BACKEND=qwen_vl) and provide GEMINI_API_KEY when using Gemini."
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
    """Multimodal Q&A via shared LLM client (Gemini or Qwen2.5-VL)."""

    def __init__(
        self,
        settings: Settings,
        *,
        llm: LLMClient | None = None,
        backend: str | None = None,
    ):
        chosen = (backend or _qa_backend(settings)).strip().lower()
        if chosen in {"qwen", "qwen2_5_vl", "qwen2.5-vl"}:
            chosen = "qwen_vl"
        client = llm or get_llm_client(settings, backend=chosen)
        if client is None:
            raise QAModelNotConfiguredError(
                f"QA backend {chosen!r} is unavailable "
                "(check GEMINI_API_KEY or local Qwen VL install)"
            )

        self.settings = settings
        self.backend = chosen
        self.model = client.model
        self._llm = client

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
            text_part(BATCH_FRAME_ANSWER_PROMPT),
            text_part(build_batch_frame_answer_prompt(question, prompt_items)),
        ]
        for item in items:
            parts.append(
                text_part(
                    f"chain_index={item.chain_index} "
                    f"video_id={item.video_id} frame_id={item.frame_id}"
                )
            )
            if item.image_path is not None and item.image_path.is_file():
                image = Image.open(item.image_path)
                if image.mode not in ("RGB", "L"):
                    image = image.convert("RGB")
                parts.append(image_part(image))

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
            text_part(ANSWER_SYSTEM_PROMPT),
            text_part(build_answer_prompt(question, descriptions, video_id, compact_groups)),
        ]
        for group_index, group in enumerate(compact_groups, start=1):
            parts.append(
                text_part(f"Group {group_index}, center frame {group.center_frame_id}")
            )
            for frame in group.frames:
                parts.append(text_part(f"frame_id={frame.frame_id}"))
                image = Image.open(frame.path)
                if image.mode not in ("RGB", "L"):
                    image = image.convert("RGB")
                parts.append(image_part(image))

        raw = self._generate_parts(parts)
        return _parse_json_object(raw)

    def _generate_text(self, prompt: str) -> str:
        return self._generate_parts([text_part(prompt)])

    def _generate_parts(self, parts: list) -> str:
        return self._llm.generate_parts(
            parts,
            json_response=True,
            component="QA generate_content",
        )


def _qa_backend(settings: Settings) -> str:
    backend = (settings.qa_llm_backend or "auto").strip().lower()
    if backend in {"qwen", "qwen2_5_vl", "qwen2.5-vl"}:
        return "qwen_vl"
    if backend in {"gemini", "qwen_vl", "none"}:
        return backend
    # auto → shared llm_backend, else gemini when keyed
    shared = resolve_llm_backend(settings)
    if shared in {"gemini", "qwen_vl"}:
        return shared
    return "none"


def create_qa_model(settings: Settings) -> QAModel:
    backend = _qa_backend(settings)
    if backend == "none":
        return UnconfiguredQAModel()
    if backend in {"gemini", "qwen_vl"}:
        try:
            return GeminiQAModel(settings, backend=backend)
        except QAModelNotConfiguredError:
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
