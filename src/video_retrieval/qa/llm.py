from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Protocol

import httpx

from video_retrieval.config import Settings
from video_retrieval.models import QAFrameGroup
from video_retrieval.qa.prompts import (
    ANSWER_SYSTEM_PROMPT,
    DECOMPOSITION_SYSTEM_PROMPT,
    build_answer_prompt,
    build_decomposition_prompt,
)


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


class UnconfiguredQAModel:
    def _raise(self) -> None:
        raise QAModelNotConfiguredError(
            "Q&A LLM is not configured. Set QA_LLM_BACKEND=openai_compatible, "
            "QA_LLM_MODEL, and QA_LLM_API_KEY."
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


class OpenAICompatibleQAModel:
    """Small multimodal adapter for OpenAI-compatible chat completion APIs."""

    def __init__(self, settings: Settings):
        if not settings.qa_llm_api_key:
            raise QAModelNotConfiguredError("QA_LLM_API_KEY is required")
        if not settings.qa_llm_model:
            raise QAModelNotConfiguredError("QA_LLM_MODEL is required")
        self.api_key = settings.qa_llm_api_key
        self.model = settings.qa_llm_model
        self.endpoint = f"{settings.qa_llm_base_url.rstrip('/')}/chat/completions"
        self.timeout = settings.qa_llm_timeout_sec

    def decompose_question(self, question: str) -> list[str]:
        raw = self._chat(
            [
                {"role": "system", "content": DECOMPOSITION_SYSTEM_PROMPT},
                {"role": "user", "content": build_decomposition_prompt(question)},
            ]
        )
        payload = _parse_json_object(raw)
        descriptions = payload.get("descriptions")
        if not isinstance(descriptions, list):
            raise InvalidQAModelResponseError("LLM response is missing descriptions[]")
        cleaned = [str(item).strip() for item in descriptions if str(item).strip()]
        if not cleaned:
            raise InvalidQAModelResponseError("LLM returned no retrieval descriptions")
        return list(dict.fromkeys(cleaned))

    def answer_with_frames(
        self,
        *,
        question: str,
        descriptions: list[str],
        video_id: str,
        frame_groups: list[QAFrameGroup],
    ) -> dict[str, object]:
        content: list[dict[str, object]] = [
            {
                "type": "text",
                "text": build_answer_prompt(question, descriptions, video_id, frame_groups),
            }
        ]
        for group_index, group in enumerate(frame_groups, start=1):
            content.append(
                {
                    "type": "text",
                    "text": f"Group {group_index}, center frame {group.center_frame_id}",
                }
            )
            for frame in group.frames:
                content.append(
                    {"type": "text", "text": f"frame_id={frame.frame_id}"}
                )
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_data_url(frame.path)},
                    }
                )

        raw = self._chat(
            [
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]
        )
        return _parse_json_object(raw)

    def _chat(self, messages: list[dict[str, object]]) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(self.endpoint, headers=headers, json=body)
            response.raise_for_status()
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise InvalidQAModelResponseError("Unexpected chat completion response") from exc
        if not isinstance(content, str):
            raise InvalidQAModelResponseError("Chat completion content must be text")
        return content


def create_qa_model(settings: Settings) -> QAModel:
    if settings.qa_llm_backend == "openai_compatible":
        return OpenAICompatibleQAModel(settings)
    if settings.qa_llm_backend == "none":
        return UnconfiguredQAModel()
    raise QAModelNotConfiguredError(
        f"Unsupported QA_LLM_BACKEND: {settings.qa_llm_backend!r}"
    )


def _image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    media_type = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


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
