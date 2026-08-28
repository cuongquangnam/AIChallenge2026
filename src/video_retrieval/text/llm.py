from __future__ import annotations

from typing import Any, Protocol


class LLMClient(Protocol):
    """Text / multimodal generation for planners, extractors, QA, OCR, etc."""

    model: str

    def generate_text(
        self,
        prompt: str,
        *,
        json_response: bool = False,
        component: str = "llm",
    ) -> str: ...

    def generate_parts(
        self,
        parts: list[Any],
        *,
        json_response: bool = False,
        component: str = "llm",
    ) -> str: ...
