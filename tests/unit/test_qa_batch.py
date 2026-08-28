"""Unit tests for batched QA VLM calls."""

from unittest.mock import patch

import pytest

from video_retrieval.config import Settings
from video_retrieval.qa.llm import GeminiQAModel, QASingleFrameRequest, UnconfiguredQAModel
from video_retrieval.qa.prompts import build_batch_frame_answer_prompt


@pytest.mark.unit
def test_build_batch_frame_answer_prompt_lists_candidates() -> None:
    text = build_batch_frame_answer_prompt(
        "What number?",
        [(0, "v1", 10, "fish on scale"), (1, "v2", 20, "")],
    )
    assert "chain_index=0" in text
    assert "chain_index=1" in text
    assert "What number?" in text


@pytest.mark.unit
def test_answer_single_frames_batch_parses_response() -> None:
    settings = Settings(gemini_api_key="test-key")
    with patch.object(GeminiQAModel, "__init__", lambda self, _settings, **kwargs: None):
        model = GeminiQAModel(settings)
        model._types = type(
            "Types",
            (),
            {
                "Part": type(
                    "Part",
                    (),
                    {"from_text": staticmethod(lambda *, text: text)},
                )
            },
        )()
    with patch.object(
        model,
        "_generate_parts",
        return_value=(
            '{"answers":['
            '{"chain_index":0,"video_id":"v1","frame_id":10,"answer":"42"},'
            '{"chain_index":1,"video_id":"v2","frame_id":20,"answer":"7"}'
            "]}"
        ),
    ):
        out = model.answer_single_frames_batch(
            question="reading?",
            items=[
                QASingleFrameRequest(0, "v1", 10, None),
                QASingleFrameRequest(1, "v2", 20, None),
            ],
        )
    assert out == {0: "42", 1: "7"}


@pytest.mark.unit
def test_unconfigured_batch_raises() -> None:
    model = UnconfiguredQAModel()
    with pytest.raises(Exception):
        model.answer_single_frames_batch(
            question="q",
            items=[QASingleFrameRequest(0, "v1", 1, None)],
        )
