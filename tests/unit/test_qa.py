from __future__ import annotations

from types import SimpleNamespace

import pytest

from video_retrieval.models import SearchHit
from video_retrieval.qa.frames import QAFrameSampler
from video_retrieval.qa.llm import _parse_json_object
from video_retrieval.qa.retrieval import (
    QACandidate,
    QACandidateRetriever,
    QARetrievalResult,
)
from video_retrieval.qa.service import QAService
from tests.helpers import write_dummy_video


class _FakeVisual:
    def encode_text(self, text: str) -> list[float]:
        return [1.0, 0.0]


class _FakeES:
    def search(self, query: str, *, limit: int = 10) -> list[SearchHit]:
        return [
            SearchHit(
                video_id="awards",
                score=3.0,
                source="text:ocr",
                frame_index=100,
                timestamp_sec=10.0,
            ),
            SearchHit(video_id="other", score=2.0, source="text:asr", timestamp_sec=2.0),
        ]


class _FakeQdrant:
    def search(
        self,
        vector: list[float],
        *,
        vector_name: str,
        limit: int,
    ) -> list[SearchHit]:
        return [
            SearchHit(
                video_id="awards",
                score=0.9,
                source="visual:siglip",
                frame_index=100,
                timestamp_sec=10.0,
            )
        ]


class _FakeModel:
    def decompose_question(self, question: str) -> list[str]:
        return ["music awards ceremony", "people walking onto the stage"]

    def answer_with_frames(self, *, question, descriptions, video_id, frame_groups):
        return {
            "video_id": video_id,
            "frame_id": frame_groups[0].center_frame_id,
            "answer": "5",
        }


class _FakeRetriever:
    def retrieve(self, queries: list[str]) -> QARetrievalResult:
        assert queries[0].startswith("Trong video")
        assert "music awards ceremony" in queries
        return QARetrievalResult(
            video_id="awards",
            video_score=1.0,
            candidates=[
                QACandidate(
                    video_id="awards",
                    score=0.5,
                    frame_index=10,
                    sources=["text:ocr", "visual:siglip"],
                )
            ],
        )


@pytest.mark.unit
def test_qa_retriever_fuses_text_and_visual_on_same_frame() -> None:
    search = SimpleNamespace(es=_FakeES(), visual=_FakeVisual(), qdrant=_FakeQdrant())
    result = QACandidateRetriever(search, limit=10).retrieve(["music awards"])

    assert result.video_id == "awards"
    assert result.candidates[0].frame_index == 100
    assert result.candidates[0].sources == ["text:ocr", "visual:siglip"]


@pytest.mark.unit
def test_frame_sampler_extracts_neighboring_frames(settings) -> None:
    settings.ensure_dirs()
    write_dummy_video(
        settings.videos_dir / "awards.mp4",
        frames=30,
        fps=10,
        with_audio=False,
    )
    candidates = [
        QACandidate(video_id="awards", score=1.0, frame_index=10, sources=["visual:siglip"])
    ]

    groups = QAFrameSampler(settings).sample(
        video_id="awards",
        candidates=candidates,
        group_count=10,
        radius=2,
        stride=1,
        min_center_gap=5,
    )

    assert len(groups) == 1
    assert groups[0].center_frame_id == 10
    assert [frame.frame_id for frame in groups[0].frames] == [8, 9, 10, 11, 12]
    assert all(frame.path.exists() for frame in groups[0].frames)


@pytest.mark.unit
def test_qa_service_returns_video_frame_and_answer(settings) -> None:
    settings.ensure_dirs()
    write_dummy_video(
        settings.videos_dir / "awards.mp4",
        frames=30,
        fps=10,
        with_audio=False,
    )
    service = QAService(
        settings,
        model=_FakeModel(),
        search=SimpleNamespace(),  # type: ignore[arg-type]
        retriever=_FakeRetriever(),  # type: ignore[arg-type]
        sampler=QAFrameSampler(settings),
    )

    result = service.answer(
        "Trong video về lễ trao giải, có bao nhiêu người lên sân khấu?",
        group_count=10,
        frame_radius=2,
    )

    assert result.video_id == "awards"
    assert result.frame_id == 10
    assert result.answer == "5"
    assert len(result.frame_groups) == 1


@pytest.mark.unit
def test_parse_json_object_accepts_markdown_fence() -> None:
    payload = _parse_json_object('```json\n{"frame_id": 10, "answer": "5"}\n```')
    assert payload == {"frame_id": 10, "answer": "5"}
