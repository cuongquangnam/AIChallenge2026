from __future__ import annotations

from video_retrieval.config import Settings, get_settings
from video_retrieval.models import QAResult
from video_retrieval.qa.frames import QAFrameSampler
from video_retrieval.qa.llm import InvalidQAModelResponseError, QAModel, create_qa_model
from video_retrieval.qa.retrieval import QACandidateRetriever
from video_retrieval.search.service import SearchService


class QAService:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model: QAModel | None = None,
        search: SearchService | None = None,
        retriever: QACandidateRetriever | None = None,
        sampler: QAFrameSampler | None = None,
    ):
        self.settings = settings or get_settings()
        self.model = model or create_qa_model(self.settings)
        self.search = search or SearchService(self.settings)
        self.retriever = retriever or QACandidateRetriever(
            self.search,
            limit=self.settings.qa_retrieval_limit,
        )
        self.sampler = sampler or QAFrameSampler(self.settings)

    def answer(
        self,
        question: str,
        *,
        group_count: int | None = None,
        frame_radius: int | None = None,
    ) -> QAResult:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")

        descriptions = self.model.decompose_question(question)
        descriptions = list(dict.fromkeys(item.strip() for item in descriptions if item.strip()))
        if not descriptions:
            raise InvalidQAModelResponseError("Question decomposition returned no descriptions")

        # Keep the full question as one ranking in addition to its simpler parts.
        retrieval = self.retriever.retrieve([question, *descriptions])
        groups = self.sampler.sample(
            video_id=retrieval.video_id,
            candidates=retrieval.candidates,
            group_count=group_count or self.settings.qa_group_count,
            radius=self.settings.qa_frame_radius if frame_radius is None else frame_radius,
            stride=self.settings.qa_frame_stride,
            min_center_gap=self.settings.qa_min_center_gap,
        )
        raw_answer = self.model.answer_with_frames(
            question=question,
            descriptions=descriptions,
            video_id=retrieval.video_id,
            frame_groups=groups,
        )
        video_id = str(raw_answer.get("video_id") or "").strip()
        answer = str(raw_answer.get("answer") or "").strip()
        try:
            frame_id = int(raw_answer["frame_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidQAModelResponseError("LLM response has an invalid frame_id") from exc

        if video_id != retrieval.video_id:
            raise InvalidQAModelResponseError(
                f"LLM returned video_id={video_id!r}; expected {retrieval.video_id!r}"
            )
        allowed_frame_ids = {frame.frame_id for group in groups for frame in group.frames}
        if frame_id not in allowed_frame_ids:
            raise InvalidQAModelResponseError(
                f"LLM returned frame_id={frame_id}, which was not supplied as evidence"
            )
        if not answer:
            raise InvalidQAModelResponseError("The supplied frames were insufficient to answer")

        return QAResult(
            question=question,
            video_id=video_id,
            frame_id=frame_id,
            answer=answer,
            descriptions=descriptions,
            frame_groups=groups,
        )
