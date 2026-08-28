from __future__ import annotations

import logging
from pathlib import Path

from video_retrieval.config import Settings
from video_retrieval.events.extractor import EventChainExtractor
from video_retrieval.events.pipeline import EventChainTaskBase
from video_retrieval.events.plan_utils import event_description, questioned_frame
from video_retrieval.events.searcher import EventChainSearcher
from video_retrieval.models import EventChain, QAAnswerHit, QAResult, QAResultItem, SearchHit
from video_retrieval.qa.frames import QAFrameSampler
from video_retrieval.qa.llm import QAModel, create_qa_model
from video_retrieval.search.kis import hits_to_submission_rows
from video_retrieval.search.service import SearchService
from video_retrieval.text.gemini_logging import log_gemini_failure

logger = logging.getLogger(__name__)


class QAService(EventChainTaskBase):
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model: QAModel | None = None,
        search: SearchService | None = None,
        extractor: EventChainExtractor | None = None,
        chain_search: EventChainSearcher | None = None,
        sampler: QAFrameSampler | None = None,
    ):
        super().__init__(settings, search=search, extractor=extractor, chain_search=chain_search)
        self.model = model or create_qa_model(self.settings)
        self.sampler = sampler or QAFrameSampler(self.settings)

    def answer(
        self,
        question: str,
        *,
        limit: int = 24,
        frame_radius: int | None = None,
    ) -> QAResult:
        if limit < 1:
            raise ValueError("limit must be >= 1")

        plan = self.extract_events(
            question,
            "qa",
            empty_message="No events extracted from question",
        )
        chains = self.search_event_chains(
            plan,
            top_chains=limit,
            top_videos=min(max(limit // 4, 3), 12),
        )
        if not chains:
            raise RuntimeError("No aligned event chains found for this question")

        results = self._answer_chains(
            question,
            plan,
            chains,
            frame_radius=frame_radius,
        )
        best = results[0]
        hits = self._build_ranked_hits(results=results, limit=limit)
        descriptions = [
            event.description or event.visual or event.ocr
            for event in plan.events
            if (event.description or event.visual or event.ocr)
        ]

        return QAResult(
            question=question.strip(),
            video_id=best.chain.video_id,
            frame_id=best.questioned_frame_id,
            answer=best.answer,
            plan=plan,
            results=results,
            hits=hits,
            descriptions=descriptions,
            frame_groups=[],
        )

    def _answer_chains(
        self,
        question: str,
        plan,
        chains: list[EventChain],
        *,
        frame_radius: int | None,
    ) -> list[QAResultItem]:
        results: list[QAResultItem] = []
        radius = (
            self.settings.qa_frame_radius if frame_radius is None else frame_radius
        )

        for chain in chains:
            q_event_id, q_frame = questioned_frame(plan, chain)
            answer = ""
            try:
                image_path = self._resolve_question_image(chain, q_frame, radius=radius)
                answer = self.model.answer_single_frame(
                    question=question,
                    video_id=chain.video_id,
                    frame_id=q_frame,
                    image_path=image_path,
                    event_description=event_description(plan, q_event_id),
                )
            except Exception as exc:
                log_gemini_failure(
                    component="QA answer_single_frame",
                    model=self.settings.gemini_model,
                    exc=exc,
                    query=question,
                    fallback="blank answer for chain result",
                )
                answer = ""

            results.append(
                QAResultItem(
                    chain=chain,
                    answer=answer,
                    questioned_event_id=q_event_id,
                    questioned_frame_id=q_frame,
                )
            )
        return results

    def _resolve_question_image(
        self,
        chain: EventChain,
        frame_id: int,
        *,
        radius: int,
    ) -> Path | None:
        for event in chain.events:
            if event.frame_index == frame_id and event.keyframe_path:
                path = Path(event.keyframe_path)
                if path.is_file():
                    return path
        if radius <= 0:
            return None
        try:
            from video_retrieval.qa.retrieval import QACandidate

            candidates = [
                QACandidate(
                    video_id=chain.video_id,
                    score=1.0,
                    frame_index=frame_id,
                )
            ]
            groups = self.sampler.sample(
                video_id=chain.video_id,
                candidates=candidates,
                group_count=1,
                radius=radius,
                stride=self.settings.qa_frame_stride,
                min_center_gap=0,
            )
            for group in groups:
                for frame in group.frames:
                    if frame.frame_id == frame_id and frame.path.is_file():
                        return frame.path
        except Exception as exc:
            logger.warning(
                "QA frame extract failed for %s f%s: %r",
                chain.video_id,
                frame_id,
                exc,
            )
        return None

    def _build_ranked_hits(
        self,
        *,
        results: list[QAResultItem],
        limit: int,
    ) -> list[QAAnswerHit]:
        rows: list[QAAnswerHit] = []
        seen: set[tuple[str, int]] = set()

        def _add(
            video: str,
            frame: int,
            answer: str,
            *,
            score: float,
            timestamp_sec: float | None = None,
            source: str = "qa",
        ) -> None:
            key = (video, frame)
            if frame < 0 or key in seen:
                return
            seen.add(key)
            rows.append(
                QAAnswerHit(
                    video_id=video,
                    frame_id=frame,
                    answer=answer,
                    score=score,
                    timestamp_sec=timestamp_sec,
                    source=source,
                )
            )

        for rank, item in enumerate(results):
            chain = item.chain
            answer = item.answer
            base_score = float(chain.score) + 10.0 * (len(results) - rank)

            q_event = next(
                (ev for ev in chain.events if ev.event_id == item.questioned_event_id),
                None,
            )
            q_ts = q_event.timestamp_sec if q_event else None
            _add(
                chain.video_id,
                item.questioned_frame_id,
                answer,
                score=base_score + 1.0,
                timestamp_sec=q_ts,
                source="qa_question",
            )

            for event in chain.events:
                if len(rows) >= limit:
                    break
                _add(
                    chain.video_id,
                    event.frame_index,
                    answer,
                    score=base_score + event.score * 0.01,
                    timestamp_sec=event.timestamp_sec,
                    source=f"qa_chain:{event.event_id}",
                )

        if len(rows) < limit and rows:
            seed_hits = [
                SearchHit(
                    video_id=row.video_id,
                    score=row.score,
                    source=row.source,
                    frame_index=row.frame_id,
                    timestamp_sec=row.timestamp_sec,
                )
                for row in rows
            ]
            try:
                padded = hits_to_submission_rows(seed_hits, limit=limit)
            except ValueError:
                padded = [(row.video_id, row.frame_id) for row in rows]
            for video, frame in padded:
                if len(rows) >= limit:
                    break
                answer = rows[0].answer if rows else ""
                _add(video, frame, answer, score=0.01, source="qa_pad")

        return rows[:limit]
