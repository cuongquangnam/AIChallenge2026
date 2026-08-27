from __future__ import annotations

import logging

from video_retrieval.config import Settings, get_settings
from video_retrieval.models import QAAnswerHit, QAResult, SearchHit
from video_retrieval.qa.frames import QAFrameSampler
from video_retrieval.qa.llm import QAModel, create_qa_model
from video_retrieval.qa.retrieval import QACandidateRetriever
from video_retrieval.search.kis import hits_to_submission_rows
from video_retrieval.search.service import SearchService
from video_retrieval.text.gemini_logging import log_gemini_failure

logger = logging.getLogger(__name__)


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
        limit: int = 24,
        group_count: int | None = None,
        frame_radius: int | None = None,
    ) -> QAResult:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")
        if limit < 1:
            raise ValueError("limit must be >= 1")

        try:
            descriptions = self.model.decompose_question(question)
            descriptions = list(
                dict.fromkeys(item.strip() for item in descriptions if item.strip())
            )
        except Exception as exc:
            log_gemini_failure(
                component="QA decompose",
                model=self.settings.gemini_model,
                exc=exc,
                query=question,
                fallback="use raw question as sole retrieval description",
            )
            descriptions = [question]
        if not descriptions:
            descriptions = [question]

        # Retrieve several videos so we can return a ranked multi-row list.
        top_videos = min(max(limit // 4, 3), 12)
        retrieval = self.retriever.retrieve(
            [question, *descriptions],
            top_videos=top_videos,
        )
        groups = self.sampler.sample(
            video_id=retrieval.video_id,
            candidates=retrieval.candidates,
            group_count=group_count or self.settings.qa_group_count,
            radius=self.settings.qa_frame_radius if frame_radius is None else frame_radius,
            stride=self.settings.qa_frame_stride,
            min_center_gap=self.settings.qa_min_center_gap,
        )

        answer = ""
        frame_id = groups[0].center_frame_id if groups else (
            retrieval.candidates[0].frame_index or 0
        )
        video_id = retrieval.video_id
        try:
            raw_answer = self.model.answer_with_frames(
                question=question,
                descriptions=descriptions,
                video_id=retrieval.video_id,
                frame_groups=groups,
            )
            video_id = str(raw_answer.get("video_id") or "").strip() or retrieval.video_id
            answer = str(raw_answer.get("answer") or "").strip()
            try:
                frame_id = int(raw_answer["frame_id"])
            except (KeyError, TypeError, ValueError):
                pass
        except Exception as exc:
            # Gemini outage / rate limit: still return ranked evidence for human QA.
            log_gemini_failure(
                component="QA answer_with_frames",
                model=self.settings.gemini_model,
                exc=exc,
                query=question,
                fallback="retrieval-only ranked hits (blank answer for human review)",
            )
            answer = ""

        if video_id != retrieval.video_id:
            video_id = retrieval.video_id
        allowed_frame_ids = {frame.frame_id for group in groups for frame in group.frames}
        if frame_id not in allowed_frame_ids and groups:
            frame_id = groups[0].center_frame_id

        hits = self._build_ranked_hits(
            answer=answer or "(review)",
            best_video_id=video_id,
            best_frame_id=int(frame_id),
            retrieval=retrieval,
            groups=groups,
            limit=limit,
        )
        display_answer = answer
        for hit in hits:
            if not answer:
                hit.answer = ""

        return QAResult(
            question=question,
            video_id=video_id,
            frame_id=int(frame_id),
            answer=display_answer,
            descriptions=descriptions,
            frame_groups=groups,
            hits=hits,
        )

    def _build_ranked_hits(
        self,
        *,
        answer: str,
        best_video_id: str,
        best_frame_id: int,
        retrieval,
        groups,
        limit: int,
    ) -> list[QAAnswerHit]:
        rows: list[QAAnswerHit] = []
        seen: set[tuple[str, int]] = set()

        def _add(
            video: str,
            frame: int,
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

        # 1) LLM-selected evidence first.
        best_ts = None
        for group in groups:
            for frame in group.frames:
                if frame.frame_id == best_frame_id:
                    best_ts = frame.timestamp_sec
                    break
        _add(best_video_id, best_frame_id, score=1.0, timestamp_sec=best_ts, source="qa_best")

        # 2) Other sampled evidence frames from the top video.
        for group in groups:
            for frame in group.frames:
                if len(rows) >= limit:
                    break
                _add(
                    best_video_id,
                    frame.frame_id,
                    score=0.5 + float(group.retrieval_score),
                    timestamp_sec=frame.timestamp_sec,
                    source="qa_evidence",
                )

        # 3) Retrieval candidates across top videos (same answer text).
        for bucket in retrieval.videos:
            for candidate in bucket.candidates:
                if len(rows) >= limit:
                    break
                if candidate.frame_index is None:
                    continue
                _add(
                    bucket.video_id,
                    int(candidate.frame_index),
                    score=float(candidate.score) + 0.1 * float(bucket.video_score),
                    timestamp_sec=candidate.timestamp_sec,
                    source="qa_retrieval",
                )

        # 4) Pad with nearby frames from the top hits (KIS-style).
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
                _add(video, frame, score=0.01, source="qa_pad")

        # Keep best answer first; then by score.
        head = rows[:1]
        rest = sorted(rows[1:], key=lambda row: row.score, reverse=True)
        ordered = head + rest
        return ordered[:limit]
