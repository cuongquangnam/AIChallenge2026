from __future__ import annotations

from dataclasses import dataclass, field

from video_retrieval.models import SearchHit
from video_retrieval.search.service import SearchService


class NoQACandidatesError(RuntimeError):
    pass


@dataclass
class QACandidate:
    video_id: str
    score: float
    frame_index: int | None = None
    timestamp_sec: float | None = None
    sources: list[str] = field(default_factory=list)


@dataclass
class QARetrievalResult:
    video_id: str
    video_score: float
    candidates: list[QACandidate]


class QACandidateRetriever:
    """Fuse text/OCR/ASR and visual rankings, first by video then by frame."""

    def __init__(self, search: SearchService, *, limit: int = 50, rrf_k: int = 60):
        self.search = search
        self.limit = limit
        self.rrf_k = rrf_k

    def retrieve(self, queries: list[str]) -> QARetrievalResult:
        queries = [query.strip() for query in queries if query.strip()]
        if not queries:
            raise NoQACandidatesError("No retrieval queries were provided")

        rankings: list[list[SearchHit]] = []
        for query in queries:
            rankings.append(self.search.es.search(query, limit=self.limit))
            vector = self.search.visual.encode_text(query)
            rankings.append(
                self.search.qdrant.search(vector, vector_name="siglip", limit=self.limit)
            )

        video_scores = self._score_videos(rankings)
        if not video_scores:
            raise NoQACandidatesError("OCR/ASR and visual retrieval returned no candidates")
        video_id, video_score = max(video_scores.items(), key=lambda item: item[1])

        frame_scores: dict[tuple[str, int], float] = {}
        frame_hits: dict[tuple[str, int], QACandidate] = {}
        for ranking in rankings:
            for rank, hit in enumerate(ranking):
                if hit.video_id != video_id:
                    continue
                key = _candidate_key(hit)
                contribution = 1.0 / (self.rrf_k + rank + 1)
                frame_scores[key] = frame_scores.get(key, 0.0) + contribution
                candidate = frame_hits.get(key)
                if candidate is None:
                    candidate = QACandidate(
                        video_id=video_id,
                        score=0.0,
                        frame_index=hit.frame_index,
                        timestamp_sec=hit.timestamp_sec,
                    )
                    frame_hits[key] = candidate
                if hit.source not in candidate.sources:
                    candidate.sources.append(hit.source)

        candidates = []
        for key, score in sorted(frame_scores.items(), key=lambda item: item[1], reverse=True):
            candidate = frame_hits[key]
            candidate.score = score
            candidates.append(candidate)
        if not candidates:
            raise NoQACandidatesError(f"No frame candidates were found for video {video_id}")
        return QARetrievalResult(
            video_id=video_id,
            video_score=video_score,
            candidates=candidates,
        )

    def _score_videos(self, rankings: list[list[SearchHit]]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for ranking in rankings:
            best_in_ranking: dict[str, float] = {}
            for rank, hit in enumerate(ranking):
                if not hit.video_id:
                    continue
                contribution = 1.0 / (self.rrf_k + rank + 1)
                best_in_ranking[hit.video_id] = max(
                    best_in_ranking.get(hit.video_id, 0.0), contribution
                )
            for video_id, contribution in best_in_ranking.items():
                scores[video_id] = scores.get(video_id, 0.0) + contribution
        return scores


def _candidate_key(hit: SearchHit) -> tuple[str, int]:
    if hit.frame_index is not None:
        return ("frame", hit.frame_index)
    # ASR hits have a time range rather than a frame id. Milliseconds are enough
    # to deduplicate the same segment across decomposed retrieval queries.
    timestamp_ms = round((hit.timestamp_sec or 0.0) * 1000)
    return ("time", timestamp_ms)
