from __future__ import annotations

from pathlib import Path

from video_retrieval.config import Settings, get_settings
from video_retrieval.encoders.visual import VisualEncoder
from video_retrieval.models import SearchHit, SearchResponse
from video_retrieval.models import Task2RetrievalResponse
from video_retrieval.storage.elasticsearch_store import ElasticsearchStore
from video_retrieval.storage.qdrant_store import QdrantStore


class SearchService:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        visual: VisualEncoder | None = None,
        qdrant: QdrantStore | None = None,
        es: ElasticsearchStore | None = None,
    ):
        self.settings = settings or get_settings()
        self.visual = visual or VisualEncoder(self.settings)
        self.qdrant = qdrant or QdrantStore(self.settings)
        self.es = es or ElasticsearchStore(self.settings)

    def search_text(self, query: str, *, limit: int = 10) -> SearchResponse:
        hits = self.es.search(query, limit=limit)
        return SearchResponse(query=query, mode="text", hits=hits)

    def search_text_filtered(
        self,
        query: str,
        *,
        limit: int = 10,
        source: str | None = None,
        video_id: str | None = None,
    ) -> SearchResponse:
        hits = self.es.search(query, limit=limit, source=source, video_id=video_id)
        mode = "text"
        if source:
            mode = f"text:{source}"
        return SearchResponse(query=query, mode=mode, hits=hits)

    def search_visual_text(
        self,
        query: str,
        *,
        limit: int = 10,
        vector_name: str = "siglip",
        video_id: str | None = None,
    ) -> SearchResponse:
        if vector_name != "siglip":
            raise ValueError("Text-to-visual search only supports the SigLIP text embedding space")
        vector = self.visual.encode_text(query)
        hits = self.qdrant.search(vector, vector_name=vector_name, limit=limit, video_id=video_id)
        return SearchResponse(query=query, mode=f"visual_text:{vector_name}", hits=hits)

    def search_image(
        self,
        image_path: Path,
        *,
        limit: int = 10,
        vector_name: str = "siglip",
    ) -> SearchResponse:
        siglip, beit3 = self.visual.encode_image(Path(image_path))
        vector = siglip if vector_name == "siglip" else beit3
        hits = self.qdrant.search(vector, vector_name=vector_name, limit=limit)
        return SearchResponse(query=str(image_path), mode=f"visual_image:{vector_name}", hits=hits)

    def search_hybrid(self, query: str, *, limit: int = 10) -> SearchResponse:
        text_hits = self.es.search(query, limit=limit)
        visual_hits = self.qdrant.search(
            self.visual.encode_text(query),
            vector_name="siglip",
            limit=limit,
        )
        merged = _rrf_fuse([text_hits, visual_hits], limit=limit)
        return SearchResponse(query=query, mode="hybrid", hits=merged)

    def search_hybrid_filtered(
        self,
        query: str,
        *,
        limit: int = 10,
        source: str | None = None,
        video_id: str | None = None,
    ) -> SearchResponse:
        text_hits = self.es.search(query, limit=limit, source=source, video_id=video_id)
        visual_hits = self.qdrant.search(
            self.visual.encode_text(query),
            vector_name="siglip",
            limit=limit,
            video_id=video_id,
        )
        merged = _rrf_fuse([text_hits, visual_hits], limit=limit)
        return SearchResponse(query=query, mode="hybrid", hits=merged)

    def retrieve_task2_candidates(
        self,
        *,
        video_id: str | None = None,
        candidates_per_query: int = 20,
        group_limit: int = 10,
        max_gap_sec: float = 10.0,
        max_gap_frames: int = 10,
        context_radius_frames: int = 5,
    ) -> Task2RetrievalResponse:
        """Return top evidence windows for the music-award Task 2 question."""
        from video_retrieval.search.task2 import retrieve_task2_candidates

        return retrieve_task2_candidates(
            self,
            video_id=video_id,
            candidates_per_query=candidates_per_query,
            group_limit=group_limit,
            max_gap_sec=max_gap_sec,
            max_gap_frames=max_gap_frames,
            context_radius_frames=context_radius_frames,
            manifests_dir=self.settings.data_dir / "manifests",
        )


def _rrf_fuse(rankings: list[list[SearchHit]], *, limit: int, k: int = 60) -> list[SearchHit]:
    scores: dict[str, float] = {}
    best: dict[str, SearchHit] = {}
    evidence: dict[str, list[dict[str, object]]] = {}

    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            key = _fusion_key(hit)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in best or hit.score > best[key].score:
                best[key] = hit
            evidence.setdefault(key, []).append(_hit_evidence(hit))

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
    fused: list[SearchHit] = []
    for key, score in ordered:
        hit = best[key].model_copy()
        hit.score = score
        hit.source = "hybrid"
        hit.payload = {**hit.payload, "evidence": evidence.get(key, [])}
        for item in evidence.get(key, []):
            if hit.text is None and item.get("text"):
                hit.text = str(item["text"])
            if hit.keyframe_path is None and item.get("keyframe_path"):
                hit.keyframe_path = str(item["keyframe_path"])
            if hit.timestamp_sec is None and item.get("timestamp_sec") is not None:
                hit.timestamp_sec = float(item["timestamp_sec"])
        fused.append(hit)
    return fused


def _fusion_key(hit: SearchHit) -> str:
    if hit.shot_index is not None:
        return f"{hit.video_id}|shot:{hit.shot_index}"
    if hit.timestamp_sec is not None:
        return f"{hit.video_id}|time:{hit.timestamp_sec:.1f}|{hit.source}"
    return f"{hit.video_id}|doc:{hit.source}|{hit.text or hit.keyframe_path or ''}"


def _hit_evidence(hit: SearchHit) -> dict[str, object]:
    return {
        "source": hit.source,
        "score": hit.score,
        "shot_index": hit.shot_index,
        "frame_index": hit.frame_index,
        "timestamp_sec": hit.timestamp_sec,
        "text": hit.text,
        "keyframe_path": hit.keyframe_path,
    }
