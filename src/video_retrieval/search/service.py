from __future__ import annotations

from pathlib import Path

from video_retrieval.config import Settings, get_settings
from video_retrieval.encoders.visual import VisualEncoder
from video_retrieval.models import SearchHit, SearchResponse
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

    def search_visual_text(
        self,
        query: str,
        *,
        limit: int = 10,
        vector_name: str = "siglip",
    ) -> SearchResponse:
        vector = self.visual.encode_text(query)
        hits = self.qdrant.search(vector, vector_name=vector_name, limit=limit)
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


def _rrf_fuse(rankings: list[list[SearchHit]], *, limit: int, k: int = 60) -> list[SearchHit]:
    scores: dict[str, float] = {}
    best: dict[str, SearchHit] = {}

    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            key = (
                f"{hit.video_id}|{hit.shot_index}|{hit.frame_index}|{hit.source}|{hit.text or ''}"
            )
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in best or hit.score > best[key].score:
                best[key] = hit

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
    fused: list[SearchHit] = []
    for key, score in ordered:
        hit = best[key].model_copy()
        hit.score = score
        fused.append(hit)
    return fused
