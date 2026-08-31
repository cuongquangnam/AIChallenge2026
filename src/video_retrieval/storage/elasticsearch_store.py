from __future__ import annotations

from typing import Any

from elasticsearch import Elasticsearch, NotFoundError

from video_retrieval.config import Settings
from video_retrieval.models import SearchHit, TextDocument


class ElasticsearchStore:
    def __init__(self, settings: Settings, client: Elasticsearch | None = None):
        self.settings = settings
        self.client = client or Elasticsearch(settings.elasticsearch_url)
        self.index = settings.es_index

    def ensure_index(self) -> None:
        if self.client.indices.exists(index=self.index):
            return
        self.client.indices.create(
            index=self.index,
            mappings={
                "properties": {
                    "video_id": {"type": "keyword"},
                    "source": {"type": "keyword"},
                    "text": {"type": "text"},
                    "shot_index": {"type": "integer"},
                    "frame_index": {"type": "integer"},
                    "role": {"type": "keyword"},
                    "start_sec": {"type": "float"},
                    "end_sec": {"type": "float"},
                    "keyframe_path": {"type": "keyword"},
                }
            },
        )

    def index_documents(self, documents: list[TextDocument], *, refresh: bool = False) -> int:
        if not documents:
            return 0
        self.ensure_index()
        operations: list[dict[str, Any]] = []
        for doc in documents:
            operations.append({"index": {"_index": self.index, "_id": doc.doc_id}})
            operations.append(
                {
                    "video_id": doc.video_id,
                    "source": doc.source,
                    "text": doc.text,
                    "shot_index": doc.shot_index,
                    "frame_index": doc.frame_index,
                    "role": doc.role.value if doc.role else None,
                    "start_sec": doc.start_sec,
                    "end_sec": doc.end_sec,
                    "keyframe_path": doc.metadata.get("keyframe_path"),
                }
            )
        self.client.bulk(operations=operations, refresh=refresh)
        return len(documents)

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        source: str | None = None,
        video_id: str | None = None,
    ) -> list[SearchHit]:
        self.ensure_index()
        filters: list[dict[str, Any]] = []
        if source:
            filters.append({"term": {"source": source}})
        if video_id:
            filters.append({"term": {"video_id": video_id}})

        body: dict[str, Any] = {
            "size": limit,
            "query": {
                "bool": {
                    "must": [{"match": {"text": {"query": query, "operator": "and"}}}],
                    "filter": filters,
                }
            },
        }
        try:
            resp = self.client.search(index=self.index, body=body)
        except NotFoundError:
            return []

        hits: list[SearchHit] = []
        for item in resp["hits"]["hits"]:
            src = item.get("_source", {})
            hits.append(
                SearchHit(
                    video_id=src.get("video_id", ""),
                    score=float(item.get("_score") or 0.0),
                    source=f"text:{src.get('source', 'unknown')}",
                    shot_index=src.get("shot_index"),
                    frame_index=src.get("frame_index"),
                    role=src.get("role"),
                    timestamp_sec=src.get("start_sec"),
                    text=src.get("text"),
                    keyframe_path=src.get("keyframe_path"),
                    payload=src,
                )
            )
        return hits
