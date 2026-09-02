from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from video_retrieval.config import Settings
from video_retrieval.models import SearchHit, VisualEmbedding
from video_retrieval.storage.backends import is_local_qdrant, qdrant_storage_path


class QdrantStore:
    """Named-vector collection: siglip + beit3 per keyframe point."""

    def __init__(self, settings: Settings, client: QdrantClient | None = None):
        self.settings = settings
        if client is not None:
            self.client = client
        elif settings.qdrant_url in {":memory:", "memory"}:
            self.client = QdrantClient(location=":memory:")
        elif is_local_qdrant(settings.qdrant_url):
            storage_path = qdrant_storage_path(settings)
            Path(storage_path).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=storage_path)
        else:
            self.client = QdrantClient(url=settings.qdrant_url)
        self.collection = settings.qdrant_collection

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                "siglip": qm.VectorParams(
                    size=self.settings.siglip_dim,
                    distance=qm.Distance.COSINE,
                ),
                "beit3": qm.VectorParams(
                    size=self.settings.beit3_dim,
                    distance=qm.Distance.COSINE,
                ),
            },
        )

    def upsert_embeddings(self, embeddings: list[VisualEmbedding]) -> int:
        if not embeddings:
            return 0
        self.ensure_collection()
        points = [self._to_point(emb) for emb in embeddings]
        self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def search(
        self,
        query_vector: list[float],
        *,
        vector_name: str = "siglip",
        limit: int = 10,
        video_id: str | None = None,
    ) -> list[SearchHit]:
        self.ensure_collection()
        query_filter = None
        if video_id:
            query_filter = qm.Filter(
                must=[qm.FieldCondition(key="video_id", match=qm.MatchValue(value=video_id))]
            )

        result = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            using=vector_name,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        hits: list[SearchHit] = []
        for point in result.points:
            payload: dict[str, Any] = point.payload or {}
            hits.append(
                SearchHit(
                    video_id=str(payload.get("video_id", "")),
                    score=float(point.score or 0.0),
                    source=f"visual:{vector_name}",
                    shot_index=payload.get("shot_index"),
                    frame_index=payload.get("frame_index"),
                    role=payload.get("role"),
                    timestamp_sec=payload.get("timestamp_sec"),
                    keyframe_path=payload.get("keyframe_path"),
                    payload=payload,
                )
            )
        return hits

    def _to_point(self, emb: VisualEmbedding) -> qm.PointStruct:
        kf = emb.keyframe
        point_id = _stable_uuid(f"{kf.video_id}:{kf.shot_index}:{kf.role.value}:{kf.frame_index}")
        return qm.PointStruct(
            id=point_id,
            vector={"siglip": emb.siglip, "beit3": emb.beit3},
            payload={
                "video_id": kf.video_id,
                "shot_index": kf.shot_index,
                "role": kf.role.value,
                "frame_index": kf.frame_index,
                "timestamp_sec": kf.timestamp_sec,
                "keyframe_path": str(kf.path),
            },
        )


def _stable_uuid(value: str) -> str:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return str(uuid.UUID(hex=digest))
