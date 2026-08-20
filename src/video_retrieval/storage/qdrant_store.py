from __future__ import annotations

import hashlib
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from video_retrieval.config import Settings
from video_retrieval.models import SearchHit, VisualEmbedding


class QdrantStore:
    """SigLIP collection: one cosine vector per keyframe point."""

    VECTOR_NAME = "siglip"

    def __init__(self, settings: Settings, client: QdrantClient | None = None):
        self.settings = settings
        if client is not None:
            self.client = client
        elif settings.qdrant_url in {":memory:", "memory"}:
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(
                url=settings.qdrant_url,
                timeout=settings.qdrant_timeout,
                check_compatibility=False,
            )
        self.collection = settings.qdrant_collection

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            if self._vector_names() == {self.VECTOR_NAME}:
                return
            print(
                f"Recreating Qdrant collection {self.collection!r} as SigLIP-only "
                f"(was {sorted(self._vector_names() or [])}). Re-index visual."
            )
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                self.VECTOR_NAME: qm.VectorParams(
                    size=self.settings.siglip_dim,
                    distance=qm.Distance.COSINE,
                ),
            },
        )

    def upsert_embeddings(self, embeddings: list[VisualEmbedding], *, batch_size: int = 64) -> int:
        if not embeddings:
            return 0
        self.ensure_collection()
        points = [self._to_point(emb) for emb in embeddings]
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            self.client.upsert(collection_name=self.collection, points=batch)
        return len(points)

    def search(
        self,
        query_vector: list[float],
        *,
        vector_name: str = VECTOR_NAME,
        limit: int = 10,
        video_id: str | None = None,
    ) -> list[SearchHit]:
        if vector_name != self.VECTOR_NAME:
            raise ValueError(f"Only {self.VECTOR_NAME!r} vectors are stored")
        self.ensure_collection()
        query_filter = None
        if video_id:
            query_filter = qm.Filter(
                must=[qm.FieldCondition(key="video_id", match=qm.MatchValue(value=video_id))]
            )

        result = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            using=self.VECTOR_NAME,
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
                    source=f"visual:{self.VECTOR_NAME}",
                    shot_index=payload.get("shot_index"),
                    frame_index=payload.get("frame_index"),
                    role=payload.get("role"),
                    timestamp_sec=payload.get("timestamp_sec"),
                    keyframe_path=payload.get("keyframe_path"),
                    payload=payload,
                )
            )
        return hits

    def count_for_video(self, video_id: str) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        result = self.client.count(
            collection_name=self.collection,
            count_filter=qm.Filter(
                must=[qm.FieldCondition(key="video_id", match=qm.MatchValue(value=video_id))]
            ),
            exact=True,
        )
        return int(result.count)

    def _vector_names(self) -> set[str] | None:
        info = self.client.get_collection(self.collection)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            return set(vectors)
        return None

    def _to_point(self, emb: VisualEmbedding) -> qm.PointStruct:
        kf = emb.keyframe
        point_id = _stable_uuid(f"{kf.video_id}:{kf.shot_index}:{kf.role.value}:{kf.frame_index}")
        return qm.PointStruct(
            id=point_id,
            vector={self.VECTOR_NAME: emb.siglip},
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
