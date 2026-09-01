from __future__ import annotations

import hashlib
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from video_retrieval.config import Settings
from video_retrieval.models import FrameObjectDetections, SearchHit, VisualEmbedding


class QdrantStore:
    """Named-vector collection: siglip + beit3 per keyframe point."""

    def __init__(self, settings: Settings, client: QdrantClient | None = None):
        self.settings = settings
        if client is not None:
            self.client = client
        elif settings.qdrant_url in {":memory:", "memory"}:
            self.client = QdrantClient(location=":memory:")
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

    def upsert_embeddings(
        self,
        embeddings: list[VisualEmbedding],
        *,
        object_detections: list[FrameObjectDetections] | None = None,
    ) -> int:
        if not embeddings:
            return 0
        self.ensure_collection()
        object_map = _object_payload_map(object_detections or [])
        points = [self._to_point(emb, object_map=object_map) for emb in embeddings]
        self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def set_object_payload(self, frames: list[FrameObjectDetections]) -> int:
        """Attach object metadata to already indexed visual points."""
        if not frames:
            return 0
        self.ensure_collection()
        for frame in frames:
            keyframe = frame.keyframe
            point_id = _point_id(keyframe)
            self.client.set_payload(
                collection_name=self.collection,
                points=[point_id],
                payload=_object_payload(frame),
            )
        return len(frames)

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

    def _to_point(
        self,
        emb: VisualEmbedding,
        *,
        object_map: dict[tuple[str, int, str, int], dict[str, Any]] | None = None,
    ) -> qm.PointStruct:
        kf = emb.keyframe
        point_id = _point_id(kf)
        object_payload = (object_map or {}).get(_keyframe_key(kf), {})
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
                **object_payload,
            },
        )


def _keyframe_key(keyframe) -> tuple[str, int, str, int]:
    return (
        keyframe.video_id,
        keyframe.shot_index,
        keyframe.role.value,
        keyframe.frame_index,
    )


def _point_id(keyframe) -> str:
    video_id, shot_index, role, frame_index = _keyframe_key(keyframe)
    return _stable_uuid(f"{video_id}:{shot_index}:{role}:{frame_index}")


def _object_payload(frame: FrameObjectDetections) -> dict[str, Any]:
    counts = frame.counts
    return {
        "objects_indexed": True,
        "objects": sorted(counts),
        "object_counts": counts,
    }


def _object_payload_map(
    frames: list[FrameObjectDetections],
) -> dict[tuple[str, int, str, int], dict[str, Any]]:
    return {_keyframe_key(frame.keyframe): _object_payload(frame) for frame in frames}


def _stable_uuid(value: str) -> str:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return str(uuid.UUID(hex=digest))
