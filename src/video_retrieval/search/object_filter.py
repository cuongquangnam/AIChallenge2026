from __future__ import annotations

from video_retrieval.models import ObjectRequirement, SearchHit


def rerank_hits_by_objects(
    hits: list[SearchHit],
    requirements: list[ObjectRequirement],
    *,
    boost: float = 0.2,
    penalty: float = 0.1,
    limit: int | None = None,
) -> list[SearchHit]:
    """Softly rerank hits using indexed object counts, preserving unknown legacy points."""
    if not requirements:
        return hits[:limit] if limit is not None else list(hits)

    reranked: list[SearchHit] = []
    for hit in hits:
        payload = hit.payload or {}
        indexed = payload.get("objects_indexed") is True
        counts = payload.get("object_counts")
        if not indexed or not isinstance(counts, dict):
            reranked.append(hit)
            continue

        satisfied = 0
        matched: dict[str, bool] = {}
        for requirement in requirements:
            try:
                actual_count = int(counts.get(requirement.label, 0))
            except (TypeError, ValueError):
                actual_count = 0
            meets = actual_count >= requirement.min_count
            matched[requirement.label] = meets
            satisfied += int(meets)

        ratio = satisfied / len(requirements)
        factor = 1.0 + max(0.0, boost) * ratio - max(0.0, penalty) * (1.0 - ratio)
        reranked.append(
            hit.model_copy(
                update={
                    "score": hit.score * max(0.0, factor),
                    "payload": {
                        **payload,
                        "object_match": matched,
                        "object_match_ratio": ratio,
                    },
                }
            )
        )

    reranked.sort(key=lambda item: item.score, reverse=True)
    return reranked[:limit] if limit is not None else reranked
