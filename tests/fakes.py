from __future__ import annotations

from video_retrieval.models import SearchHit, TextDocument


class FakeElasticsearchStore:
    """In-process stand-in for ElasticsearchStore (offline integration)."""

    def __init__(self) -> None:
        self.docs: dict[str, TextDocument] = {}

    def ensure_index(self) -> None:
        return None

    def index_documents(self, documents: list[TextDocument], *, refresh: bool = False) -> int:
        for doc in documents:
            self.docs[doc.doc_id] = doc
        return len(documents)

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        source: str | None = None,
        video_id: str | None = None,
    ) -> list[SearchHit]:
        terms = [t.lower() for t in query.split() if t]
        hits: list[SearchHit] = []
        for doc in self.docs.values():
            if source and doc.source != source:
                continue
            if video_id and doc.video_id != video_id:
                continue
            text = doc.text.lower()
            if terms and not all(term in text for term in terms):
                continue
            hits.append(
                SearchHit(
                    video_id=doc.video_id,
                    score=1.0,
                    source=f"text:{doc.source}",
                    shot_index=doc.shot_index,
                    frame_index=doc.frame_index,
                    role=doc.role,
                    timestamp_sec=doc.start_sec,
                    text=doc.text,
                    keyframe_path=doc.metadata.get("keyframe_path"),
                    payload={"text": doc.text, "source": doc.source},
                )
            )
        return hits[:limit]
