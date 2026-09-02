from __future__ import annotations

import json
from pathlib import Path

from video_retrieval.models import FrameRole, SearchHit, TextDocument


class MemoryTextStore:
    """In-process text search store for environments without Elasticsearch.

    Indexes are hydrated from ``manifests/*.json`` on startup so search works
    across Colab sessions when ``DATA_DIR`` points at a persisted folder.
    """

    def __init__(self, *, manifests_dir: Path | None = None) -> None:
        self.docs: dict[str, TextDocument] = {}
        if manifests_dir is not None:
            self.hydrate_from_manifests(manifests_dir)

    def ensure_index(self) -> None:
        return None

    def index_documents(self, documents: list[TextDocument]) -> int:
        for doc in documents:
            self.docs[doc.doc_id] = doc
        return len(documents)

    def hydrate_from_manifests(self, manifests_dir: Path) -> int:
        """Load OCR/ASR documents saved by the offline indexer."""
        if not manifests_dir.exists():
            return 0
        loaded = 0
        for manifest_path in sorted(manifests_dir.glob("*.json")):
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            text_docs = payload.get("text_docs")
            if not isinstance(text_docs, list):
                continue
            for item in text_docs:
                if not isinstance(item, dict):
                    continue
                try:
                    doc = TextDocument.model_validate(item)
                except Exception:
                    continue
                self.docs[doc.doc_id] = doc
                loaded += 1
        return loaded

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        source: str | None = None,
        video_id: str | None = None,
        strict: bool = False,
    ) -> list[SearchHit]:
        terms = [term.lower() for term in query.split() if term]
        hits: list[SearchHit] = []
        for doc in self.docs.values():
            if source and doc.source != source:
                continue
            if video_id and doc.video_id != video_id:
                continue
            text = doc.text.lower()
            if terms:
                if strict:
                    if not all(term in text for term in terms):
                        continue
                elif not any(term in text for term in terms):
                    continue
            hits.append(
                SearchHit(
                    video_id=doc.video_id,
                    score=_score_text(text, terms),
                    source=f"text:{doc.source}",
                    shot_index=doc.shot_index,
                    frame_index=doc.frame_index,
                    role=doc.role if isinstance(doc.role, FrameRole) else None,
                    timestamp_sec=doc.start_sec,
                    text=doc.text,
                    keyframe_path=doc.metadata.get("keyframe_path"),
                    payload={
                        "text": doc.text,
                        "source": doc.source,
                        "start_sec": doc.start_sec,
                        "end_sec": doc.end_sec,
                    },
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]


def _score_text(text: str, terms: list[str]) -> float:
    if not terms:
        return 1.0
    matched = sum(1 for term in terms if term in text)
    return float(matched) / len(terms)
