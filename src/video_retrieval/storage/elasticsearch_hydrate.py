from __future__ import annotations

import logging
from pathlib import Path

from video_retrieval.config import Settings
from video_retrieval.storage.elasticsearch_bootstrap import (
    ensure_elasticsearch,
    find_es_ndjson_export,
)
from video_retrieval.storage.elasticsearch_store import ElasticsearchStore

logger = logging.getLogger(__name__)


def hydrate_elasticsearch_index(
    settings: Settings,
    *,
    progress: bool = False,
) -> dict[str, int | str]:
    """Start Elasticsearch on Colab and load text data from ndjson or manifests."""
    if progress:
        print(f"[es] ensuring Elasticsearch at {settings.elasticsearch_url} ...", flush=True)
    ensure_elasticsearch(
        settings.elasticsearch_url,
        install_dir=Path(settings.colab_elasticsearch_install_dir),
        data_dir=settings.data_dir / "elasticsearch_data",
    )
    if progress:
        print("[es] Elasticsearch ready", flush=True)

    store = ElasticsearchStore(settings)
    ndjson_path = find_es_ndjson_export(settings.data_dir, settings.es_index)
    if ndjson_path is not None:
    if progress:
        print(f"[es] loading ndjson export: {ndjson_path}", flush=True)
        print("[es] bulk import can take several minutes on Colab ...", flush=True)
    imported = store.bulk_import_ndjson(ndjson_path, progress=progress)
        return {"source": "ndjson", "imported": imported, "path": ndjson_path.name}

    if progress:
        print(f"[es] no ndjson found; importing manifests from {settings.manifests_dir}", flush=True)
    imported = store.import_from_manifests(settings.manifests_dir)
    if progress:
        print(f"[es] imported {imported} doc(s) from manifests", flush=True)
    return {"source": "manifests", "imported": imported}
