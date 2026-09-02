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


def hydrate_elasticsearch_index(settings: Settings) -> dict[str, int | str]:
    """Start Elasticsearch on Colab and load text data from ndjson or manifests."""
    ensure_elasticsearch(
        settings.elasticsearch_url,
        install_dir=Path(settings.colab_elasticsearch_install_dir),
        data_dir=settings.data_dir / "elasticsearch_data",
    )
    store = ElasticsearchStore(settings)
    ndjson_path = find_es_ndjson_export(settings.data_dir, settings.es_index)
    if ndjson_path is not None:
        imported = store.bulk_import_ndjson(ndjson_path)
        return {"source": "ndjson", "imported": imported, "path": ndjson_path.name}

    imported = store.import_from_manifests(settings.manifests_dir)
    return {"source": "manifests", "imported": imported}
