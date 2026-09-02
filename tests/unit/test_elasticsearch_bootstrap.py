from pathlib import Path

import pytest

from video_retrieval.storage.elasticsearch_bootstrap import find_es_ndjson_export


@pytest.mark.unit
def test_find_es_ndjson_export_prefers_exact_match(tmp_path: Path) -> None:
    es_dir = tmp_path / "elasticsearch"
    es_dir.mkdir()
    (es_dir / "video_text.ndjson").write_text("", encoding="utf-8")
    (es_dir / "video_text_transnet.ndjson").write_text("", encoding="utf-8")

    found = find_es_ndjson_export(tmp_path, "video_text")
    assert found == es_dir / "video_text.ndjson"


@pytest.mark.unit
def test_find_es_ndjson_export_falls_back_to_prefix(tmp_path: Path) -> None:
    es_dir = tmp_path / "elasticsearch"
    es_dir.mkdir()
    (es_dir / "video_text_transnet.ndjson").write_text("", encoding="utf-8")

    found = find_es_ndjson_export(tmp_path, "video_text")
    assert found == es_dir / "video_text_transnet.ndjson"


@pytest.mark.unit
def test_find_es_ndjson_export_missing_dir(tmp_path: Path) -> None:
    assert find_es_ndjson_export(tmp_path, "video_text") is None
