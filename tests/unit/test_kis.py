from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from video_retrieval.models import SearchHit
from video_retrieval.search.kis import (
    hits_to_submission_rows,
    load_queries,
    package_kis_zip,
    write_kis_csv,
)


@pytest.mark.unit
def test_load_queries(tmp_path: Path) -> None:
    path = tmp_path / "q.json"
    path.write_text(
        json.dumps({"query-p1-1-kis": "four astronauts", "query-p1-2-kis": "tiger cubs"}),
        encoding="utf-8",
    )
    queries = load_queries(path)
    assert list(queries) == ["query-p1-1-kis", "query-p1-2-kis"]
    assert queries["query-p1-2-kis"] == "tiger cubs"


@pytest.mark.unit
def test_hits_to_submission_rows_pads_to_limit() -> None:
    hits = [
        SearchHit(video_id="L21_V015", score=1.0, source="mixed", frame_index=100),
        SearchHit(video_id="L21_V015", score=0.9, source="mixed", frame_index=100),
        SearchHit(video_id="L21_V018", score=0.8, source="mixed", frame_index=200),
    ]
    rows = hits_to_submission_rows(hits, limit=100)
    assert len(rows) == 100
    assert rows[0] == ("L21_V015", 100)
    assert rows[1] == ("L21_V018", 200)
    assert len(set(rows)) == 100


@pytest.mark.unit
def test_write_kis_csv(tmp_path: Path) -> None:
    path = tmp_path / "query-p1-1-kis.csv"
    write_kis_csv(path, [("L00_V000", 1234), ("L00_V055", 5555)])
    assert path.read_text(encoding="utf-8") == "L00_V000,1234\nL00_V055,5555\n"


@pytest.mark.unit
def test_package_kis_zip_is_utf8_without_macos_junk(tmp_path: Path) -> None:
    csv_dir = tmp_path / "csvs"
    csv_dir.mkdir()
    write_kis_csv(csv_dir / "query-p1-1-kis.csv", [("L00_V000", 1)])
    write_kis_csv(csv_dir / "query-p1-2-kis.csv", [("L00_V001", 2)])
    (csv_dir / "._query-p1-1-kis.csv").write_bytes(b"\x00\x05\x16\x07not-utf8")
    zip_path = package_kis_zip(csv_dir, tmp_path / "submission.zip")
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        assert names == [
            "submission/query-p1-1-kis.csv",
            "submission/query-p1-2-kis.csv",
        ]
        for name in names:
            archive.read(name).decode("utf-8")
