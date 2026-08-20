#!/usr/bin/env python3
"""Run Textual KIS queries via video-index (Gemini planner + mixed search)."""

from __future__ import annotations

from pathlib import Path

from video_retrieval.config import get_settings
from video_retrieval.search.kis import load_queries, run_kis_batch
from video_retrieval.search.service import SearchService

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERIES = ROOT / "queries" / "kis_p1.json"
OUT_DIR = ROOT / "submissions" / "kis_p1"


def main() -> None:
    settings = get_settings()
    queries = load_queries(DEFAULT_QUERIES)
    print(
        f"planner={settings.query_planner} model={settings.gemini_model} "
        f"queries={len(queries)} out={OUT_DIR}",
        flush=True,
    )
    service = SearchService(settings)
    run_kis_batch(service, queries, OUT_DIR, mode="mixed", limit=100)


if __name__ == "__main__":
    main()
