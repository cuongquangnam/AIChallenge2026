from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
from typing import Iterable

from video_retrieval.models import SearchHit, SearchResponse
from video_retrieval.search.service import SearchService


def load_queries(path: Path) -> dict[str, str]:
    """Load ``{query_id: query_text}`` from a JSON object file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Query file must be a JSON object: {path}")
    queries: dict[str, str] = {}
    for key, value in payload.items():
        query_id = str(key).strip()
        text = str(value).strip()
        if not query_id or not text:
            raise ValueError(f"Empty query id or text in {path}: {key!r}")
        queries[query_id] = text
    if not queries:
        raise ValueError(f"No queries found in {path}")
    return queries


def hits_to_submission_rows(
    hits: Iterable[SearchHit],
    *,
    limit: int = 100,
) -> list[tuple[str, int]]:
    """Convert hits to unique ``(video_id, frame_index)`` rows (no padding)."""
    if limit < 1:
        raise ValueError("limit must be >= 1")

    rows: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for hit in hits:
        if hit.frame_index is None:
            continue
        video_id = str(hit.video_id).strip()
        frame_idx = int(hit.frame_index)
        if not video_id or frame_idx < 0:
            continue
        key = (video_id, frame_idx)
        if key in seen:
            continue
        seen.add(key)
        rows.append(key)
        if len(rows) >= limit:
            return rows

    if not rows:
        raise ValueError("Need at least one hit to build submission rows")
    return rows


def write_kis_csv(path: Path, rows: Iterable[tuple[str, int]]) -> None:
    """Write submission CSV: ``video_id,frame_idx`` with no header (UTF-8, LF)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for video_id, frame_idx in rows:
            writer.writerow([video_id, frame_idx])


def package_kis_zip(
    csv_dir: Path,
    zip_path: Path,
    *,
    arc_dir: str = "submission",
) -> Path:
    """Zip only ``*.csv`` files (no macOS ``__MACOSX`` / ``._*`` junk)."""
    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {csv_dir}")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for csv_path in csv_files:
            # Skip AppleDouble / resource-fork names if present on disk.
            if csv_path.name.startswith("._"):
                continue
            data = csv_path.read_bytes()
            data.decode("utf-8")  # fail fast if not UTF-8
            archive.writestr(f"{arc_dir}/{csv_path.name}", data)
    return zip_path


def search_to_rows(
    service: SearchService,
    query: str,
    *,
    mode: str = "mixed",
    limit: int = 100,
) -> tuple[SearchResponse, list[tuple[str, int]]]:
    """Run one search and convert hits to a fixed-length submission list."""
    mode = mode.strip().lower()
    if mode == "ocr":
        response = service.search_ocr(query, limit=limit)
    elif mode == "asr":
        response = service.search_asr(query, limit=limit)
    elif mode == "visual":
        response = service.search_visual(query, limit=limit)
    elif mode == "mixed":
        response = service.search_mixed(query, limit=limit)
    else:
        raise ValueError("mode must be one of: visual, asr, ocr, mixed")
    return response, hits_to_submission_rows(response.hits, limit=limit)


def run_kis_batch(
    service: SearchService,
    queries: dict[str, str],
    out_dir: Path,
    *,
    mode: str = "mixed",
    limit: int = 100,
    progress: bool = True,
) -> dict[str, Path]:
    """Search each query and write ``{query_id}.csv`` under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for query_id, query in queries.items():
        if progress:
            print(f"=== {query_id} ===", flush=True)
        response, rows = search_to_rows(service, query, mode=mode, limit=limit)
        if progress:
            plan = response.plan
            if plan is not None:
                print(
                    f"plan visual={plan.visual[:100]!r} "
                    f"ocr={plan.ocr[:60]!r} asr={plan.asr[:60]!r} "
                    f"weights={plan.weights}",
                    flush=True,
                )
            if response.hits:
                top = response.hits[0]
                print(
                    f"top {top.video_id},{top.frame_index} "
                    f"score={top.score:.4f} hits={len(response.hits)}",
                    flush=True,
                )
            else:
                print("top (none)", flush=True)
        out_path = out_dir / f"{query_id}.csv"
        write_kis_csv(out_path, rows)
        written[query_id] = out_path
        if progress:
            print(f"wrote {out_path} ({len(rows)} rows)", flush=True)
    return written
