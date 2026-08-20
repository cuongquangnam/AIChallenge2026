from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import print

from video_retrieval.config import get_settings
from video_retrieval.pipeline.indexer import VideoIndexer, normalize_stages
from video_retrieval.search.kis import load_queries, package_kis_zip, run_kis_batch
from video_retrieval.search.service import SearchService

app = typer.Typer(help="Index and search videos (keyframes + OCR/ASR).")


def _data_dir_option() -> Path | None:
    return typer.Option(
        None,
        "--data-dir",
        help="Output directory for videos, keyframes, audio, and manifests (overrides DATA_DIR).",
        dir_okay=True,
        file_okay=False,
        resolve_path=True,
    )


@app.command("index")
def index_cmd(
    path: Path = typer.Argument(
        ...,
        help="Video file, directory of videos, or keyframes directory (video-id folders)",
    ),
    video_id: Optional[str] = typer.Option(None, help="Override video id for a single file"),
    data_dir: Optional[Path] = _data_dir_option(),
    only: Optional[str] = typer.Option(
        None,
        "--only",
        help="Run a single encoding stage: visual | ocr | asr",
    ),
    stages: Optional[str] = typer.Option(
        None,
        "--stages",
        help="Comma-separated stages to run, e.g. visual,ocr (default: visual,ocr,asr)",
    ),
    reuse_extract: bool = typer.Option(
        True,
        "--reuse-extract/--reextract",
        help="Reuse extracted keyframes/audio when present.",
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--rerun",
        help="Skip videos whose requested stages are already in the manifest / Qdrant.",
    ),
) -> None:
    settings = get_settings(data_dir=data_dir)
    indexer = VideoIndexer(settings)
    selected = _cli_stages(only, stages)
    if selected:
        try:
            normalize_stages(selected)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    kwargs = {"stages": selected, "reuse_extract": reuse_extract, "resume": resume}
    if path.is_dir():
        results = indexer.index_directory(path, **kwargs)
        for result in results:
            print(result)
    else:
        result = indexer.index_video(path, video_id=video_id, **kwargs)
        print(result)


def _cli_stages(only: str | None, stages: str | None) -> list[str] | None:
    if only and stages:
        raise typer.BadParameter("Use --only or --stages, not both.")
    if only:
        return [only.strip().lower()]
    if stages:
        return [part.strip().lower() for part in stages.split(",") if part.strip()]
    return None


@app.command("search")
def search_cmd(
    query: str = typer.Argument(...),
    mode: str = typer.Option("mixed", help="visual | asr | ocr | mixed"),
    limit: int = typer.Option(10, min=1, max=100),
    data_dir: Optional[Path] = _data_dir_option(),
) -> None:
    service = SearchService(get_settings(data_dir=data_dir))
    if mode == "ocr":
        response = service.search_ocr(query, limit=limit)
    elif mode == "asr":
        response = service.search_asr(query, limit=limit)
    elif mode == "visual":
        response = service.search_visual(query, limit=limit)
    elif mode == "mixed":
        response = service.search_mixed(query, limit=limit)
    else:
        raise typer.BadParameter("mode must be one of: visual, asr, ocr, mixed")
    print(response)


@app.command("kis")
def kis_cmd(
    queries: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="JSON object mapping query_id -> query text (e.g. queries/kis_p1.json)",
    ),
    out_dir: Path = typer.Option(
        Path("submissions/kis_p1"),
        "--out-dir",
        help="Directory for {query_id}.csv files (video_id,frame_idx).",
        resolve_path=True,
    ),
    mode: str = typer.Option("mixed", help="visual | asr | ocr | mixed"),
    limit: int = typer.Option(100, min=1, max=100, help="Rows per CSV (KIS expects 100)."),
    data_dir: Optional[Path] = _data_dir_option(),
    quiet: bool = typer.Option(False, "--quiet", help="Less progress output."),
    zip_path: Optional[Path] = typer.Option(
        None,
        "--zip",
        help="Also write a clean submission zip (no __MACOSX / ._ files).",
        resolve_path=True,
    ),
) -> None:
    """Run Textual KIS queries and write one 100-line CSV per query."""
    if mode not in {"visual", "asr", "ocr", "mixed"}:
        raise typer.BadParameter("mode must be one of: visual, asr, ocr, mixed")
    settings = get_settings(data_dir=data_dir)
    query_map = load_queries(queries)
    print(
        f"planner={settings.query_planner} model={settings.gemini_model} "
        f"queries={len(query_map)} out={out_dir}"
    )
    service = SearchService(settings)
    written = run_kis_batch(
        service,
        query_map,
        out_dir,
        mode=mode,
        limit=limit,
        progress=not quiet,
    )
    print(f"done: wrote {len(written)} CSV files to {out_dir}")
    if zip_path is not None:
        packaged = package_kis_zip(out_dir, zip_path)
        print(f"zip: {packaged}")


@app.command("kis-zip")
def kis_zip_cmd(
    csv_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        resolve_path=True,
        help="Directory containing query-*.csv files",
    ),
    zip_path: Path = typer.Option(
        None,
        "--zip",
        help="Output zip path (default: <csv_dir>/submission.zip)",
        resolve_path=True,
    ),
) -> None:
    """Package KIS CSVs into a clean UTF-8 zip (no macOS AppleDouble junk)."""
    out = zip_path or (csv_dir / "submission.zip")
    packaged = package_kis_zip(csv_dir, out)
    print(f"zip: {packaged}")


@app.command("serve")
def serve_cmd(
    host: Optional[str] = None,
    port: Optional[int] = None,
    data_dir: Optional[Path] = _data_dir_option(),
) -> None:
    import uvicorn

    settings = get_settings(data_dir=data_dir)
    from video_retrieval import api as api_module

    api_module.settings = settings
    api_module.settings.ensure_dirs()
    uvicorn.run(
        api_module.app,
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    app()
