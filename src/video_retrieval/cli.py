from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import print

from video_retrieval.config import get_settings
from video_retrieval.pipeline.indexer import VideoIndexer, normalize_stages
from video_retrieval.search.kis import load_queries, package_kis_zip, run_kis_batch
from video_retrieval.search.service import SearchService
from video_retrieval.search.trake import TrakeService

app = typer.Typer(help="Index and search videos (keyframes + OCR/ASR).")
colab_app = typer.Typer(help="Google Colab CLI remote search (search, KIS, QA, TRAKE).")
app.add_typer(colab_app, name="colab")


@app.callback()
def _cli_main() -> None:
    from video_retrieval.log_setup import configure_logging

    configure_logging()


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
    path: Path = typer.Argument(..., help="Video file or directory of videos"),
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
) -> None:
    settings = get_settings(data_dir=data_dir)
    selected = _cli_stages(only, stages)
    if selected:
        try:
            normalize_stages(selected)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    kwargs = {"stages": selected, "reuse_extract": reuse_extract}
    indexer = VideoIndexer(settings)
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


def _remote_gateway(settings):
    from video_retrieval.remote.gateway import RemoteComputeGateway

    return RemoteComputeGateway(settings)


@app.command("search")
def search_cmd(
    query: str = typer.Argument(...),
    mode: str = typer.Option("mixed", help="visual | asr | ocr | mixed"),
    limit: int = typer.Option(10),
    data_dir: Optional[Path] = _data_dir_option(),
    remote: bool = typer.Option(
        False,
        "--remote",
        help="Run search on Colab via google-colab-cli (requires cloud storage + REMOTE_COMPUTE=colab).",
    ),
) -> None:
    settings = get_settings(data_dir=data_dir)
    if remote or settings.uses_remote_compute:
        result = _remote_gateway(settings).search(query, mode=mode, limit=limit)
        print(result)
        return
    service = SearchService(settings)
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
    zip_path: Optional[Path] = typer.Option(
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


@app.command("trake")
def trake_cmd(
    query: str = typer.Argument(..., help="Full TRAKE query with E1, E2, … events"),
    top_chains: int = typer.Option(5, min=1, max=20),
    data_dir: Optional[Path] = _data_dir_option(),
    remote: bool = typer.Option(
        False,
        "--remote",
        help="Run TRAKE on Colab via google-colab-cli.",
    ),
) -> None:
    """Parse TRAKE events and print aligned video/frame chains."""
    settings = get_settings(data_dir=data_dir)
    if remote or settings.uses_remote_compute:
        result = _remote_gateway(settings).trake(query, top_chains=top_chains)
        print(result)
        return
    result = TrakeService(settings).run(query, top_chains=top_chains)
    print(result)


@app.command("serve")
def serve_cmd(
    host: Optional[str] = None,
    port: Optional[int] = None,
    data_dir: Optional[Path] = _data_dir_option(),
) -> None:
    import uvicorn

    from video_retrieval.log_setup import configure_logging, uvicorn_log_config

    configure_logging()
    settings = get_settings(data_dir=data_dir)
    from video_retrieval import api as api_module

    api_module.settings = settings
    api_module.settings.ensure_dirs()
    from video_retrieval.runtime import init_runtime

    init_runtime(settings, force=True)
    if settings.uses_remote_compute:
        storage = f"Drive:{settings.drive_data_path or settings.drive_local_path}"
        print(
            f"UI at http://{host or settings.api_host}:{port or settings.api_port} "
            f"(search via Colab, data in {storage})"
        )
    uvicorn.run(
        api_module.app,
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=False,
        log_level="info",
        log_config=uvicorn_log_config(),
    )


@colab_app.command("session")
def colab_session_cmd(data_dir: Optional[Path] = _data_dir_option()) -> None:
    """Verify Colab session is reachable (manual setup: scripts/colab/MANUAL_SETUP.md)."""
    settings = get_settings(data_dir=data_dir)
    gateway = _remote_gateway(settings)
    gateway.ensure_session()
    print(f"Colab session reachable: {settings.colab_session}")
    if settings.colab_auto_manage:
        print(f"Data at {settings.colab_remote_data_dir} (auto-managed)")
    else:
        print("Manual mode: ensure you ran scripts/colab/laptop_bootstrap.sh on Colab first.")


@colab_app.command("worker-start")
def colab_worker_start_cmd() -> None:
    """Start the persistent Colab worker (models loaded once on the VM)."""
    import subprocess

    script = Path(__file__).resolve().parents[2] / "scripts" / "colab" / "laptop_start_worker.sh"
    raise SystemExit(subprocess.call([str(script)]))


@colab_app.command("worker-stop")
def colab_worker_stop_cmd() -> None:
    """Stop the persistent Colab worker."""
    import subprocess

    script = Path(__file__).resolve().parents[2] / "scripts" / "colab" / "laptop_stop_worker.sh"
    raise SystemExit(subprocess.call([str(script)]))


@colab_app.command("worker-status")
def colab_worker_status_cmd() -> None:
    """Show persistent Colab worker health/status."""
    import subprocess

    script = Path(__file__).resolve().parents[2] / "scripts" / "colab" / "laptop_worker_status.sh"
    raise SystemExit(subprocess.call([str(script)]))


@colab_app.command("pull")
def colab_pull_cmd(
    data_dir: Optional[Path] = _data_dir_option(),
    keyframes: bool = typer.Option(
        True,
        "--keyframes/--no-keyframes",
        help="Also pull keyframes/ for local UI display.",
    ),
) -> None:
    """Pull indexed data from Google Drive to the laptop DATA_DIR."""
    settings = get_settings(data_dir=data_dir)
    from video_retrieval.storage.data_sync import create_data_sync
    from video_retrieval.storage.sync_paths import SEARCH_PULL_PATHS

    paths = list(SEARCH_PULL_PATHS)
    if keyframes:
        paths.append("keyframes")
    sync = create_data_sync(settings, mount_drive=False)
    count = sync.pull(paths=paths)
    source = settings.drive_local_path or settings.drive_data_path
    print(f"Pulled {count} file(s) from Drive:{source}")


@colab_app.command("search")
def colab_search_cmd(
    query: str = typer.Argument(...),
    mode: str = typer.Option("mixed", help="visual | asr | ocr | mixed"),
    limit: int = typer.Option(10),
    data_dir: Optional[Path] = _data_dir_option(),
) -> None:
    """Run a search on Colab and sync keyframe thumbnails to the laptop."""
    settings = get_settings(data_dir=data_dir)
    result = _remote_gateway(settings).search(query, mode=mode, limit=limit)
    print(result)


@colab_app.command("kis")
def colab_kis_cmd(
    query: str = typer.Argument(...),
    limit: int = typer.Option(100, min=1, max=100),
    data_dir: Optional[Path] = _data_dir_option(),
) -> None:
    """Run KIS event-chain search on Colab."""
    settings = get_settings(data_dir=data_dir)
    result = _remote_gateway(settings).kis(query, limit=limit)
    print(result)


@colab_app.command("qa")
def colab_qa_cmd(
    question: str = typer.Argument(...),
    limit: int = typer.Option(24, min=1, max=100),
    frame_radius: Optional[int] = typer.Option(None, min=0, max=30),
    data_dir: Optional[Path] = _data_dir_option(),
) -> None:
    """Run QA (event chains + VLM answer) on Colab."""
    settings = get_settings(data_dir=data_dir)
    result = _remote_gateway(settings).qa(
        question,
        limit=limit,
        frame_radius=frame_radius,
    )
    print(result)


@colab_app.command("trake")
def colab_trake_cmd(
    query: str = typer.Argument(...),
    top_chains: int = typer.Option(5, min=1, max=20),
    data_dir: Optional[Path] = _data_dir_option(),
) -> None:
    """Run TRAKE temporal chain search on Colab."""
    settings = get_settings(data_dir=data_dir)
    result = _remote_gateway(settings).trake(query, top_chains=top_chains)
    print(result)


if __name__ == "__main__":
    app()
