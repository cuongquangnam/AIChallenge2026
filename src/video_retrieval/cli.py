from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import print

from video_retrieval.config import get_settings
from video_retrieval.pipeline.indexer import VideoIndexer
from video_retrieval.search.service import SearchService

app = typer.Typer(help="Index and search videos (keyframes + OCR/ASR).")


@app.command("index")
def index_cmd(
    path: Path = typer.Argument(..., help="Video file or directory of videos"),
    video_id: Optional[str] = typer.Option(None, help="Override video id for a single file"),
) -> None:
    settings = get_settings()
    indexer = VideoIndexer(settings)
    if path.is_dir():
        results = indexer.index_directory(path)
        for result in results:
            print(result)
    else:
        result = indexer.index_video(path, video_id=video_id)
        print(result)


@app.command("index-keyframes")
def index_keyframes_cmd(
    path: Path = typer.Argument(..., help="Directory containing video_id/*.jpg keyframes"),
    limit: Optional[int] = typer.Option(None, help="Optional max number of images to index"),
) -> None:
    settings = get_settings()
    indexer = VideoIndexer(settings)
    results = indexer.index_keyframe_directory(path, limit=limit)
    for result in results:
        print(result)


@app.command("search")
def search_cmd(
    query: str = typer.Argument(...),
    mode: str = typer.Option("hybrid", help="text | visual | hybrid"),
    limit: int = typer.Option(10),
    source: Optional[str] = typer.Option(None, help="Filter text source: ocr | asr"),
    video_id: Optional[str] = typer.Option(None, help="Filter to one video id"),
    vector_name: str = typer.Option("siglip", help="Visual vector: siglip"),
) -> None:
    service = SearchService(get_settings())
    if mode == "text":
        response = service.search_text_filtered(
            query, limit=limit, source=source, video_id=video_id
        )
    elif mode == "visual":
        response = service.search_visual_text(
            query, limit=limit, vector_name=vector_name, video_id=video_id
        )
    else:
        response = service.search_hybrid_filtered(
            query, limit=limit, source=source, video_id=video_id
        )
    print(response)


@app.command("serve")
def serve_cmd(host: Optional[str] = None, port: Optional[int] = None) -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "video_retrieval.api:app",
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    app()
