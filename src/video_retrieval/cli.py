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


@app.command("search")
def search_cmd(
    query: str = typer.Argument(...),
    mode: str = typer.Option("hybrid", help="text | visual | hybrid"),
    limit: int = typer.Option(10),
) -> None:
    service = SearchService(get_settings())
    if mode == "text":
        response = service.search_text(query, limit=limit)
    elif mode == "visual":
        response = service.search_visual_text(query, limit=limit)
    else:
        response = service.search_hybrid(query, limit=limit)
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
