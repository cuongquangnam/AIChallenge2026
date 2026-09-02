from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RemoteJobRequest(BaseModel):
    """Payload executed on a Colab VM via google-colab-cli (search tasks only)."""

    job: Literal["session_pull", "search", "kis", "qa", "trake"]
    drive_mount: str = "/content/drive"
    drive_data_path: str = "MyDrive/video-retrieval"
    drive_local_path: str = ""
    remote_data_dir: str = "/content/data"
    pull_paths: list[str] = Field(default_factory=list)
    settings_overrides: dict[str, Any] = Field(default_factory=dict)

    query: str | None = None
    mode: Literal["visual", "asr", "ocr", "mixed"] = "mixed"
    limit: int = 10
    vector_name: Literal["siglip", "beit3"] = "siglip"
    top_chains: int | None = None
    frame_radius: int | None = None


class RemoteJobResponse(BaseModel):
    ok: bool
    result: dict[str, Any] | None = None
    error: str | None = None
