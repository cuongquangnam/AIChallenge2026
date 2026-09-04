from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from video_retrieval.config import Settings
from video_retrieval.remote.colab import ColabRunner
from video_retrieval.storage.data_sync import create_data_sync, validate_remote_storage
from video_retrieval.storage.sync_paths import SEARCH_PULL_PATHS

logger = logging.getLogger(__name__)


class RemoteComputeGateway:
    """Laptop-side facade: Colab CLI for compute, Google Drive for data + keyframe sync."""

    def __init__(self, settings: Settings):
        if not settings.uses_remote_compute:
            raise ValueError("REMOTE_COMPUTE must be 'colab' to use RemoteComputeGateway")
        validate_remote_storage(settings)
        self.settings = settings
        self.runner = ColabRunner(settings)
        self.sync = create_data_sync(
            settings,
            local_dir=settings.data_dir,
            mount_drive=False,
        )

    def ensure_session(self) -> None:
        self.runner.ensure_session()

    def search(
        self,
        query: str,
        *,
        mode: str = "mixed",
        limit: int = 10,
        vector_name: str = "siglip",
        sync_keyframes: bool = True,
    ) -> dict[str, Any]:
        result = self.runner.search(
            query,
            mode=mode,
            limit=limit,
            vector_name=vector_name,
        )
        if sync_keyframes:
            self.sync_keyframes_for_payload(result)
        return result

    def kis(
        self,
        query: str,
        *,
        limit: int = 100,
        top_chains: int | None = None,
        sync_keyframes: bool = True,
    ) -> dict[str, Any]:
        result = self.runner.kis(query, limit=limit, top_chains=top_chains)
        if sync_keyframes:
            self.sync_keyframes_for_payload(result)
        return result

    def qa(
        self,
        question: str,
        *,
        limit: int = 24,
        frame_radius: int | None = None,
        sync_keyframes: bool = True,
    ) -> dict[str, Any]:
        result = self.runner.qa(
            question,
            limit=limit,
            frame_radius=frame_radius,
        )
        if sync_keyframes:
            self.sync_keyframes_for_payload(result)
        return result

    def trake(
        self,
        query: str,
        *,
        top_chains: int | None = None,
        sync_keyframes: bool = True,
    ) -> dict[str, Any]:
        result = self.runner.trake(query, top_chains=top_chains)
        if sync_keyframes:
            self.sync_keyframes_for_payload(result)
        return result

    def pull_data(self, *, paths: list[str] | None = None) -> int:
        """Pull indexed data from cloud storage to the laptop DATA_DIR."""
        return self.sync.pull(paths=paths or list(SEARCH_PULL_PATHS))

    def sync_keyframes_for_payload(self, payload: dict[str, Any]) -> int:
        """Download keyframes referenced by search/KIS/QA/TRAKE results."""
        hits = _collect_keyframe_hits(payload)
        return self.sync_keyframes_for_hits(hits)

    def sync_keyframes_for_hits(self, hits: list[dict[str, Any]]) -> int:
        """Download keyframe images referenced by search hits so the local UI can serve them."""
        relative_paths: list[str] = []
        seen: set[str] = set()
        for hit in hits:
            relative = _keyframe_relative_path(hit, data_dir=self.settings.data_dir)
            if relative and relative not in seen:
                seen.add(relative)
                relative_paths.append(relative)
        if not relative_paths:
            return 0
        try:
            logger.info(
                "Syncing %s keyframe(s) from Drive for UI display ...",
                len(relative_paths),
            )
            downloaded = self.sync.download_paths(relative_paths)
            logger.info("Synced %s keyframe(s) from Drive for UI display", downloaded)
            return downloaded
        except ImportError:
            logger.warning(
                "Drive sync unavailable; keyframe images may be missing locally. "
                "Set DRIVE_LOCAL_PATH to the Google Drive desktop sync folder."
            )
            return 0


def _collect_keyframe_hits(payload: dict[str, Any]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = list(payload.get("hits") or [])
    for chain in payload.get("chains") or []:
        video_id = chain.get("video_id")
        for event in chain.get("events") or []:
            hits.append(
                {
                    "video_id": video_id,
                    "keyframe_path": event.get("keyframe_path"),
                    "shot_index": event.get("shot_index"),
                    "role": event.get("role"),
                    "frame_index": event.get("frame_index"),
                    "payload": event.get("payload") or {},
                }
            )
    for item in payload.get("results") or []:
        chain = item.get("chain") or {}
        video_id = chain.get("video_id")
        for event in chain.get("events") or []:
            hits.append(
                {
                    "video_id": video_id,
                    "keyframe_path": event.get("keyframe_path"),
                    "shot_index": event.get("shot_index"),
                    "role": event.get("role"),
                    "frame_index": event.get("frame_index"),
                    "payload": event.get("payload") or {},
                }
            )
    return hits


def _keyframe_relative_path(hit: dict[str, Any], *, data_dir: Path) -> str | None:
    keyframe_path = hit.get("keyframe_path")
    video_id = hit.get("video_id")
    if keyframe_path:
        path = Path(str(keyframe_path))
        name = path.name
        if video_id and name:
            return f"keyframes/{video_id}/{name}"
        try:
            resolved = path.resolve()
            data_root = data_dir.resolve()
            if resolved.is_relative_to(data_root):
                return resolved.relative_to(data_root).as_posix()
        except (OSError, ValueError):
            pass
        if name and "keyframes" in str(keyframe_path):
            parts = Path(str(keyframe_path)).parts
            try:
                idx = parts.index("keyframes")
                return "/".join(parts[idx:])
            except ValueError:
                pass
    if video_id:
        payload = hit.get("payload") or {}
        role = hit.get("role") or payload.get("role")
        shot_index = hit.get("shot_index")
        if shot_index is not None and role:
            return f"keyframes/{video_id}/shot_{int(shot_index):04d}_{role}.jpg"
    return None
