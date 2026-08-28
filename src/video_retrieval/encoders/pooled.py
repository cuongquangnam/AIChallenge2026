from __future__ import annotations

from pathlib import Path

from video_retrieval.config import Settings
from video_retrieval.encoders.visual import VisualEncoder
from video_retrieval.model_pool import ModelPool


class PooledVisualEncoder:
    """SigLIP-only search encoder with exclusive access per pool slot."""

    supports_beit = False

    def __init__(self, pool: ModelPool[VisualEncoder], settings: Settings):
        self.settings = settings
        self._pool = pool

    def encode_text(self, text: str) -> list[float]:
        with self._pool.borrow() as encoder:
            return encoder.encode_text(text)

    def encode_image(self, image_path: Path) -> tuple[list[float], list[float]]:
        with self._pool.borrow() as encoder:
            siglip, _beit = encoder.encode_image(image_path)
            return siglip, _beit
