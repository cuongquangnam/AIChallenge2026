from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from video_retrieval.config import Settings
from video_retrieval.models import KeyFrame, VisualEmbedding


class VisualEncoder:
    """SigLIP + BEiT3 visual encoders with a deterministic mock backend."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.backend = settings.visual_backend
        self._siglip = None
        self._siglip_processor = None
        self._beit = None
        self._beit_processor = None
        if self.backend == "real":
            self._load_real_models()

    def _load_real_models(self) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Install ML extras: pip install '.[ml]' before VISUAL_BACKEND=real"
            ) from exc

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._siglip_processor = AutoProcessor.from_pretrained(self.settings.siglip_model_id)
        self._siglip = AutoModel.from_pretrained(self.settings.siglip_model_id).to(self._device)
        self._siglip.eval()

        # BEiT is used as a practical stand-in if a BEiT-3 checkpoint is unavailable.
        self._beit_processor = AutoImageProcessor.from_pretrained(self.settings.beit3_model_id)
        self._beit = AutoModel.from_pretrained(self.settings.beit3_model_id).to(self._device)
        self._beit.eval()

    def encode_image(self, image_path: Path) -> tuple[list[float], list[float]]:
        if self.backend == "real":
            return self._encode_real(image_path)
        return self._encode_mock(image_path)

    def encode_text(self, text: str) -> list[float]:
        """Text → SigLIP space for cross-modal query."""
        if self.backend == "real":
            return self._encode_text_real(text)
        return self._hash_embed(text, self.settings.siglip_dim)

    def encode_keyframes(self, keyframes: list[KeyFrame]) -> list[VisualEmbedding]:
        out: list[VisualEmbedding] = []
        for kf in keyframes:
            siglip, beit3 = self.encode_image(kf.path)
            out.append(VisualEmbedding(keyframe=kf, siglip=siglip, beit3=beit3))
        return out

    def _encode_mock(self, image_path: Path) -> tuple[list[float], list[float]]:
        seed = f"{image_path.resolve()}:{image_path.stat().st_size}"
        return (
            self._hash_embed(seed + ":siglip", self.settings.siglip_dim),
            self._hash_embed(seed + ":beit3", self.settings.beit3_dim),
        )

    def _encode_real(self, image_path: Path) -> tuple[list[float], list[float]]:
        image = Image.open(image_path).convert("RGB")
        torch = self._torch

        with torch.no_grad():
            siglip_inputs = self._siglip_processor(images=image, return_tensors="pt").to(
                self._device
            )
            siglip_out = self._siglip.get_image_features(**siglip_inputs)
            siglip_vec = torch.nn.functional.normalize(siglip_out, dim=-1)[0].cpu().tolist()

            beit_inputs = self._beit_processor(images=image, return_tensors="pt").to(self._device)
            beit_out = self._beit(**beit_inputs).last_hidden_state[:, 0]
            beit_vec = torch.nn.functional.normalize(beit_out, dim=-1)[0].cpu().tolist()

        return siglip_vec, beit_vec

    def _encode_text_real(self, text: str) -> list[float]:
        torch = self._torch
        with torch.no_grad():
            inputs = self._siglip_processor(
                text=[text],
                padding="max_length",
                return_tensors="pt",
            ).to(self._device)
            feats = self._siglip.get_text_features(**inputs)
            return torch.nn.functional.normalize(feats, dim=-1)[0].cpu().tolist()

    @staticmethod
    def _hash_embed(seed: str, dim: int) -> list[float]:
        rng = np.random.default_rng(abs(hash(seed)) % (2**32))
        vec = rng.normal(size=dim).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-8
        return vec.tolist()
