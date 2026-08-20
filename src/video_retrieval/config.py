import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Path("./data")
    qdrant_url: str = "http://localhost:6333"
    elasticsearch_url: str = "http://localhost:9200"

    qdrant_collection: str = "video_keyframes"
    qdrant_timeout: int = 300  # seconds; default Qdrant client REST timeout is 5s
    es_index: str = "video_text"

    siglip_dim: int = 768

    visual_backend: str = "mock"  # mock | real
    ocr_backend: str = "mock"  # mock | gemini | rapidocr
    asr_backend: str = "mock"  # mock | whisper
    shot_backend: str = "opencv"  # opencv | transnetv2
    ocr_workers: int = 4

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    gemini_rpm: int = 5
    gemini_max_retries: int = 8
    gemini_batch_size: int = 10
    query_planner: str = "auto"  # auto | gemini | heuristic | ollama
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    whisper_model: str = "base"
    siglip_model_id: str = "google/siglip-base-patch16-224"
    # After the first Hugging Face download, set to true to skip Hub metadata checks.
    transformers_offline: bool = False
    hf_hub_offline: bool = False

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @property
    def videos_dir(self) -> Path:
        return self.data_dir / "videos"

    @property
    def keyframes_dir(self) -> Path:
        return self.data_dir / "keyframes"

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    def ensure_dirs(self) -> None:
        for path in (
            self.data_dir,
            self.videos_dir,
            self.keyframes_dir,
            self.audio_dir,
            self.manifests_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def manifests_dir(self) -> Path:
        return self.data_dir / "manifests"

    def with_data_dir(self, data_dir: Path | str) -> "Settings":
        path = Path(data_dir).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return self.model_copy(update={"data_dir": path.resolve()})


def _apply_hf_offline_env(settings: Settings) -> None:
    """Export HF offline flags so transformers/huggingface_hub read os.environ."""
    if settings.transformers_offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if settings.hf_hub_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"


def get_settings(*, data_dir: Path | str | None = None) -> Settings:
    settings = Settings()
    if data_dir is not None:
        settings = settings.with_data_dir(data_dir)
    _apply_hf_offline_env(settings)
    return settings
