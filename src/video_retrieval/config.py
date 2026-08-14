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
    es_index: str = "video_text"

    siglip_dim: int = 768
    beit3_dim: int = 768

    visual_backend: str = "mock"  # mock | real
    ocr_backend: str = "mock"  # mock | gemini
    asr_backend: str = "mock"  # mock | whisper
    shot_backend: str = "opencv"  # opencv | transnetv2

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    gemini_rpm: int = 5
    gemini_max_retries: int = 5
    gemini_batch_size: int = 10
    whisper_model: str = "base"
    siglip_model_id: str = "google/siglip-base-patch16-224"
    beit3_model_id: str = "microsoft/beit-base-patch16-224"

    # Multimodal Q&A. The openai_compatible backend expects a
    # /chat/completions endpoint that accepts image_url content parts.
    qa_llm_backend: str = "none"  # none | openai_compatible
    qa_llm_api_key: str = ""
    qa_llm_base_url: str = "https://api.openai.com/v1"
    qa_llm_model: str = ""
    qa_llm_timeout_sec: float = 120.0
    qa_retrieval_limit: int = 50
    qa_group_count: int = 10
    qa_frame_radius: int = 5
    qa_frame_stride: int = 1
    qa_min_center_gap: int = 10

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
        for path in (self.data_dir, self.videos_dir, self.keyframes_dir, self.audio_dir):
            path.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    return Settings()
