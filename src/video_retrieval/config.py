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
    ocr_backend: str = "mock"  # mock | gemini | qwen_vl
    asr_backend: str = "mock"  # mock | whisper
    shot_backend: str = "opencv"  # opencv | transnetv2

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    gemini_rpm: int = 5
    gemini_max_retries: int = 8
    gemini_batch_size: int = 10
    # Shared multimodal LLM: auto | gemini | qwen_vl | none
    llm_backend: str = "auto"
    qwen_vl_model_id: str = "Qwen/Qwen3-VL-32B-Instruct"
    qwen_vl_dtype: str = "bf16"  # bf16 | fp16 | 4bit
    qwen_vl_device: str = "auto"
    qwen_vl_max_new_tokens: int = 2048
    query_planner: str = "auto"  # auto | gemini | qwen_vl | heuristic
    whisper_model: str = "base"
    siglip_model_id: str = "google/siglip-base-patch16-224"
    beit3_model_id: str = "microsoft/beit-base-patch16-224"

    # Q&A multimodal backend: auto (follow llm_backend) | gemini | qwen_vl | none
    qa_llm_backend: str = "auto"
    qa_retrieval_limit: int = 50
    qa_group_count: int = 6
    qa_frame_radius: int = 2
    qa_frame_stride: int = 1
    qa_min_center_gap: int = 10

    kis_max_events: int = 5
    kis_top_chains: int = 24

    trake_event_limit: int = 100
    trake_top_videos: int = 15
    trake_top_chains: int = 10
    trake_candidates_per_event: int = 50

    chain_rerank_enabled: bool = True
    chain_rerank_backend: str = "auto"  # auto | mock | real
    chain_rerank_model_id: str = "Salesforce/blip-itm-base-coco"
    chain_rerank_blend: float = 0.6
    model_pool_size: int = 3
    chain_gap_weight: float = 0.5
    chain_gap_hard_factor: float = 3.0
    video_fps: float = 25.0

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Google Drive cloud storage for indexed data
    drive_mount: str = "/content/drive"
    drive_data_path: str = "MyDrive/video-retrieval"
    # Laptop: Google Drive desktop sync folder (e.g. ~/Library/CloudStorage/.../video-retrieval)
    drive_local_path: str = ""

    # Remote compute via google-colab-cli (laptop hosts UI, Colab runs search/index GPU work)
    remote_compute: str = ""  # "" | "colab"
    colab_session: str = "video-retrieval"
    colab_cli: str = "colab"
    colab_gpu: str = "T4"
    colab_high_mem: bool = False
    colab_timeout_sec: float = 600.0
    colab_remote_data_dir: str = "/content/data"
    colab_elasticsearch_install_dir: str = "/content/elasticsearch"
    colab_qdrant_install_dir: str = "/content/qdrant"
    colab_runtime: bool = False
    # When false (default), you start/bootstrap Colab manually (see scripts/colab/MANUAL_SETUP.md).
    colab_auto_manage: bool = False
    colab_env_file: str = "/content/video-retrieval/.env.colab"
    # How laptop jobs talk to Colab:
    #   auto        — prefer persistent worker HTTP; fall back to oneshot exec
    #   persistent  — require long-running worker on the VM
    #   oneshot     — legacy: load models inside each colab exec
    #   tunnel      — laptop POSTs directly to COLAB_WORKER_PUBLIC_URL (Cloudflare)
    colab_worker_mode: str = "auto"
    colab_worker_port: int = 8765
    colab_worker_ready_timeout_sec: float = 900.0
    # How often the laptop polls GET /jobs/{id} when using COLAB_WORKER_PUBLIC_URL.
    colab_job_poll_interval_sec: float = 2.0
    # Public Cloudflare (or other) tunnel URL to the Colab worker, e.g. https://xxx.trycloudflare.com
    # When set, laptop search/KIS/QA skip Colab CLI and call this URL directly (async submit+poll).
    colab_worker_public_url: str = ""

    @property
    def uses_remote_compute(self) -> bool:
        return self.remote_compute.strip().lower() == "colab"

    @property
    def uses_public_worker(self) -> bool:
        return bool(self.colab_worker_public_url.strip())

    @property
    def colab_worker_url(self) -> str:
        public = self.colab_worker_public_url.strip().rstrip("/")
        if public:
            return public
        return f"http://127.0.0.1:{int(self.colab_worker_port)}"

    @property
    def qdrant_dir(self) -> Path:
        return self.data_dir / "qdrant"

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

    def remote_settings_overrides(self) -> dict[str, object]:
        """Settings applied on the Colab VM (Elasticsearch + resolved Qdrant URL)."""
        return {
            "elasticsearch_url": self.elasticsearch_url,
            "visual_backend": self.visual_backend,
            "ocr_backend": self.ocr_backend,
            "asr_backend": self.asr_backend,
            "chain_rerank_enabled": self.chain_rerank_enabled,
            "chain_rerank_backend": self.chain_rerank_backend,
            "query_planner": self.query_planner,
            "llm_backend": self.llm_backend,
            "qa_llm_backend": self.qa_llm_backend,
            "gemini_model": self.gemini_model,
            "gemini_rpm": self.gemini_rpm,
            "gemini_max_retries": self.gemini_max_retries,
            "gemini_batch_size": self.gemini_batch_size,
            "qwen_vl_model_id": self.qwen_vl_model_id,
            "qwen_vl_dtype": self.qwen_vl_dtype,
            "qwen_vl_device": self.qwen_vl_device,
            "qwen_vl_max_new_tokens": self.qwen_vl_max_new_tokens,
            "whisper_model": self.whisper_model,
            "siglip_model_id": self.siglip_model_id,
            "beit3_model_id": self.beit3_model_id,
            "siglip_dim": self.siglip_dim,
            "beit3_dim": self.beit3_dim,
            "qdrant_collection": self.qdrant_collection,
            "es_index": self.es_index,
            "drive_mount": self.drive_mount,
            "drive_data_path": self.drive_data_path,
            "drive_local_path": self.drive_local_path,
            "colab_elasticsearch_install_dir": self.colab_elasticsearch_install_dir,
            "colab_qdrant_install_dir": self.colab_qdrant_install_dir,
        }


def get_settings(
    *,
    data_dir: Path | str | None = None,
    colab: bool = False,
) -> Settings:
    env_file: Path | str | None = None
    if colab:
        for candidate in (
            Path("/content/video-retrieval/.env.colab"),
            Path("/content/.env.colab"),
        ):
            if candidate.is_file():
                env_file = candidate
                break

    settings = Settings(_env_file=env_file) if env_file else Settings()
    if colab:
        settings = settings.model_copy(update={"colab_runtime": True})
        settings = _resolve_colab_qdrant_settings(settings, data_dir)
    if data_dir is not None:
        return settings.with_data_dir(data_dir)
    return settings


def _resolve_colab_qdrant_settings(
    settings: Settings,
    data_dir: Path | str | None,
) -> Settings:
    """Use Qdrant HTTP server on Colab when Drive data includes ``.snapshot`` files."""
    if data_dir is not None:
        settings = settings.with_data_dir(data_dir)
    try:
        from video_retrieval.storage.qdrant_bootstrap import resolve_colab_qdrant_url

        qdrant_url = resolve_colab_qdrant_url(settings)
    except Exception:
        return settings
    if qdrant_url == settings.qdrant_url:
        return settings
    return settings.model_copy(update={"qdrant_url": qdrant_url})
