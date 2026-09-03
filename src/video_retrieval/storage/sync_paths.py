"""DATA_DIR subfolders pulled from cloud storage for remote search tasks."""

SEARCH_PULL_PATHS = ("elasticsearch", "qdrant")
CHAIN_TASK_PULL_PATHS = ("elasticsearch", "qdrant", "keyframes")
# Full layout for push / manual sync (includes source videos).
QA_PULL_PATHS = ("elasticsearch", "qdrant", "keyframes", "videos", "manifests")
# Pulled once at Colab session start. Keyframes/videos stay on Drive by default;
# result thumbnails and QA source videos are pulled lazily when needed.
SESSION_PULL_PATHS = ("elasticsearch", "qdrant", "manifests")
