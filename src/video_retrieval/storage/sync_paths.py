"""DATA_DIR subfolders pulled from cloud storage for remote search tasks."""

SEARCH_PULL_PATHS = ("elasticsearch", "qdrant")
CHAIN_TASK_PULL_PATHS = ("elasticsearch", "qdrant", "keyframes")
# Full layout for push / manual sync (includes source videos).
QA_PULL_PATHS = ("elasticsearch", "qdrant", "keyframes", "videos", "manifests")
# Pulled once at Colab session start. Includes keyframes zip extract by default;
# pass --skip-keyframes / PULL_KEYFRAMES=False for a faster start.
SESSION_PULL_PATHS = ("elasticsearch", "qdrant", "manifests", "keyframes")
