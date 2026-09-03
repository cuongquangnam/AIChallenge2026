#!/usr/bin/env bash
# Run ON the Colab VM (console / notebook !bash), not from the laptop.
#
# Prerequisites:
#   1. Repo at /content/video-retrieval
#   2. /content/video-retrieval/.env.colab present
#   3. Google Drive mounted at /content/drive  (do this in the browser notebook:)
#        from google.colab import drive
#        drive.mount("/content/drive")
#
# Usage (on the VM):
#   bash /content/video-retrieval/scripts/colab/setup_on_vm.sh
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/content/video-retrieval}"
cd "$REPO_ROOT"

if [[ ! -d "$REPO_ROOT/.git" && ! -f "$REPO_ROOT/pyproject.toml" ]]; then
  echo "Repo not found at $REPO_ROOT" >&2
  exit 1
fi
if [[ ! -f "$REPO_ROOT/.env.colab" ]]; then
  echo "Missing $REPO_ROOT/.env.colab — upload from laptop first:" >&2
  echo "  ./scripts/colab/laptop_upload_env.sh" >&2
  exit 1
fi
if [[ ! -d /content/drive/MyDrive ]]; then
  echo "Google Drive not mounted. In a Colab notebook cell run:" >&2
  echo "  from google.colab import drive" >&2
  echo "  drive.mount('/content/drive')" >&2
  echo "Then re-run this script." >&2
  exit 1
fi

export COLAB_REMOTE_DATA_DIR="${COLAB_REMOTE_DATA_DIR:-/content/data}"
export COLAB_WORKER_PORT="${COLAB_WORKER_PORT:-8765}"
export COLAB_WORKER_READY_TIMEOUT_SEC="${COLAB_WORKER_READY_TIMEOUT_SEC:-900}"

echo "==> Bootstrap (pip install + Drive pull + Elasticsearch)..."
python3 "$REPO_ROOT/scripts/colab/bootstrap_remote.py"

echo "==> Start persistent worker..."
python3 "$REPO_ROOT/scripts/colab/start_worker_remote.py"

echo "==> Status:"
python3 "$REPO_ROOT/scripts/colab/worker_status_remote.py"

echo ""
echo "VM setup done. From your laptop:"
echo "  video-index serve"
echo "  video-index colab search \"your query\""
