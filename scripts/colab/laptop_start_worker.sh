#!/usr/bin/env bash
# Sync latest remote/ code to Colab and start the persistent worker (models stay loaded).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION="${COLAB_SESSION:-video-retrieval}"
CLI="${COLAB_CLI:-colab}"
TIMEOUT="${COLAB_WORKER_READY_TIMEOUT_SEC:-900}"
DOTENV="$ROOT/.env"
SCRIPTS="$ROOT/scripts/colab"
# shellcheck source=lib.sh
source "$SCRIPTS/lib.sh"

echo "Checking session $SESSION..."
if ! colab_session_active "$CLI" "$SESSION"; then
  echo "Session is not reachable. Start it first:"
  echo "  ./scripts/colab/laptop_start_session.sh"
  exit 1
fi

# Sync worker code from this laptop (includes uncommitted local changes).
echo "Syncing remote package + worker scripts to Colab..."
"$CLI" upload -s "$SESSION" "$ROOT/src/video_retrieval/remote" /content/video-retrieval/src/video_retrieval/remote
"$CLI" upload -s "$SESSION" "$SCRIPTS/start_worker_remote.py" /content/video-retrieval/scripts/colab/start_worker_remote.py
"$CLI" upload -s "$SESSION" "$SCRIPTS/stop_worker_remote.py" /content/video-retrieval/scripts/colab/stop_worker_remote.py
"$CLI" upload -s "$SESSION" "$SCRIPTS/worker_status_remote.py" /content/video-retrieval/scripts/colab/worker_status_remote.py

PORT="$(dotenv_get "$DOTENV" COLAB_WORKER_PORT 2>/dev/null || true)"
PORT="${PORT:-8765}"
DATA_DIR="$(dotenv_get "$DOTENV" COLAB_REMOTE_DATA_DIR 2>/dev/null || true)"
DATA_DIR="${DATA_DIR:-/content/data}"

ENV_ARGS=(
  "COLAB_WORKER_PORT=$PORT"
  "COLAB_REMOTE_DATA_DIR=$DATA_DIR"
  "COLAB_WORKER_READY_TIMEOUT_SEC=$TIMEOUT"
)

echo "Starting persistent worker on Colab (first start loads models; may take several minutes)..."
if ! colab_exec_script "$CLI" "$SESSION" "$TIMEOUT" "$SCRIPTS/start_worker_remote.py" "${ENV_ARGS[@]}"; then
  echo "Failed to start worker. Check VM log:"
  echo "  colab exec -s $SESSION -f $SCRIPTS/worker_status_remote.py"
  echo "  or: colab console -s $SESSION   then   tail -n 100 /content/video-retrieval/worker.log"
  exit 1
fi

echo "Worker status:"
"$CLI" exec -s "$SESSION" -f "$SCRIPTS/worker_status_remote.py" --timeout 60 || true
echo ""
echo "Done. Searches will reuse the warm worker:"
echo "  video-index colab search \"yellow lion\""
