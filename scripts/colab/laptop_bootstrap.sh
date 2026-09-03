#!/usr/bin/env bash
# pip install + Drive pull + Elasticsearch load on Colab (after laptop_upload_env.sh).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION="${COLAB_SESSION:-video-retrieval}"
CLI="${COLAB_CLI:-colab}"
TIMEOUT="${COLAB_TIMEOUT_SEC:-3600}"
DOTENV="$ROOT/.env"
SCRIPTS="$ROOT/scripts/colab"
MOUNT="/content/drive"
# shellcheck source=lib.sh
source "$SCRIPTS/lib.sh"

if [[ -f "$DOTENV" ]]; then
  from_env="$(dotenv_get "$DOTENV" DRIVE_MOUNT || true)"
  [[ -n "$from_env" ]] && MOUNT="$from_env"
fi

echo "Checking session $SESSION..."
if ! colab_session_active "$CLI" "$SESSION"; then
  echo "Session is not reachable. Start it first:"
  echo "  ./scripts/colab/laptop_start_session.sh"
  exit 1
fi

# drive.mount() cannot auth from colab exec/console — mount via CLI first.
colab_ensure_drive_mounted "$CLI" "$SESSION" "$MOUNT"

echo "Bootstrapping Colab session $SESSION (timeout=${TIMEOUT}s)..."
"$CLI" exec -s "$SESSION" -f "$SCRIPTS/bootstrap_remote.py" --timeout "$TIMEOUT"
echo "Bootstrap done."
echo "Next: ./scripts/colab/laptop_start_worker.sh"
echo "Then: video-index serve  or  video-index colab search \"query\""
