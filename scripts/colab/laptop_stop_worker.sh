#!/usr/bin/env bash
# Stop the persistent Colab worker.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION="${COLAB_SESSION:-video-retrieval}"
CLI="${COLAB_CLI:-colab}"
SCRIPTS="$ROOT/scripts/colab"
# shellcheck source=lib.sh
source "$SCRIPTS/lib.sh"

echo "Checking session $SESSION..."
if ! colab_session_active "$CLI" "$SESSION"; then
  echo "Session is not reachable."
  exit 1
fi

"$CLI" upload -s "$SESSION" "$SCRIPTS/stop_worker_remote.py" /content/video-retrieval/scripts/colab/stop_worker_remote.py >/dev/null || true
"$CLI" exec -s "$SESSION" -f "$SCRIPTS/stop_worker_remote.py" --timeout 60
echo "Done."
