#!/usr/bin/env bash
# Show persistent Colab worker status.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION="${COLAB_SESSION:-video-retrieval}"
CLI="${COLAB_CLI:-colab}"
SCRIPTS="$ROOT/scripts/colab"
# shellcheck source=lib.sh
source "$SCRIPTS/lib.sh"

if ! colab_session_active "$CLI" "$SESSION"; then
  echo "Session is not reachable."
  exit 1
fi

"$CLI" exec -s "$SESSION" -f "$SCRIPTS/worker_status_remote.py" --timeout 60
