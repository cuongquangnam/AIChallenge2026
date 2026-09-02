#!/usr/bin/env bash
# Optional: upload local uncommitted changes instead of git clone (slow for large repos).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION="${COLAB_SESSION:-video-retrieval}"
CLI="${COLAB_CLI:-colab}"
REMOTE_ROOT="/content/video-retrieval"
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"

echo "Checking session $SESSION..."
if ! colab_session_active "$CLI" "$SESSION"; then
  echo "Session is not reachable. Start it first:"
  echo "  ./scripts/colab/laptop_start_session.sh"
  exit 1
fi

echo "Uploading $ROOT -> $REMOTE_ROOT (prefer laptop_clone.sh for git-based setup)"
"$CLI" upload -s "$SESSION" "$ROOT" "$REMOTE_ROOT"
echo "Upload done."
