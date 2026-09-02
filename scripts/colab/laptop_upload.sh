#!/usr/bin/env bash
# Optional: upload local uncommitted changes instead of git clone (slow for large repos).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION="${COLAB_SESSION:-video-retrieval}"
CLI="${COLAB_CLI:-colab}"
REMOTE_ROOT="/content/video-retrieval"

echo "Checking session $SESSION..."
if ! "$CLI" status -s "$SESSION" 2>/dev/null | grep -qi "running\|active"; then
  echo "Session does not look active. Start it first:"
  echo "  ./scripts/colab/laptop_start_session.sh"
  exit 1
fi

echo "Uploading $ROOT -> $REMOTE_ROOT (prefer laptop_clone.sh for git-based setup)"
"$CLI" upload -s "$SESSION" "$ROOT" "$REMOTE_ROOT"
echo "Upload done."
