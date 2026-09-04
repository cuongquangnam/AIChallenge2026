#!/usr/bin/env bash
# Start Colab session, upload .env + setup notebook, print browser URL.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION="${COLAB_SESSION:-video-retrieval}"
CLI="${COLAB_CLI:-colab}"
SCRIPTS="$ROOT/scripts/colab"
NOTEBOOK="$SCRIPTS/colab_setup.ipynb"
# shellcheck source=lib.sh
source "$SCRIPTS/lib.sh"

if [[ ! -f "$NOTEBOOK" ]]; then
  echo "Missing notebook: $NOTEBOOK" >&2
  exit 1
fi

echo "Checking session $SESSION..."
if ! colab_session_active "$CLI" "$SESSION"; then
  echo "Session not running. Starting..."
  "$SCRIPTS/laptop_start_session.sh"
fi

echo "Uploading setup notebook..."
colab_write_remote_file "$CLI" "$SESSION" "/content/colab_setup.ipynb" "$NOTEBOOK"

echo ""
echo "Open the Colab UI and run /content/colab_setup.ipynb:"
"$CLI" url -s "$SESSION" --open || "$CLI" url -s "$SESSION" || true
echo ""
echo "Notebook path on VM: /content/colab_setup.ipynb"
echo "1. Add Colab Secrets: GEMINI_API_KEY, COLAB_REPO_URL (+ optional COLAB_REPO_BRANCH, GITHUB_TOKEN)"
echo "2. Runtime → Run all"
echo "3. Leave keepalive cell running"
echo ""
echo "After worker is healthy, on laptop:"
echo "  video-index serve"
