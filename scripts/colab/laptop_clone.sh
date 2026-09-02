#!/usr/bin/env bash
# Clone/update the repo on Colab via git (run on laptop after session is up).
# Set COLAB_REPO_URL in Colab Secrets — the URL is NOT passed from this shell.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION="${COLAB_SESSION:-video-retrieval}"
CLI="${COLAB_CLI:-colab}"

echo "Checking session $SESSION..."
if ! "$CLI" status -s "$SESSION" 2>/dev/null | grep -qi "running\|active"; then
  echo "Session does not look active. Start it first:"
  echo "  ./scripts/colab/laptop_start_session.sh"
  exit 1
fi

echo "Cloning/updating repo on Colab (reads COLAB_REPO_URL from Colab Secrets)..."
echo "Add COLAB_REPO_URL=https://github.com/you/AIChallenge2026_2.git in Colab Secrets first."
"$CLI" exec -s "$SESSION" -f "$ROOT/scripts/colab/clone_remote.py"
echo "Clone done. Next: ./scripts/colab/laptop_set_remote_env.sh"
