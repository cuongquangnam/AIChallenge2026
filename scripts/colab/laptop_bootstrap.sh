#!/usr/bin/env bash
# pip install + Drive pull + Elasticsearch load on Colab (after laptop_upload_env.sh).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION="${COLAB_SESSION:-video-retrieval}"
CLI="${COLAB_CLI:-colab}"
TIMEOUT="${COLAB_TIMEOUT_SEC:-3600}"

echo "Bootstrapping Colab session $SESSION (timeout=${TIMEOUT}s)..."
"$CLI" exec -s "$SESSION" -f "$ROOT/scripts/colab/bootstrap_remote.py" --timeout "$TIMEOUT"
echo "Bootstrap done. From laptop: video-index serve  or  video-index colab search \"query\""
