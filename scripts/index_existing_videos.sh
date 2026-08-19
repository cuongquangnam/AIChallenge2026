#!/usr/bin/env bash
# Index visual + OCR + ASR for videos already in data/videos.
# Reuses keyframes/audio; does not copy videos that are already there.
# caffeinate keeps the Mac awake until indexing finishes.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

DATA_DIR="${DATA_DIR:-./data}"
if [[ "$DATA_DIR" != /* ]]; then
  DATA_DIR="$ROOT/$DATA_DIR"
fi
VIDEOS_DIR="${VIDEOS_DIR:-$DATA_DIR/videos}"
STAGES="${STAGES:-visual,ocr,asr}"
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
ELASTICSEARCH_URL="${ELASTICSEARCH_URL:-http://localhost:9200}"

if [[ ! -d "$VIDEOS_DIR" ]]; then
  echo "Videos directory not found: $VIDEOS_DIR" >&2
  exit 1
fi

INDEXER="$ROOT/.venv/bin/video-index"
if [[ ! -x "$INDEXER" ]]; then
  INDEXER="$(command -v video-index || true)"
fi
if [[ -z "$INDEXER" ]]; then
  echo "video-index not found. Run: source .venv/bin/activate && pip install -e ." >&2
  exit 1
fi

if ! command -v caffeinate >/dev/null; then
  echo "caffeinate not found (macOS only)." >&2
  exit 1
fi

wait_http() {
  local url="$1"
  local label="$2"
  local n=0
  until curl -sf "$url" >/dev/null; do
    n=$((n + 1))
    if [[ "$n" -ge 60 ]]; then
      echo "Timed out waiting for $label at $url" >&2
      echo "Start infra with: docker compose up -d" >&2
      exit 1
    fi
    sleep 2
  done
}

echo "Starting Qdrant + Elasticsearch..."
docker compose up -d qdrant elasticsearch
wait_http "$QDRANT_URL/collections" "Qdrant"
wait_http "$ELASTICSEARCH_URL" "Elasticsearch"

echo "Indexing $VIDEOS_DIR (stages=$STAGES, resume completed videos, reuse keyframes/audio)"
echo "Mac will stay awake via caffeinate until this finishes."

# -d display  -i idle  -m disk  -s system (AC power)
exec caffeinate -dims -- "$INDEXER" index "$VIDEOS_DIR" \
  --stages "$STAGES" \
  --reuse-extract \
  --resume \
  --data-dir "$DATA_DIR" \
  "$@"
