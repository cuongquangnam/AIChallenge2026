#!/usr/bin/env bash
# Export local Qdrant + Elasticsearch into backups/aic-search-backup.zip for Google Drive.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
ELASTICSEARCH_URL="${ELASTICSEARCH_URL:-http://localhost:9200}"
QDRANT_COLLECTION="${QDRANT_COLLECTION:-video_keyframes}"
ES_INDEX="${ES_INDEX:-video_text}"

BACKUP_DIR="$ROOT/backups"
QDRANT_DIR="$BACKUP_DIR/qdrant"
ES_DIR="$BACKUP_DIR/elasticsearch"
ZIP_PATH="$BACKUP_DIR/aic-search-backup.zip"

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

echo "Starting Qdrant + Elasticsearch (if needed)..."
docker compose up -d qdrant elasticsearch
wait_http "$QDRANT_URL/collections" "Qdrant"
wait_http "$ELASTICSEARCH_URL" "Elasticsearch"

rm -rf "$QDRANT_DIR" "$ES_DIR"
mkdir -p "$QDRANT_DIR" "$ES_DIR"

echo "Creating Qdrant snapshot for $QDRANT_COLLECTION..."
SNAP_JSON="$(curl -sf -X POST "$QDRANT_URL/collections/${QDRANT_COLLECTION}/snapshots")"
SNAP_NAME="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["result"]["name"])' "$SNAP_JSON")"
curl -sf -o "$QDRANT_DIR/$SNAP_NAME" \
  "$QDRANT_URL/collections/${QDRANT_COLLECTION}/snapshots/${SNAP_NAME}"
echo "  saved $QDRANT_DIR/$SNAP_NAME"

echo "Exporting Elasticsearch index $ES_INDEX..."
python3 "$ROOT/scripts/es_backup.py" dump \
  --es-url "$ELASTICSEARCH_URL" \
  --index "$ES_INDEX" \
  --dir "$ES_DIR"

{
  echo "aic-search-backup"
  echo "created=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "qdrant_url=$QDRANT_URL"
  echo "qdrant_collection=$QDRANT_COLLECTION"
  echo "qdrant_snapshot=$SNAP_NAME"
  echo "elasticsearch_url=$ELASTICSEARCH_URL"
  echo "es_index=$ES_INDEX"
} > "$BACKUP_DIR/MANIFEST.txt"

rm -f "$ZIP_PATH"
(
  cd "$BACKUP_DIR"
  zip -q -r aic-search-backup.zip qdrant elasticsearch MANIFEST.txt
)

echo
echo "Backup ready: $ZIP_PATH"
echo "Upload that zip to Google Drive (do not include .env)."
if command -v open >/dev/null; then
  open "$BACKUP_DIR"
fi
