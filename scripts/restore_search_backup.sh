#!/usr/bin/env bash
# Restore Qdrant + Elasticsearch from aic-search-backup.zip into local Docker.
# Usage:
#   ./scripts/restore_search_backup.sh
#   ./scripts/restore_search_backup.sh /path/to/aic-search-backup.zip
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

ARCHIVE="${1:-$ROOT/backups/aic-search-backup.zip}"
WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/aic-restore.XXXXXX")"
cleanup() { rm -rf "$WORKDIR"; }
trap cleanup EXIT

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

if [[ -d "$ARCHIVE" ]]; then
  SRC="$ARCHIVE"
elif [[ -f "$ARCHIVE" ]]; then
  echo "Unpacking $ARCHIVE..."
  unzip -q "$ARCHIVE" -d "$WORKDIR"
  SRC="$WORKDIR"
else
  echo "Backup not found: $ARCHIVE" >&2
  echo "Download aic-search-backup.zip from Drive, then:" >&2
  echo "  ./scripts/restore_search_backup.sh /path/to/aic-search-backup.zip" >&2
  exit 1
fi

QDRANT_DIR="$(find "$SRC" -type d -name qdrant | head -n 1)"
ES_DIR="$(find "$SRC" -type d -name elasticsearch | head -n 1)"
SNAP_FILE="$(find "${QDRANT_DIR:-/nonexistent}" -name '*.snapshot' | head -n 1 || true)"

if [[ -z "${QDRANT_DIR:-}" || -z "$SNAP_FILE" ]]; then
  echo "No Qdrant snapshot found in $ARCHIVE" >&2
  exit 1
fi
if [[ -z "${ES_DIR:-}" || ! -f "$ES_DIR/${ES_INDEX}.ndjson" ]]; then
  echo "No Elasticsearch dump ($ES_INDEX.ndjson) found in $ARCHIVE" >&2
  exit 1
fi

echo "Starting Qdrant + Elasticsearch..."
docker compose up -d qdrant elasticsearch
wait_http "$QDRANT_URL/collections" "Qdrant"
wait_http "$ELASTICSEARCH_URL" "Elasticsearch"

echo "Restoring Qdrant collection $QDRANT_COLLECTION from $(basename "$SNAP_FILE")..."
curl -sf -X DELETE "$QDRANT_URL/collections/${QDRANT_COLLECTION}" >/dev/null || true
curl -sf -X POST \
  -H "Content-Type: multipart/form-data" \
  -F "snapshot=@${SNAP_FILE}" \
  "$QDRANT_URL/collections/${QDRANT_COLLECTION}/snapshots/upload?priority=snapshot"
echo

echo "Restoring Elasticsearch index $ES_INDEX..."
python3 "$ROOT/scripts/es_backup.py" restore \
  --es-url "$ELASTICSEARCH_URL" \
  --index "$ES_INDEX" \
  --dir "$ES_DIR"

echo
echo "Done. Point .env at:"
echo "  QDRANT_URL=$QDRANT_URL"
echo "  ELASTICSEARCH_URL=$ELASTICSEARCH_URL"
echo
curl -sf "$QDRANT_URL/collections/${QDRANT_COLLECTION}" | python3 -c \
  'import sys,json; r=json.load(sys.stdin)["result"]; print("qdrant points:", r.get("points_count"))'
curl -sf "$ELASTICSEARCH_URL/${ES_INDEX}/_count" | python3 -c \
  'import sys,json; print("es docs:", json.load(sys.stdin).get("count"))'
