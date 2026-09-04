#!/usr/bin/env bash
# Migrate a Qdrant ≤1.12 collection snapshot so Colab's 1.19 musl binary can recover it.
#
# Why: GitHub Linux binaries ≥1.17 omit RocksDB; old snapshots use payload_storage_type=on_disk
# and fail with "unknown variant `on_disk`". Qdrant 1.12 also panics on Colab's cgroup v2.
# Docker 1.16.3 still migrates RocksDB; then 1.19 can re-export a Colab-safe snapshot.
#
# Usage:
#   ./scripts/colab/migrate_qdrant_snapshot_for_colab.sh \
#     export_transnet/qdrant/video_keyframes_transnet-....snapshot \
#     export_transnet/qdrant/video_keyframes_transnet.snapshot
set -euo pipefail

SRC="${1:?source .snapshot path}"
DEST="${2:?destination .snapshot path}"
COLLECTION="${QDRANT_COLLECTION:-video_keyframes_transnet}"
PORT="${QDRANT_MIGRATE_PORT:-6335}"
WORK="${TMPDIR:-/tmp}/qdrant_migrate_colab_$$"

cleanup() {
  docker rm -f qdrant-migrate-colab >/dev/null 2>&1 || true
  rm -rf "$WORK"
}
trap cleanup EXIT

mkdir -p "$WORK/snapshots" "$WORK/storage" "$(dirname "$DEST")"
cp "$SRC" "$WORK/snapshots/${COLLECTION}.snapshot"

echo "[migrate] recover with Docker qdrant:v1.16.3 ..."
docker pull qdrant/qdrant:v1.16.3 >/dev/null
docker run -d --name qdrant-migrate-colab \
  -p "${PORT}:6333" \
  -v "$WORK/storage:/qdrant/storage" \
  -v "$WORK/snapshots:/qdrant/snapshots" \
  qdrant/qdrant:v1.16.3 >/dev/null

for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/readyz" >/dev/null; then
    break
  fi
  sleep 1
done

curl -sf -X PUT \
  "http://127.0.0.1:${PORT}/collections/${COLLECTION}/snapshots/recover?wait=true" \
  -H 'Content-Type: application/json' \
  -d "{\"location\":\"file:///qdrant/snapshots/${COLLECTION}.snapshot\",\"priority\":\"snapshot\"}" \
  >/dev/null

POINTS="$(curl -sf "http://127.0.0.1:${PORT}/collections/${COLLECTION}" | python3 -c "import sys,json; print(json.load(sys.stdin)['result']['points_count'])")"
echo "[migrate] 1.16.3 points=${POINTS}"

echo "[migrate] upgrade storage with Docker qdrant:v1.19.0 ..."
docker rm -f qdrant-migrate-colab >/dev/null
docker pull qdrant/qdrant:v1.19.0 >/dev/null
docker run -d --name qdrant-migrate-colab \
  -p "${PORT}:6333" \
  -v "$WORK/storage:/qdrant/storage" \
  -v "$WORK/snapshots:/qdrant/snapshots" \
  qdrant/qdrant:v1.19.0 >/dev/null

for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/readyz" >/dev/null; then
    break
  fi
  sleep 1
done

POINTS="$(curl -sf "http://127.0.0.1:${PORT}/collections/${COLLECTION}" | python3 -c "import sys,json; print(json.load(sys.stdin)['result']['points_count'])")"
echo "[migrate] 1.19.0 points=${POINTS}"

CREATE="$(curl -sf -X POST "http://127.0.0.1:${PORT}/collections/${COLLECTION}/snapshots?wait=true")"
NAME="$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['result']['name'])" "$CREATE")"
docker cp "qdrant-migrate-colab:/qdrant/snapshots/${COLLECTION}/${NAME}" "$DEST"
echo "[migrate] wrote $DEST ($(du -h "$DEST" | awk '{print $1}'))"
echo "[migrate] upload this file to Drive as MyDrive/video-retrieval/qdrant/video_keyframes_transnet.snapshot"
