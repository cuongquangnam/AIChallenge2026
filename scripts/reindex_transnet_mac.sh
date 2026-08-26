#!/usr/bin/env bash
# Mac-efficient full reindex for data/videos:
#   TransNetV2 shot detect → dense keyframes → batched SigLIP → RapidOCR → faster-whisper
#
# Usage:
#   ./scripts/reindex_transnet_mac.sh
#   STAGES=visual,ocr ./scripts/reindex_transnet_mac.sh          # skip ASR first (<1 day)
#   START_FROM=L21_V050 ./scripts/reindex_transnet_mac.sh        # resume mid-batch
#   MAX_VIDEOS=5 ./scripts/reindex_transnet_mac.sh               # smoke test
#
# Expected wall time for ~873 videos / ~131h media on Apple Silicon:
#   extract+visual+ocr ≈ 8–14h
#   + ASR (faster-whisper base/int8) ≈ +10–20h
# Tip: run STAGES=visual,ocr first, then STAGES=asr --reuse-extract overnight.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Capture caller overrides before .env can overwrite them.
_CALLER_DATA_DIR="${DATA_DIR-}"
_CALLER_VIDEOS_DIR="${VIDEOS_DIR-}"
_CALLER_STAGES="${STAGES-}"
_CALLER_QDRANT_URL="${QDRANT_URL-}"
_CALLER_ES_URL="${ELASTICSEARCH_URL-}"
_CALLER_QDRANT_COLLECTION="${QDRANT_COLLECTION-}"
_CALLER_ES_INDEX="${ES_INDEX-}"
_CALLER_START_FROM="${START_FROM-}"
_CALLER_MAX_VIDEOS="${MAX_VIDEOS-}"
_CALLER_MAX_RETRIES="${MAX_RETRIES-}"
_CALLER_LOG_FILE="${LOG_FILE-}"
_CALLER_REEXTRACT="${REEXTRACT-}"
_CALLER_VISUAL_BACKEND="${VISUAL_BACKEND-}"
_CALLER_OCR_BACKEND="${OCR_BACKEND-}"
_CALLER_ASR_BACKEND="${ASR_BACKEND-}"
_CALLER_SHOT_BACKEND="${SHOT_BACKEND-}"
_CALLER_SIGLIP_BATCH_SIZE="${SIGLIP_BATCH_SIZE-}"
_CALLER_MAX_SHOT_SEC="${MAX_SHOT_SEC-}"
_CALLER_TRANSNET_DEVICE="${TRANSNET_DEVICE-}"
_CALLER_OCR_WORKERS="${OCR_WORKERS-}"
_CALLER_WHISPER_MODEL="${WHISPER_MODEL-}"
_CALLER_FASTER_WHISPER_COMPUTE_TYPE="${FASTER_WHISPER_COMPUTE_TYPE-}"
_CALLER_TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE-}"
_CALLER_HF_HUB_OFFLINE="${HF_HUB_OFFLINE-}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Caller/env-prefix wins over .env.
DATA_DIR="${_CALLER_DATA_DIR:-${DATA_DIR:-./data}}"
if [[ "$DATA_DIR" != /* ]]; then
  DATA_DIR="$ROOT/$DATA_DIR"
fi
VIDEOS_DIR="${_CALLER_VIDEOS_DIR:-${VIDEOS_DIR:-$DATA_DIR/videos}}"
STAGES="${_CALLER_STAGES:-${STAGES:-visual,ocr,asr}}"
QDRANT_URL="${_CALLER_QDRANT_URL:-${QDRANT_URL:-http://localhost:6333}}"
ELASTICSEARCH_URL="${_CALLER_ES_URL:-${ELASTICSEARCH_URL:-http://localhost:9200}}"
QDRANT_COLLECTION="${_CALLER_QDRANT_COLLECTION:-${QDRANT_COLLECTION:-video_keyframes}}"
ES_INDEX="${_CALLER_ES_INDEX:-${ES_INDEX:-video_text}}"
START_FROM="${_CALLER_START_FROM:-${START_FROM:-}}"
MAX_VIDEOS="${_CALLER_MAX_VIDEOS:-${MAX_VIDEOS:-0}}"
MAX_RETRIES="${_CALLER_MAX_RETRIES:-${MAX_RETRIES:-2}}"
LOG_FILE="${_CALLER_LOG_FILE:-${LOG_FILE:-$ROOT/reindex_transnet_mac.log}}"
REEXTRACT="${_CALLER_REEXTRACT:-${REEXTRACT:-1}}"

# Force Mac-efficient defaults (override via env before launch if needed)
export VISUAL_BACKEND="${_CALLER_VISUAL_BACKEND:-${VISUAL_BACKEND:-real}}"
export OCR_BACKEND="${_CALLER_OCR_BACKEND:-${OCR_BACKEND:-rapidocr}}"
export ASR_BACKEND="${_CALLER_ASR_BACKEND:-${ASR_BACKEND:-faster_whisper}}"
export SHOT_BACKEND="${_CALLER_SHOT_BACKEND:-${SHOT_BACKEND:-transnetv2}}"
export SIGLIP_BATCH_SIZE="${_CALLER_SIGLIP_BATCH_SIZE:-${SIGLIP_BATCH_SIZE:-16}}"
export MAX_SHOT_SEC="${_CALLER_MAX_SHOT_SEC:-${MAX_SHOT_SEC:-10}}"
export TRANSNET_DEVICE="${_CALLER_TRANSNET_DEVICE:-${TRANSNET_DEVICE:-mps}}"
export OCR_WORKERS="${_CALLER_OCR_WORKERS:-${OCR_WORKERS:-6}}"
export WHISPER_MODEL="${_CALLER_WHISPER_MODEL:-${WHISPER_MODEL:-base}}"
export FASTER_WHISPER_COMPUTE_TYPE="${_CALLER_FASTER_WHISPER_COMPUTE_TYPE:-${FASTER_WHISPER_COMPUTE_TYPE:-int8}}"
export TRANSFORMERS_OFFLINE="${_CALLER_TRANSFORMERS_OFFLINE:-${TRANSFORMERS_OFFLINE:-1}}"
export HF_HUB_OFFLINE="${_CALLER_HF_HUB_OFFLINE:-${HF_HUB_OFFLINE:-1}}"
# video-index reads DATA_DIR from the environment / Settings.
export DATA_DIR
export QDRANT_URL ELASTICSEARCH_URL QDRANT_COLLECTION ES_INDEX

INDEXER="$ROOT/.venv/bin/video-index"
PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$INDEXER" ]]; then
  INDEXER="$(command -v video-index || true)"
fi
if [[ -z "$INDEXER" || ! -x "$PYTHON" ]]; then
  echo "Activate venv and install: pip install -e '.[ml]'" >&2
  exit 1
fi

if [[ ! -d "$VIDEOS_DIR" ]]; then
  echo "Videos directory not found: $VIDEOS_DIR" >&2
  exit 1
fi

# Keep Mac awake for the whole run.
if [[ -z "${_VR_CAFFEINATED:-}" ]] && command -v caffeinate >/dev/null; then
  echo "Keeping Mac awake with caffeinate..."
  export _VR_CAFFEINATED=1
  exec caffeinate -dims -- "$0" "$@"
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

clear_video_index() {
  local video_id="$1"
  "$PYTHON" - "$QDRANT_URL" "$QDRANT_COLLECTION" "$ELASTICSEARCH_URL" "$ES_INDEX" "$video_id" "$STAGES" <<'PY'
import json
import sys
import urllib.error
import urllib.request

qdrant_url, collection, es_url, es_index, video_id, stages_raw = sys.argv[1:]
stages = {part.strip().lower() for part in stages_raw.split(",") if part.strip()}


def request(url, *, method="GET", body=None, allow_missing=False):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        if allow_missing and exc.code in {404}:
            return {}
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"{method} {url} failed ({exc.code}): {detail}") from exc
    return json.loads(payload) if payload else {}


if "visual" in stages:
    request(
        f"{qdrant_url.rstrip('/')}/collections/{collection}/points/delete",
        method="POST",
        body={"filter": {"must": [{"key": "video_id", "match": {"value": video_id}}]}},
        allow_missing=True,
    )
    print(f"  cleared qdrant for {video_id}")

text_sources = [s for s in ("ocr", "asr") if s in stages]
if text_sources:
    request(
        f"{es_url.rstrip('/')}/{es_index}/_delete_by_query?refresh=true",
        method="POST",
        body={
            "query": {
                "bool": {
                    "must": [
                        {"term": {"video_id": video_id}},
                        {"terms": {"source": text_sources}},
                    ]
                }
            }
        },
        allow_missing=True,
    )
    print(f"  cleared es {text_sources} for {video_id}")
PY
}

index_video_with_retry() {
  local video="$1"
  local attempt=1
  local extract_flag=(--reextract)
  if [[ "$REEXTRACT" == "0" ]]; then
    extract_flag=(--reuse-extract)
  fi
  while [[ "$attempt" -le "$MAX_RETRIES" ]]; do
    if "$INDEXER" index "$video" \
      --stages "$STAGES" \
      "${extract_flag[@]}" \
      --rerun \
      --data-dir "$DATA_DIR"; then
      return 0
    fi
    echo "Attempt $attempt/$MAX_RETRIES failed for $video" >&2
    attempt=$((attempt + 1))
    sleep 5
  done
  return 1
}

echo "Checking packages..."
"$PYTHON" - <<'PY'
import importlib
missing = []
for name in ("torch", "transformers", "transnetv2_pytorch", "faster_whisper", "rapidocr"):
    try:
        importlib.import_module(name)
    except ImportError:
        missing.append(name)
if missing:
    raise SystemExit(
        "Missing: "
        + ", ".join(missing)
        + "\nInstall with: pip install -e '.[ml]'"
    )
import torch
print(
    "torch",
    torch.__version__,
    "mps",
    bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
)
PY

echo "Starting Qdrant + Elasticsearch..."
docker compose up -d qdrant elasticsearch
wait_http "$QDRANT_URL/collections" "Qdrant"
wait_http "$ELASTICSEARCH_URL" "Elasticsearch"

VIDEOS=()
while IFS= read -r video; do
  VIDEOS+=("$video")
done < <(find "$VIDEOS_DIR" -maxdepth 1 -type f \( -name '*.mp4' -o -name '*.mov' -o -name '*.mkv' -o -name '*.webm' \) | sort)
if [[ "${#VIDEOS[@]}" -eq 0 ]]; then
  echo "No videos found in $VIDEOS_DIR" >&2
  exit 1
fi

echo "Re-indexing ${#VIDEOS[@]} videos"
echo "  DATA_DIR=$DATA_DIR"
echo "  VIDEOS_DIR=$VIDEOS_DIR"
echo "  stages=$STAGES shot=$SHOT_BACKEND visual=$VISUAL_BACKEND ocr=$OCR_BACKEND asr=$ASR_BACKEND"
echo "  SIGLIP_BATCH_SIZE=$SIGLIP_BATCH_SIZE MAX_SHOT_SEC=$MAX_SHOT_SEC TRANSNET_DEVICE=$TRANSNET_DEVICE"
echo "  QDRANT_COLLECTION=$QDRANT_COLLECTION ES_INDEX=$ES_INDEX"
echo "  REEXTRACT=$REEXTRACT log=$LOG_FILE"
if [[ -n "$START_FROM" ]]; then
  echo "  START_FROM=$START_FROM"
fi

{
  echo "==== $(date) start stages=$STAGES ===="
} >>"$LOG_FILE"

total="${#VIDEOS[@]}"
n=0
done_count=0
failed=0
started=false
t0="$(date +%s)"

for video in "${VIDEOS[@]}"; do
  video_id="$(basename "$video")"
  video_id="${video_id%.*}"

  if [[ -n "$START_FROM" && "$started" == false ]]; then
    if [[ "$video_id" != "$START_FROM" ]]; then
      continue
    fi
    started=true
  fi

  if [[ "$MAX_VIDEOS" -gt 0 && "$done_count" -ge "$MAX_VIDEOS" ]]; then
    echo "Reached MAX_VIDEOS=$MAX_VIDEOS"
    break
  fi

  n=$((n + 1))
  done_count=$((done_count + 1))
  now="$(date +%s)"
  elapsed=$((now - t0))
  if [[ "$done_count" -gt 1 ]]; then
    rate=$((elapsed / (done_count - 1)))
    eta=$((rate * (total - n + 1)))
  else
    eta=0
  fi

  echo
  echo "[$done_count] $video_id  ($n/$total)  elapsed=${elapsed}s  eta≈${eta}s"
  echo "[$done_count] $video_id start $(date)" >>"$LOG_FILE"
  clear_video_index "$video_id"
  if index_video_with_retry "$video"; then
    echo "[$done_count] $video_id ok $(date)" >>"$LOG_FILE"
  else
    echo "FAILED: $video_id" >&2
    echo "[$done_count] $video_id FAIL $(date)" >>"$LOG_FILE"
    failed=$((failed + 1))
  fi
done

echo
echo "Finished. failures=$failed"
echo "==== $(date) done failures=$failed ====" >>"$LOG_FILE"
[[ "$failed" -eq 0 ]]
