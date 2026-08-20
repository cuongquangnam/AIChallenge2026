# Video Retrieval Pipeline

Offline indexing + search for a video collection, matching this architecture:

1. **Key Frame Extraction** — shot detection → start / middle / end frames  
2. **Audio Extraction** — WAV for ASR  
3. **Visual indexing** — SigLIP embeddings of start / middle / end keyframes → **Qdrant**
4. **Textual indexing** — Gemini OCR (middle frames) + Whisper ASR → **Elasticsearch**

Default backends are lightweight mocks so you can wire infra and run end-to-end without GPU weights. Flip env flags for real models.

## Quick start

```bash
# 1) Infra
docker compose up -d

# 2) Python env
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env

# 3) Index a video (all stages, or split them)
video-index index /path/to/video.mp4 --data-dir /path/to/outputs
video-index index /path/to/video.mp4 --only visual
video-index index /path/to/video.mp4 --only ocr --data-dir /path/to/outputs
video-index index /path/to/video.mp4 --only asr
video-index index /path/to/video.mp4 --stages visual,ocr

# 4) Search
video-index search "your query" --mode mixed

# 5) API
video-index serve
# POST /index  { "path": "/path/to/video.mp4" }
# POST /search { "query": "...", "mode": "mixed" }
```

## Layout

```
src/video_retrieval/
  extraction/   # shots, keyframes, audio
  encoders/     # SigLIP visual encoder
  text/         # Gemini OCR, Whisper ASR
  storage/      # Qdrant + Elasticsearch clients
  pipeline/     # offline indexer orchestrator
  search/       # text / visual / hybrid retrieval
  api.py        # FastAPI
  cli.py        # Typer CLI
```

## Backends

| Component | Env | Values |
|-----------|-----|--------|
| Shot detection | `SHOT_BACKEND` | `opencv` (default) \| `transnetv2` |
| Visual encoders | `VISUAL_BACKEND` | `mock` \| `real` (uses Apple MPS on Mac) |
| OCR | `OCR_BACKEND` | `mock` \| `rapidocr` (on-device) \| `gemini` |
| ASR | `ASR_BACKEND` | `mock` \| `whisper` |
| Query planner | `QUERY_PLANNER` | `heuristic` \| `ollama` (local LLM) \| `auto` \| `gemini` |

Run the whole pipeline on a Mac with no Gemini key:

```bash
pip install -e ".[ml]"
# .env
# VISUAL_BACKEND=real
# OCR_BACKEND=rapidocr
# OCR_WORKERS=4
# ASR_BACKEND=whisper
# QUERY_PLANNER=heuristic
# + ffmpeg on PATH for audio extraction
```

`OCR_BACKEND=gemini` still works if you want cloud OCR (`GEMINI_API_KEY` required).

## Search modes

- `visual` — SigLIP nearest keyframes in Qdrant  
- `asr` — spoken-text search in Elasticsearch  
- `ocr` — on-screen text search in Elasticsearch  
- `mixed` — planner (heuristic / Ollama / Gemini) splits the query into OCR, ASR, and visual, then **scores each frame** as a weighted mix

### Textual KIS batch (100 answers per query)

Put queries in a JSON object (`query_id` → text), then:

```bash
# uses QUERY_PLANNER / GEMINI_* from .env
video-index kis queries/kis_p1.json --out-dir submissions/kis_p1 --limit 100
# or:
python scripts/run_kis_p1.py
```

Each output file is `{query_id}.csv` with 100 lines of `video_id,frame_idx` (no header).

## Index stages

By default indexing runs **visual + OCR + ASR**. Split them with:

- `--only visual` — SigLIP of start / middle / end keyframes → Qdrant  
- `--only ocr` — RapidOCR or Gemini on **middle** frames → Elasticsearch  
- `--only asr` — Whisper ASR → Elasticsearch  
- `--stages visual,ocr` — any subset  

`--reuse-extract` (default) reuses keyframes/audio/manifest when present; `--reextract` forces shot/audio extraction again.

To run **only OCR + ASR** on videos whose keyframes are already in `data/keyframes/`:

```bash
docker compose up -d elasticsearch
video-index index data/videos --stages ocr,asr
# or, same thing from the keyframe folders:
video-index index data/keyframes --stages ocr,asr
```

That skips shot detection as long as `shot_XXXX_{start,middle,end}.jpg` files exist. ASR still uses `data/audio/<id>.wav`. Elasticsearch must be running.

Overnight / long run on videos already in `data/videos` (Mac stays awake, videos are not copied):

```bash
./scripts/index_existing_videos.sh
# resume after a crash / timeout (skips videos already done):
./scripts/index_existing_videos.sh
# force every video again:
# add --rerun
STAGES=ocr,asr ./scripts/index_existing_videos.sh --rerun
```

`QUERY_PLANNER=auto` uses Gemini when `GEMINI_API_KEY` is set, otherwise the full query is sent to the selected channel(s).

`QUERY_PLANNER=ollama` uses a local model via [Ollama](https://ollama.com) (`OLLAMA_MODEL`, default `llama3.2`). If Ollama is down, search falls back to heuristic.

## Share Qdrant + Elasticsearch (Google Drive)

Export the local Docker **search indexes** (not Docker volume folders, manifests, or keyframe files):

```bash
docker compose up -d
./scripts/export_search_backup.sh
```

That writes `backups/aic-search-backup.zip`. Upload **only that zip** to Drive. The script snapshots Qdrant collection `video_keyframes` and uses `scripts/es_backup.py` to dump Elasticsearch index `video_text`.

The zip includes enough metadata to search (`shot_index`, `role`, timestamps, `keyframe_path`). It does **not** include:

- full shot lists (`data/manifests/*.json` with `start_frame` / `end_frame`)
- keyframe JPEGs (`data/keyframes/`)
- videos or audio

Teammates can restore indexes and search. They still need `data/manifests/` and `data/keyframes/` to browse shots or re-index.

```bash
docker compose up -d
./scripts/restore_search_backup.sh /path/to/aic-search-backup.zip
```

This replaces local `video_keyframes` (Qdrant) and `video_text` (Elasticsearch). Keep Qdrant `v1.12.5` and Elasticsearch `8.15.3` from `docker-compose.yml`. Do not share `.env`.

## Tests

```bash
# Unit + offline integration (in-memory Qdrant / fake ES)
pytest -m "unit or integration" -q

# Live Elasticsearch integration (needs docker compose up)
docker compose up -d elasticsearch
pytest tests/integration -q
```
