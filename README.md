# Video Retrieval Pipeline

Offline indexing + search for a video collection, matching this architecture:

1. **Key Frame Extraction** — shot detection → start / middle / end frames  
2. **Audio Extraction** — WAV for ASR  
3. **Visual indexing** — SigLIP + BEiT(3) embeddings → **Qdrant**  
4. **Textual indexing** — Gemini OCR + Whisper ASR → **Elasticsearch**

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
  encoders/     # SigLIP + BEiT visual encoders
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
| Visual encoders | `VISUAL_BACKEND` | `mock` \| `real` |
| OCR | `OCR_BACKEND` | `mock` \| `gemini` |
| ASR | `ASR_BACKEND` | `mock` \| `whisper` |

Real models need:

```bash
pip install -e ".[ml]"
# + GEMINI_API_KEY for OCR
# + ffmpeg on PATH for audio extraction
```

## Search modes

- `visual` — SigLIP nearest keyframes in Qdrant  
- `asr` — spoken-text search in Elasticsearch  
- `ocr` — on-screen text search in Elasticsearch  
- `mixed` — Gemini (or heuristic) splits the query into those three channels, then **scores each frame** as a weighted mix

## Index stages

By default indexing runs **visual + OCR + ASR**. Split them with:

- `--only visual` — SigLIP/BEiT → Qdrant  
- `--only ocr` — Gemini OCR → Elasticsearch  
- `--only asr` — Whisper ASR → Elasticsearch  
- `--stages visual,ocr` — any subset  

`--reuse-extract` (default) reuses keyframes/audio/manifest when present; `--reextract` forces shot/audio extraction again.

`QUERY_PLANNER=auto` uses Gemini when `GEMINI_API_KEY` is set, otherwise the full query is sent to the selected channel(s).

## Tests

```bash
# Unit + offline integration (in-memory Qdrant / fake ES)
pytest -m "unit or integration" -q

# Live Elasticsearch integration (needs docker compose up)
docker compose up -d elasticsearch
pytest tests/integration -q
```
