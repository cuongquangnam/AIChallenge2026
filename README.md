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

# 3) Index a video
video-index index /path/to/video.mp4

# 4) Search
video-index search "your query" --mode hybrid

# 5) API
video-index serve
# POST /index  { "path": "/path/to/video.mp4" }
# POST /search { "query": "...", "mode": "hybrid" }
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

- `text` — keyword search over OCR + ASR in Elasticsearch  
- `visual` — text→SigLIP embedding, nearest keyframes in Qdrant  
- `hybrid` — reciprocal-rank fusion of both

## Task 2: music-award evidence retrieval

`task2-candidates` retrieves evidence for the question about how many winners
walk on stage to accept the largest music award. It searches visual keyframes,
OCR, and ASR with channel-specific queries; fuses their rankings; then groups
nearby results from the same video into event windows.

```bash
video-index task2-candidates --video-id L27_V001
```

The equivalent API endpoint is `POST /task2/candidates`:

```json
{
  "video_id": "L27_V001",
  "candidates_per_query": 20,
  "group_limit": 10,
  "max_gap_sec": 10,
  "max_gap_frames": 10,
  "context_radius_frames": 5
}
```

Each returned group contains evidence hits, a score, frame IDs around its
center, and nearby keyframe paths from the index manifest. The caller can send
those paths to a VLM to verify that it is the major award and count recipients.

## Tests

```bash
# Unit + offline integration (in-memory Qdrant / fake ES)
pytest -m "unit or integration" -q

# Live Elasticsearch integration (needs docker compose up)
docker compose up -d elasticsearch
pytest tests/integration -q
```
