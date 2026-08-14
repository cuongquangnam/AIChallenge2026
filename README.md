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
# POST /qa     { "question": "...", "group_count": 10, "frame_radius": 5 }
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

## Video question answering

The Q&A pipeline asks an LLM to decompose a question, fuses OCR/ASR and visual
retrieval to select a video and candidate keyframes, samples neighboring frames,
then asks a multimodal LLM for `video_id`, `frame_id`, and `answer`.

Configure an OpenAI-compatible multimodal endpoint in `.env`:

```bash
QA_LLM_BACKEND=openai_compatible
QA_LLM_API_KEY=...
QA_LLM_BASE_URL=https://api.openai.com/v1
QA_LLM_MODEL=your-multimodal-model
```

Then run:

```bash
video-index qa "Trong video về lễ trao giải thưởng âm nhạc, có bao nhiêu người lên sân khấu?"
```

## Tests

```bash
# Unit + offline integration (in-memory Qdrant / fake ES)
pytest -m "unit or integration" -q

# Live Elasticsearch integration (needs docker compose up)
docker compose up -d elasticsearch
pytest tests/integration -q
```
