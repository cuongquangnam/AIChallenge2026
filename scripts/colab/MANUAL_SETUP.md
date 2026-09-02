# Manual Colab session setup

The laptop **does not** auto-start Colab or send `GEMINI_API_KEY` in job payloads.
You start the session, **git clone** the repo on Colab, and set secrets **on the VM**.

## Prerequisites (laptop)

```bash
pip install google-colab-cli   # provides the `colab` command
colab auth login               # once
```

`.env` on the laptop:

```env
REMOTE_COMPUTE=colab
COLAB_SESSION=video-retrieval
COLAB_AUTO_MANAGE=false
DRIVE_DATA_PATH=MyDrive/video-retrieval
DRIVE_LOCAL_PATH=/path/to/synced/video-retrieval   # for UI keyframes
# Do NOT put GEMINI_API_KEY on the laptop for remote mode.
```

## Colab Secrets (set in the notebook UI, lock icon)

| Secret | Example |
|--------|---------|
| `COLAB_REPO_URL` | `https://github.com/you/AIChallenge2026_2.git` |
| `COLAB_REPO_BRANCH` | `main` (optional) |
| `GITHUB_TOKEN` | for private repos (optional) |
| `GEMINI_API_KEY` | your Gemini key |

## Step 1 — Start a Colab session (laptop)

```bash
chmod +x scripts/colab/*.sh
./scripts/colab/laptop_start_session.sh
```

Or start a notebook at [colab.research.google.com](https://colab.research.google.com) with session name `video-retrieval`.

## Step 2 — Clone the repo on Colab

**Option A — from laptop** (reads `COLAB_REPO_URL` from Colab Secrets):

```bash
./scripts/colab/laptop_clone.sh
```

**Option B — in a Colab notebook cell** (after adding Secrets):

```python
!python3 -c "
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'requests'])
"
# If clone script not on VM yet, one-time bootstrap:
!git clone --depth 1 https://github.com/you/AIChallenge2026_2.git /content/video-retrieval
!python3 /content/video-retrieval/scripts/colab/clone_remote.py
```

Or simply:

```python
!git clone --branch main --depth 1 https://github.com/you/AIChallenge2026_2.git /content/video-retrieval
```

To update an existing clone later:

```bash
./scripts/colab/laptop_clone.sh   # runs git pull on Colab
```

**Option C — upload local uncommitted changes** (slow; use only when needed):

```bash
./scripts/colab/laptop_upload.sh
```

## Step 3 — Set API key on Colab (NOT on laptop)

After clone, with `GEMINI_API_KEY` in Colab Secrets:

```python
!python3 /content/video-retrieval/scripts/colab/set_remote_env.py
```

Or from laptop:

```bash
./scripts/colab/laptop_set_remote_env.sh
```

Writes `/content/video-retrieval/.env.colab` on the VM only.

## Step 4 — Bootstrap: install + pull Drive data (laptop)

```bash
./scripts/colab/laptop_bootstrap.sh
```

Runs `pip install -e ".[ml]"`, mounts Drive, pulls index data, loads Elasticsearch.

## Step 5 — Use from laptop

```bash
video-index serve
video-index colab search "your query"
```

Each search uploads a small `request.json` (no API key) and runs `colab exec`.

## Drive layout

```
My Drive/video-retrieval/
├── elasticsearch/video_text*.ndjson
├── qdrant/
├── keyframes/
├── manifests/
└── videos/          # QA pulls lazily per video
```

## Troubleshooting

```bash
colab status -s video-retrieval
colab exec -s video-retrieval -f scripts/colab/clone_remote.py
colab exec -s video-retrieval -c "ls -la /content/video-retrieval"
```

If the session disconnects, repeat from Step 1 (clone is fast with `--depth 1`).
