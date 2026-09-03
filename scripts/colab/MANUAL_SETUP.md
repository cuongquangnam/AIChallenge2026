# Manual Colab session setup

Laptop hosts the UI; **Colab CLI VM** runs search/KIS/QA on a **persistent worker**
(models loaded once). Configure everything in **`.env`** on your laptop.

## Prerequisites

```bash
pip install 'google-colab-cli' 'jupyter-kernel-client==0.15.0'
colab auth login
chmod +x scripts/colab/*.sh
cp .env.example .env
```

Required in `.env`:

```env
REMOTE_COMPUTE=colab
COLAB_SESSION=video-retrieval
COLAB_AUTO_MANAGE=false
COLAB_WORKER_MODE=auto
DRIVE_DATA_PATH=MyDrive/video-retrieval
DRIVE_LOCAL_PATH=

COLAB_REPO_URL=https://github.com/you/AIChallenge2026_2.git
COLAB_REPO_BRANCH=cuong/use-colab
GEMINI_API_KEY=your-key
```

## Setup (laptop → CLI VM)

```bash
./scripts/colab/laptop_start_session.sh
./scripts/colab/laptop_setup_all.sh
```

Or step by step:

```bash
./scripts/colab/laptop_clone.sh
./scripts/colab/laptop_upload_env.sh
./scripts/colab/laptop_bootstrap.sh      # runs colab drivemount, then bootstrap_remote.py
./scripts/colab/laptop_start_worker.sh   # long-running process; loads SigLIP/BLIP once
```

While bootstrap is running, monitor progress from another laptop terminal:

```bash
./scripts/colab/laptop_monitor_progress.sh        # poll every 15s
./scripts/colab/laptop_monitor_progress.sh 30     # poll every 30s
./scripts/colab/laptop_monitor_progress.sh --once # single snapshot
```

Drive auth cannot run inside `colab console` / raw `python3` (`drive.mount` needs the notebook UI).
The laptop scripts call `colab drivemount` for you. Manual equivalent:

```bash
./scripts/colab/laptop_drivemount.sh
# or: colab drivemount -s video-retrieval /content/drive
```

### Persistent worker

After bootstrap, start (or restart) the warm worker:

```bash
./scripts/colab/laptop_start_worker.sh
./scripts/colab/laptop_worker_status.sh
./scripts/colab/laptop_stop_worker.sh    # optional
```

- Listens on the VM at `http://127.0.0.1:8765` (not exposed to the internet)
- Laptop jobs still use `colab exec`, but only as a thin HTTP proxy to that worker
- First start can take several minutes (model load); later searches reuse the process
- Logs: `/content/video-retrieval/worker.log` (view via `colab console` or `colab log`)

`COLAB_WORKER_MODE`:

| Value | Behavior |
|---|---|
| `auto` (default) | Prefer worker; fall back to oneshot exec if worker is down |
| `persistent` | Require worker (fail if not running) |
| `oneshot` | Old behavior: load models inside every `colab exec` |

### What `laptop_upload_env.sh` does

Reads your laptop **`.env`**, extracts Colab-relevant keys (`GEMINI_API_KEY`, `DRIVE_DATA_PATH`, …), and writes them to `/content/video-retrieval/.env.colab` on the CLI VM. Search jobs do not resend the API key — the worker loads `.env.colab` on the VM.

## Use

```bash
video-index serve
video-index colab search "your query"
```

In `ps` on the VM you should see a long-running `uvicorn ...worker_server:app` process while the worker is up.

## Drive layout

```
My Drive/video-retrieval/
├── elasticsearch/
│   └── video_text_transnet.ndjson    # bulk ES export (not an ES snapshot)
├── qdrant/
│   └── video_keyframes_transnet.snapshot   # Qdrant collection snapshot
├── keyframes/
│   └── keyframes.zip                 # or keyframes_000.zip, keyframes_001.zip, ...
│                                     # zip archives are extracted automatically on the VM
├── manifests/                        # optional fallback for ES + QA metadata
└── videos/                           # optional; QA pulls lazily on demand
```

For **TransNet** data, set in your laptop `.env` before `laptop_upload_env.sh`:

```env
QDRANT_COLLECTION=video_keyframes_transnet
ES_INDEX=video_text_transnet
```

After changing `.env`, re-upload env and re-run bootstrap:

```bash
./scripts/colab/laptop_upload_env.sh
./scripts/colab/laptop_bootstrap.sh
# or full: ./scripts/colab/laptop_setup_all.sh
```

Default bootstrap pulls only the smaller index artifacts:

```text
elasticsearch/
qdrant/
manifests/
```

Keyframes stay on Drive by default. If you later need the full keyframe archive(s)
on the VM for image-heavy reranking/debugging, run on the VM:

```bash
cd /content/video-retrieval
python3 scripts/colab/pull_data_remote.py --with-keyframes
```

## Troubleshooting

```bash
colab status -s video-retrieval
./scripts/colab/laptop_worker_status.sh
colab console -s video-retrieval
# then: tail -n 100 /content/video-retrieval/worker.log

pip install 'jupyter-kernel-client==0.15.0'   # if colab exec fails with KernelClient error
```

Session lost → repeat `laptop_start_session.sh` and `laptop_setup_all.sh` (or at least bootstrap + start worker).
