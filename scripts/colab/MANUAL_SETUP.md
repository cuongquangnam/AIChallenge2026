# Manual Colab session setup

Laptop hosts the UI; **Colab GPU VM** runs search/KIS/QA on a **persistent worker**
(models loaded once). Preferred control plane: a **Colab notebook** for mount /
bootstrap / worker / keepalive. Configure secrets in laptop **`.env`**.

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
QDRANT_COLLECTION=video_keyframes_transnet
ES_INDEX=video_text_transnet
```

## Recommended setup (notebook + laptop UI)

### Laptop

```bash
./scripts/colab/laptop_open_notebook.sh
```

Or open `scripts/colab/colab_setup.ipynb` in Colab manually.

That starts the session (if needed), uploads `scripts/colab/colab_setup.ipynb`
to `/content/colab_setup.ipynb`, and opens the Colab UI.

### Notebook flow

1. CONFIG (Colab Secrets) → Mount Drive → Clone/pull → Write `.env.colab`
2. Bootstrap (ES + Qdrant)
3. Start worker `:8765`
4. **Cloudflare tunnel** cell → copy `COLAB_WORKER_PUBLIC_URL`
5. Keepalive cell running

### Laptop `.env` for tunnel (no Colab CLI needed for search)

```env
REMOTE_COMPUTE=colab
COLAB_WORKER_PUBLIC_URL=https://xxxx.trycloudflare.com
COLAB_WORKER_MODE=tunnel
```

```bash
video-index colab search "yellow lion"
# or
video-index serve
```

`COLAB_WORKER_MODE`:

| Value | Behavior |
|---|---|
| `auto` (default) | Prefer worker; fall back to oneshot exec if worker is down |
| `persistent` | Require worker (fail if not running) |
| `oneshot` | Old behavior: load models inside every `colab exec` |
| `tunnel` | Laptop POSTs directly to `COLAB_WORKER_PUBLIC_URL` (Cloudflare) |

### Colab notebook

Open `/content/colab_setup.ipynb`:

1. Add Colab Secrets (sidebar key icon): `GEMINI_API_KEY`, `COLAB_REPO_URL`, optional `COLAB_REPO_BRANCH`, `GITHUB_TOKEN`
2. **Runtime → Run all**
3. Leave the keepalive cell running

No laptop `.env` upload is required for this path; the notebook writes `.env.colab` from secrets + defaults.

### Laptop UI

```bash
video-index serve
```

Do **not** run `laptop_monitor_progress.sh` while searching; the notebook
keepalive is enough and avoids Jupyter kernel contention.

## Alternative setup (laptop → CLI only)

```bash
./scripts/colab/laptop_start_session.sh
./scripts/colab/laptop_setup_all.sh
```

Or step by step:

```bash
./scripts/colab/laptop_clone.sh
./scripts/colab/laptop_upload_env.sh
./scripts/colab/laptop_bootstrap.sh
./scripts/colab/laptop_start_worker.sh
```

Drive auth cannot run inside `colab console` / raw `python3` (`drive.mount` needs the notebook UI).
Notebook mount is preferred. CLI equivalent:

```bash
./scripts/colab/laptop_drivemount.sh
# or: colab drivemount -s video-retrieval /content/drive
```

### Persistent worker

After bootstrap, start (or restart) the warm worker from the notebook cell, or:

```bash
./scripts/colab/laptop_start_worker.sh
./scripts/colab/laptop_worker_status.sh
./scripts/colab/laptop_stop_worker.sh    # optional
```

- Listens on the VM at `http://127.0.0.1:8765` (not exposed to the internet)
- With tunnel mode, laptop submits via `POST /jobs` and polls `GET /jobs/{id}` (avoids Cloudflare ~100s cutoff); otherwise jobs use `colab exec` as a thin HTTP proxy
- First start can take several minutes (model load); later searches reuse the process
- Logs: `/content/video-retrieval/worker.log`

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
├── keyframes.zip                     # preferred: zip at Drive data root
│                                     # also OK: keyframes/keyframes.zip or shards
├── keyframes/                        # optional loose JPGs if no zip
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

Default bootstrap pulls the smaller index artifacts:

```text
elasticsearch/
qdrant/
manifests/
```

The Colab notebook sets `PULL_KEYFRAMES=True` and runs bootstrap with
`--with-keyframes`. That **extracts Drive zip(s) directly** into
`/content/data/keyframes/{video_id}/*.jpg` (no full zip copy onto the VM).
Looks for `keyframes.zip` at the Drive data root
(`MyDrive/video-retrieval/keyframes.zip`) or under `keyframes/`.
Chain rerank / QA resolve frames via basename under that folder.

To skip (faster start) or pull later:

```bash
# notebook CONFIG: PULL_KEYFRAMES=False
# or later on the VM:
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
