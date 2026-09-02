# Manual Colab session setup

Laptop hosts the UI; **Colab CLI VM** runs search/KIS/QA. Configure everything in **`.env`** on your laptop.

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
DRIVE_DATA_PATH=MyDrive/video-retrieval
DRIVE_LOCAL_PATH=/path/to/synced/video-retrieval

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
./scripts/colab/laptop_clone.sh        # git clone (uses COLAB_REPO_* from .env)
./scripts/colab/laptop_upload_env.sh   # upload Colab vars from .env → VM .env.colab
./scripts/colab/laptop_bootstrap.sh    # pip install + Drive pull + Elasticsearch
```

### What `laptop_upload_env.sh` does

Reads your laptop **`.env`**, extracts Colab-relevant keys (`GEMINI_API_KEY`, `DRIVE_DATA_PATH`, …), and writes them to `/content/video-retrieval/.env.colab` on the CLI VM. Search jobs do not resend the API key — the worker loads `.env.colab` on the VM.

## Use

```bash
video-index serve
video-index colab search "your query"
```

## Drive layout

```
My Drive/video-retrieval/
├── elasticsearch/video_text*.ndjson
├── qdrant/
├── keyframes/
├── manifests/
└── videos/
```

## Troubleshooting

```bash
colab status -s video-retrieval
pip install 'jupyter-kernel-client==0.15.0'   # if colab exec fails with KernelClient error
```

Session lost → repeat `laptop_start_session.sh` and `laptop_setup_all.sh`.
