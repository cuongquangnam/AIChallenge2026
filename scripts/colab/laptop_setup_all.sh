#!/usr/bin/env bash
# Full Colab VM setup from laptop .env (clone + upload .env + bootstrap + worker).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION="${COLAB_SESSION:-video-retrieval}"
CLI="${COLAB_CLI:-colab}"
CLONE_TIMEOUT="${COLAB_TIMEOUT_SEC:-600}"
BOOTSTRAP_TIMEOUT="${COLAB_BOOTSTRAP_TIMEOUT_SEC:-3600}"
DOTENV="$ROOT/.env"
SCRIPTS="$ROOT/scripts/colab"
# shellcheck source=lib.sh
source "$SCRIPTS/lib.sh"

echo "Checking session $SESSION..."
if ! colab_session_active "$CLI" "$SESSION"; then
  echo "Session is not reachable. Start it first:"
  echo "  ./scripts/colab/laptop_start_session.sh"
  exit 1
fi
"$CLI" status -s "$SESSION" 2>&1 || true

if [[ ! -f "$DOTENV" ]]; then
  echo "Missing $DOTENV — copy .env.example to .env and set COLAB_REPO_URL + GEMINI_API_KEY."
  exit 1
fi

COLAB_ENV_ARGS=()
colab_env_args_from_file "$DOTENV" \
  COLAB_REPO_URL COLAB_REPO_BRANCH GITHUB_TOKEN

has_repo=false
has_gemini=false
gemini_key="$(dotenv_get "$DOTENV" GEMINI_API_KEY || true)"
repo_url="$(dotenv_get "$DOTENV" COLAB_REPO_URL || true)"
[[ -n "$repo_url" ]] && has_repo=true
[[ -n "$gemini_key" ]] && has_gemini=true

if ! $has_repo; then
  echo "COLAB_REPO_URL is required in $DOTENV"
  exit 1
fi
if ! $has_gemini; then
  echo "GEMINI_API_KEY is required in $DOTENV"
  exit 1
fi

echo ""
echo "Step 1/4 — Clone repo on CLI VM..."
if ! colab_exec_script "$CLI" "$SESSION" "$CLONE_TIMEOUT" "$SCRIPTS/clone_remote.py" "${COLAB_ENV_ARGS[@]}"; then
  echo "Clone failed."
  exit 1
fi
if ! colab_remote_dir_exists "$CLI" "$SESSION" "/content/video-retrieval/.git"; then
  echo "Clone did not create /content/video-retrieval/.git"
  exit 1
fi

echo ""
echo "Step 2/4 — Upload .env to CLI VM..."
if ! "$SCRIPTS/laptop_upload_env.sh"; then
  echo "Env upload failed."
  exit 1
fi

echo ""
echo "Step 3/4 — Mount Drive, install package, pull data (may take several minutes)..."
MOUNT="/content/drive"
from_env="$(dotenv_get "$DOTENV" DRIVE_MOUNT || true)"
[[ -n "$from_env" ]] && MOUNT="$from_env"
colab_ensure_drive_mounted "$CLI" "$SESSION" "$MOUNT"
if ! "$CLI" exec -s "$SESSION" -f "$SCRIPTS/bootstrap_remote.py" --timeout "$BOOTSTRAP_TIMEOUT"; then
  echo "Bootstrap failed."
  exit 1
fi

echo ""
echo "Step 4/4 — Start persistent Colab worker (loads models once)..."
if ! "$SCRIPTS/laptop_start_worker.sh"; then
  echo "Worker start failed. You can retry: ./scripts/colab/laptop_start_worker.sh"
  exit 1
fi

echo ""
echo "Setup complete on CLI session $SESSION."
echo "  video-index serve"
echo "  video-index colab search \"your query\""
echo "  ./scripts/colab/laptop_worker_status.sh"
