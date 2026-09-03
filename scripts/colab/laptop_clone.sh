#!/usr/bin/env bash
# Clone/update the repo on Colab via git (reads COLAB_REPO_* from laptop .env).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION="${COLAB_SESSION:-video-retrieval}"
CLI="${COLAB_CLI:-colab}"
TIMEOUT="${COLAB_TIMEOUT_SEC:-600}"
DOTENV="$ROOT/.env"
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"

echo "Checking session $SESSION..."
if ! colab_session_active "$CLI" "$SESSION"; then
  echo "Session is not reachable. Start it first:"
  echo "  ./scripts/colab/laptop_start_session.sh"
  exit 1
fi
status_out=$("$CLI" status -s "$SESSION" 2>&1 || true)
echo "$status_out"

REPO_URL="$(dotenv_get "$DOTENV" COLAB_REPO_URL || true)"
REPO_BRANCH="$(dotenv_get "$DOTENV" COLAB_REPO_BRANCH || true)"
GITHUB_TOKEN="$(dotenv_get "$DOTENV" GITHUB_TOKEN || true)"

ENV_ARGS=()
[[ -n "$REPO_URL" ]] && ENV_ARGS+=("COLAB_REPO_URL=$REPO_URL")
[[ -n "$REPO_BRANCH" ]] && ENV_ARGS+=("COLAB_REPO_BRANCH=$REPO_BRANCH")
[[ -n "$GITHUB_TOKEN" ]] && ENV_ARGS+=("GITHUB_TOKEN=$GITHUB_TOKEN")

if [[ -z "$REPO_URL" ]]; then
  echo "COLAB_REPO_URL is required in $DOTENV"
  exit 1
fi

echo "Cloning/updating repo on Colab (branch=${REPO_BRANCH:-main})..."
if ! colab_exec_script "$CLI" "$SESSION" "$TIMEOUT" "$ROOT/scripts/colab/clone_remote.py" "${ENV_ARGS[@]}"; then
  echo "Clone failed."
  exit 1
fi
if ! colab_remote_dir_exists "$CLI" "$SESSION" "/content/video-retrieval/.git"; then
  echo "Clone did not create /content/video-retrieval/.git — check repo URL/branch and retry."
  exit 1
fi
echo "Clone done. Next: ./scripts/colab/laptop_upload_env.sh && ./scripts/colab/laptop_bootstrap.sh"
