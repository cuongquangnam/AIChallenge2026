#!/usr/bin/env bash
# Upload Colab-relevant vars from laptop .env → CLI VM as .env.colab
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION="${COLAB_SESSION:-video-retrieval}"
CLI="${COLAB_CLI:-colab}"
DOTENV="$ROOT/.env"
REMOTE_ENV="/content/video-retrieval/.env.colab"
REMOTE_ENV_FALLBACK="/content/.env.colab"
SCRIPTS="$ROOT/scripts/colab"
# shellcheck source=lib.sh
source "$SCRIPTS/lib.sh"

echo "Checking session $SESSION..."
if ! colab_session_active "$CLI" "$SESSION"; then
  echo "Session is not reachable. Start it first:"
  echo "  ./scripts/colab/laptop_start_session.sh"
  exit 1
fi

if [[ ! -f "$DOTENV" ]]; then
  echo "Missing $DOTENV"
  echo "  cp .env.example .env"
  exit 1
fi

tmp_env="$(mktemp "${TMPDIR:-/tmp}/vr-env.colab.XXXXXX")"
trap 'rm -f "$tmp_env"' EXIT
python3 "$SCRIPTS/build_vm_env.py" "$DOTENV" "$tmp_env"

echo "Uploading .env (Colab vars) to CLI VM..."
if printf 'import os, sys\nsys.exit(0 if os.path.isdir("/content/video-retrieval") else 1)\n' \
  | "$CLI" exec -s "$SESSION" --timeout 30; then
  colab_write_remote_file "$CLI" "$SESSION" "$REMOTE_ENV" "$tmp_env"
else
  echo "Repo not cloned yet — writing $REMOTE_ENV_FALLBACK (run laptop_clone.sh or laptop_setup_all.sh)."
  colab_write_remote_file "$CLI" "$SESSION" "$REMOTE_ENV_FALLBACK" "$tmp_env"
fi

echo "Done."
