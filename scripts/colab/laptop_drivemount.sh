#!/usr/bin/env bash
# Mount Google Drive on the Colab VM via CLI (no notebook UI required).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION="${COLAB_SESSION:-video-retrieval}"
CLI="${COLAB_CLI:-colab}"
MOUNT="${COLAB_DRIVE_MOUNT:-/content/drive}"
DOTENV="$ROOT/.env"
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"

if [[ -f "$DOTENV" ]]; then
  from_env="$(dotenv_get "$DOTENV" DRIVE_MOUNT || true)"
  [[ -n "$from_env" ]] && MOUNT="$from_env"
fi

if ! colab_session_active "$CLI" "$SESSION"; then
  echo "Session is not reachable. Start it first:"
  echo "  ./scripts/colab/laptop_start_session.sh"
  exit 1
fi

colab_ensure_drive_mounted "$CLI" "$SESSION" "$MOUNT"
echo "Done. Verify with: ls via colab exec, or continue bootstrap."
