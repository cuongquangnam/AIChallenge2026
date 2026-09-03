#!/usr/bin/env bash
# Poll Colab VM setup/pull progress from the laptop.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION="${COLAB_SESSION:-video-retrieval}"
CLI="${COLAB_CLI:-colab}"
INTERVAL="${1:-15}"
SCRIPTS="$ROOT/scripts/colab"
# shellcheck source=lib.sh
source "$SCRIPTS/lib.sh"

if [[ "$INTERVAL" == "--once" ]]; then
  INTERVAL=0
fi

if ! [[ "$INTERVAL" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 [interval_seconds|--once]" >&2
  exit 2
fi

if ! colab_session_active "$CLI" "$SESSION"; then
  echo "Session is not reachable. Start it first:"
  echo "  ./scripts/colab/laptop_start_session.sh"
  exit 1
fi

echo "Monitoring Colab session $SESSION (interval=${INTERVAL}s). Ctrl+C to stop."
while true; do
  printf '\n'
  if ! "$CLI" exec -s "$SESSION" -f "$SCRIPTS/monitor_progress_remote.py" --timeout 120; then
    echo "Monitor poll failed; retrying..."
  fi
  if [[ "$INTERVAL" == "0" ]]; then
    break
  fi
  sleep "$INTERVAL"
done
