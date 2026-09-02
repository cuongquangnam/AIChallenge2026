#!/usr/bin/env bash
# Run set_remote_env.py on Colab WITHOUT passing the key from this laptop.
# Add GEMINI_API_KEY to Colab Secrets first, or run set_remote_env.py in a notebook cell.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION="${COLAB_SESSION:-video-retrieval}"
CLI="${COLAB_CLI:-colab}"

echo "Running set_remote_env.py on Colab session $SESSION..."
echo "Ensure GEMINI_API_KEY is in Colab Secrets (lock icon) or exported in the VM."
"$CLI" exec -s "$SESSION" -f "$ROOT/scripts/colab/set_remote_env.py"
echo "Done. Next: ./scripts/colab/laptop_bootstrap.sh"
