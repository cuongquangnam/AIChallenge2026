#!/usr/bin/env bash
# Start a named Colab GPU session (run on your laptop).
set -euo pipefail

SESSION="${COLAB_SESSION:-video-retrieval}"
GPU="${COLAB_GPU:-T4}"
HIGH_MEM="${COLAB_HIGH_MEM:-false}"
CLI="${COLAB_CLI:-colab}"

ARGS=(new -s "$SESSION" --gpu "$GPU")
if [[ "$HIGH_MEM" == "true" ]]; then
  ARGS+=(--high-mem)
fi

echo "Starting Colab session: $SESSION (GPU=$GPU)"
"$CLI" "${ARGS[@]}"
echo "Session started. Next: ./scripts/colab/laptop_setup_all.sh"
