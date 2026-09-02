#!/usr/bin/env python3
"""Write Colab-only secrets to /content/video-retrieval/.env.colab (run on the VM).

Set GEMINI_API_KEY once per Colab session using one of:
  1. Colab Secrets (recommended): add GEMINI_API_KEY in the notebook sidebar, then run this script.
  2. Shell before exec: export GEMINI_API_KEY=...  then run this script.
  3. Interactive prompt when neither is available.
"""
from __future__ import annotations

import getpass
import os
from pathlib import Path

REPO_ROOT = Path("/content/video-retrieval")
ENV_FILE = REPO_ROOT / ".env.colab"


def _read_gemini_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key

    try:
        from google.colab import userdata

        key = userdata.get("GEMINI_API_KEY").strip()
        if key:
            return key
    except Exception:
        pass

    entered = getpass.getpass("GEMINI_API_KEY (input hidden): ").strip()
    if not entered:
        raise SystemExit("GEMINI_API_KEY is required.")
    return entered


def main() -> None:
    key = _read_gemini_api_key()
    REPO_ROOT.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Colab session secrets — do not commit. Created by scripts/colab/set_remote_env.py",
        f"GEMINI_API_KEY={key}",
    ]
    optional = [
        "GEMINI_MODEL",
        "QUERY_PLANNER",
        "VISUAL_BACKEND",
        "ELASTICSEARCH_URL",
    ]
    for name in optional:
        value = os.environ.get(name, "").strip()
        if value:
            lines.append(f"{name}={value}")

    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.environ["GEMINI_API_KEY"] = key
    print(f"Wrote {ENV_FILE} ({len(lines)} vars). GEMINI_API_KEY is set for this session.")


if __name__ == "__main__":
    main()
