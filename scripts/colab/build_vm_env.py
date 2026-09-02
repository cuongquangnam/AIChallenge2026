#!/usr/bin/env python3
"""Build a VM-ready .env.colab from the repo .env (laptop only)."""
from __future__ import annotations

import argparse
from pathlib import Path

# Copied to the Colab VM (excludes laptop-only keys like COLAB_REPO_URL, API_PORT).
VM_KEYS = (
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "GEMINI_RPM",
    "GEMINI_MAX_RETRIES",
    "GEMINI_BATCH_SIZE",
    "QUERY_PLANNER",
    "VISUAL_BACKEND",
    "OCR_BACKEND",
    "ASR_BACKEND",
    "ELASTICSEARCH_URL",
    "ES_INDEX",
    "QDRANT_COLLECTION",
    "WHISPER_MODEL",
    "SIGLIP_MODEL_ID",
    "BEIT3_MODEL_ID",
    "SIGLIP_DIM",
    "BEIT3_DIM",
    "DRIVE_MOUNT",
    "DRIVE_DATA_PATH",
    "COLAB_ELASTICSEARCH_INSTALL_DIR",
)


def _parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def build_vm_env(*, source: Path, dest: Path) -> None:
    values = _parse_env(source)
    if not values.get("GEMINI_API_KEY", "").strip():
        raise SystemExit(f"GEMINI_API_KEY is required in {source}")

    lines = [
        "# Colab VM env — uploaded from laptop .env. Do not commit.",
        f"# Source: {source}",
    ]
    for key in VM_KEYS:
        value = values.get(key, "").strip()
        if value:
            lines.append(f"{key}={value}")

    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {dest} ({len(lines) - 2} vars)")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build VM .env.colab from repo .env")
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=repo_root / ".env",
    )
    parser.add_argument(
        "dest",
        nargs="?",
        type=Path,
        default=Path("/tmp/video-retrieval.env.colab"),
    )
    args = parser.parse_args()
    if not args.source.is_file():
        raise SystemExit(f"{args.source} not found. Copy .env.example to .env and fill in values.")
    build_vm_env(source=args.source, dest=args.dest)


if __name__ == "__main__":
    main()
