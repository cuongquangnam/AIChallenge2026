#!/usr/bin/env python3
"""Clone or update the repo on Colab at /content/video-retrieval.

Configure the repo URL on the Colab VM (not on your laptop), using one of:
  1. Colab Secret: COLAB_REPO_URL (and optional COLAB_REPO_BRANCH)
  2. Environment: export COLAB_REPO_URL=https://github.com/you/repo.git
  3. CLI: python clone_remote.py --repo-url https://github.com/you/repo.git

For private GitHub repos, add a Colab Secret GITHUB_TOKEN and use a HTTPS URL like:
  https://github.com/you/repo.git
(the script injects the token into the clone URL).
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse, urlunparse

REPO_ROOT = Path("/content/video-retrieval")
DEFAULT_BRANCH = "main"


def _secret(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    try:
        from google.colab import userdata

        return userdata.get(name).strip()
    except Exception:
        return ""


def _auth_repo_url(url: str) -> str:
    token = _secret("GITHUB_TOKEN")
    if not token or url.startswith("git@"):
        return url
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return url
    if parsed.username:
        return url
    netloc = f"{token}@{parsed.hostname}"
    if parsed.port:
        netloc += f":{parsed.port}"
    return urlunparse(parsed._replace(netloc=netloc))


def _run_git(*args: str, cwd: Path | None = None) -> None:
    subprocess.check_call(["git", *args], cwd=cwd or REPO_ROOT)


def clone_or_update(*, repo_url: str, branch: str) -> None:
    repo_url = _auth_repo_url(repo_url.strip())
    if not repo_url:
        raise SystemExit(
            "COLAB_REPO_URL is required.\n"
            "Set COLAB_REPO_URL in laptop .env (or pass --repo-url)."
        )

    if REPO_ROOT.is_dir() and (REPO_ROOT / ".git").is_dir():
        print(f"Updating existing repo at {REPO_ROOT}...")
        _run_git("fetch", "--all", "--prune")
        _run_git("checkout", branch)
        _run_git("pull", "--ff-only", "origin", branch)
        return

    if REPO_ROOT.exists():
        raise SystemExit(f"{REPO_ROOT} exists but is not a git repo. Remove it first.")

    print(f"Cloning {repo_url} -> {REPO_ROOT} (branch={branch})...")
    _run_git("clone", "--branch", branch, "--depth", "1", repo_url, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone video-retrieval on Colab")
    parser.add_argument(
        "--repo-url",
        default=_secret("COLAB_REPO_URL"),
        help="Git remote URL (or set COLAB_REPO_URL secret / env)",
    )
    parser.add_argument(
        "--branch",
        default=_secret("COLAB_REPO_BRANCH") or os.environ.get("COLAB_REPO_BRANCH", DEFAULT_BRANCH),
        help=f"Branch to checkout (default: {DEFAULT_BRANCH})",
    )
    args, _unknown = parser.parse_known_args()
    clone_or_update(repo_url=args.repo_url, branch=args.branch)
    print(f"Ready at {REPO_ROOT}")


if __name__ == "__main__":
    main()
