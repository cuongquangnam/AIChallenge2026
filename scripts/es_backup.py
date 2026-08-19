#!/usr/bin/env python3
"""Dump or restore the Elasticsearch text index as mapping + NDJSON."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MAPPING = {
    "mappings": {
        "properties": {
            "video_id": {"type": "keyword"},
            "source": {"type": "keyword"},
            "text": {"type": "text"},
            "shot_index": {"type": "integer"},
            "frame_index": {"type": "integer"},
            "role": {"type": "keyword"},
            "start_sec": {"type": "float"},
            "end_sec": {"type": "float"},
            "keyframe_path": {"type": "keyword"},
        }
    }
}


def _request(
    url: str,
    *,
    method: str = "GET",
    body: object | None = None,
    content_type: str = "application/json",
    allow_missing: bool = False,
) -> object:
    data = None
    headers = {}
    if body is not None:
        if isinstance(body, (bytes, bytearray)):
            data = body
        else:
            data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            payload = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if allow_missing and exc.code in {404}:
            return {}
        raise SystemExit(f"Elasticsearch {method} {url} failed ({exc.code}): {detail}") from exc
    return json.loads(payload) if payload else {}


def dump_index(es_url: str, index: str, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    exists = _request(f"{es_url}/{index}")
    if isinstance(exists, dict) and exists.get("error"):
        raise SystemExit(f"Index {index!r} was not found at {es_url}")

    mapping = _request(f"{es_url}/{index}/_mapping")
    (out_dir / f"{index}.mapping.json").write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    search = _request(
        f"{es_url}/{index}/_search?scroll=2m",
        method="POST",
        body={"size": 1000, "query": {"match_all": {}}, "sort": ["_doc"]},
    )
    ndjson = out_dir / f"{index}.ndjson"
    count = 0
    with ndjson.open("w", encoding="utf-8") as handle:
        while True:
            hits = ((search.get("hits") or {}).get("hits") or []) if isinstance(search, dict) else []
            if not hits:
                break
            for hit in hits:
                handle.write(
                    json.dumps(
                        {"_id": hit.get("_id"), "_source": hit.get("_source") or {}},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                count += 1
            scroll_id = search.get("_scroll_id")
            if not scroll_id:
                break
            search = _request(
                f"{es_url}/_search/scroll",
                method="POST",
                body={"scroll": "2m", "scroll_id": scroll_id},
            )
    print(f"exported {count} docs → {ndjson}")
    return count


def _index_body(mapping_path: Path, index: str) -> dict:
    if not mapping_path.exists():
        return DEFAULT_MAPPING
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    if index in payload and isinstance(payload[index], dict) and "mappings" in payload[index]:
        return {"mappings": payload[index]["mappings"]}
    if "mappings" in payload:
        return {"mappings": payload["mappings"]}
    return DEFAULT_MAPPING


def restore_index(es_url: str, index: str, source_dir: Path) -> int:
    mapping_path = source_dir / f"{index}.mapping.json"
    ndjson_path = source_dir / f"{index}.ndjson"
    if not ndjson_path.exists():
        raise SystemExit(f"Missing {ndjson_path}")

    _request(f"{es_url}/{index}", method="DELETE", allow_missing=True)
    _request(f"{es_url}/{index}", method="PUT", body=_index_body(mapping_path, index))

    count = 0
    batch: list[str] = []

    def flush() -> None:
        nonlocal batch, count
        if not batch:
            return
        payload = ("\n".join(batch) + "\n").encode("utf-8")
        result = _request(
            f"{es_url}/_bulk?refresh=true",
            method="POST",
            body=payload,
            content_type="application/x-ndjson",
        )
        if isinstance(result, dict) and result.get("errors"):
            raise SystemExit(f"Elasticsearch bulk restore reported errors: {result}")
        count += sum(1 for line in batch if '"index"' in line[:40])
        batch = []

    for line in ndjson_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        doc_id = row.get("_id")
        source = row.get("_source") or {}
        batch.append(json.dumps({"index": {"_index": index, "_id": doc_id}}))
        batch.append(json.dumps(source, ensure_ascii=False))
        if len(batch) >= 1000:
            flush()
    flush()
    print(f"restored {count} docs → {index}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("dump", "restore"))
    parser.add_argument("--es-url", default="http://localhost:9200")
    parser.add_argument("--index", default="video_text")
    parser.add_argument("--dir", required=True, help="Directory for mapping.json + ndjson")
    args = parser.parse_args()
    es_url = args.es_url.rstrip("/")
    out_dir = Path(args.dir)
    if args.action == "dump":
        dump_index(es_url, args.index, out_dir)
    else:
        restore_index(es_url, args.index, out_dir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
