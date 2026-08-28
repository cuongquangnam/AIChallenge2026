from __future__ import annotations

from collections import defaultdict

from video_retrieval.models import SearchHit


def score_videos(event_hits: dict[str, list[SearchHit]]) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    coverage: dict[str, set[str]] = defaultdict(set)
    for event_id, hits in event_hits.items():
        best: dict[str, float] = {}
        for rank, hit in enumerate(hits):
            if not hit.video_id:
                continue
            contribution = 1.0 / (60 + rank + 1)
            best[hit.video_id] = max(best.get(hit.video_id, 0.0), contribution)
            coverage[hit.video_id].add(event_id)
        for video_id, contribution in best.items():
            scores[video_id] += contribution
    for video_id, events in coverage.items():
        scores[video_id] += 0.5 * len(events)
    return dict(scores)


def best_monotonic_path(
    per_event_cands: list[list[tuple[int, float, SearchHit]]],
) -> list[tuple[int, float, SearchHit]] | None:
    """DP: maximize sum of scores with strictly increasing frame indices."""
    paths = top_monotonic_paths(per_event_cands, limit=1)
    return paths[0] if paths else None


def top_monotonic_paths(
    per_event_cands: list[list[tuple[int, float, SearchHit]]],
    *,
    limit: int = 1,
) -> list[list[tuple[int, float, SearchHit]]]:
    """Return up to ``limit`` distinct monotonic paths, best score first."""
    if limit <= 0:
        return []
    if not per_event_cands or any(not cands for cands in per_event_cands):
        return []

    dp, prev = _monotonic_dp(per_event_cands)
    if dp is None or prev is None:
        fallback = greedy_monotonic_path(per_event_cands)
        return [fallback] if fallback else []

    last_scores = dp[-1]
    ranked_ends = sorted(
        range(len(last_scores)),
        key=lambda idx: last_scores[idx],
        reverse=True,
    )

    paths: list[list[tuple[int, float, SearchHit]]] = []
    seen: set[tuple[int, ...]] = set()
    for end_idx in ranked_ends:
        if last_scores[end_idx] == float("-inf"):
            continue
        path = _backtrack_monotonic_path(per_event_cands, prev, end_idx)
        if path is None:
            continue
        key = tuple(frame for frame, _, _ in path)
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)
        if len(paths) >= limit:
            break

    if not paths:
        fallback = greedy_monotonic_path(per_event_cands)
        if fallback:
            paths.append(fallback)
    return paths


def _monotonic_dp(
    per_event_cands: list[list[tuple[int, float, SearchHit]]],
) -> tuple[list[list[float]], list[list[int | None]]] | tuple[None, None]:
    n = len(per_event_cands)
    dp: list[list[float]] = []
    prev: list[list[int | None]] = []

    first = per_event_cands[0]
    dp.append([score for _, score, _ in first])
    prev.append([None] * len(first))

    for i in range(1, n):
        cur = per_event_cands[i]
        dp_row: list[float] = []
        prev_row: list[int | None] = []
        for j, (frame_j, score_j, _) in enumerate(cur):
            best_score = float("-inf")
            best_k: int | None = None
            for k, (frame_k, _, _) in enumerate(per_event_cands[i - 1]):
                if frame_k >= frame_j:
                    continue
                candidate = dp[i - 1][k] + score_j
                if candidate > best_score:
                    best_score = candidate
                    best_k = k
            dp_row.append(best_score)
            prev_row.append(best_k)
        dp.append(dp_row)
        prev.append(prev_row)

    return dp, prev


def _backtrack_monotonic_path(
    per_event_cands: list[list[tuple[int, float, SearchHit]]],
    prev: list[list[int | None]],
    end_idx: int,
) -> list[tuple[int, float, SearchHit]] | None:
    n = len(per_event_cands)
    j = end_idx
    path_rev: list[tuple[int, float, SearchHit]] = []
    for i in range(n - 1, -1, -1):
        if j < 0 or j >= len(per_event_cands[i]):
            return None
        path_rev.append(per_event_cands[i][j])
        parent = prev[i][j]
        if i == 0:
            break
        if parent is None:
            return None
        j = parent
    path_rev.reverse()
    return path_rev


def greedy_monotonic_path(
    per_event_cands: list[list[tuple[int, float, SearchHit]]],
) -> list[tuple[int, float, SearchHit]] | None:
    if not per_event_cands or any(not cands for cands in per_event_cands):
        return None
    path: list[tuple[int, float, SearchHit]] = []
    prev_frame = -1
    for cands in per_event_cands:
        options = [item for item in cands if item[0] > prev_frame]
        if not options:
            return None
        chosen = max(options, key=lambda item: item[1])
        path.append(chosen)
        prev_frame = chosen[0]
    return path
