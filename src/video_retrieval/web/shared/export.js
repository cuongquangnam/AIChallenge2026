/** CSV export helpers for flat frame submissions. */

export function queryIdFromFilename(name) {
  const base = String(name || "")
    .replace(/^.*[\\/]/, "")
    .replace(/\.csv$/i, "")
    .trim();
  return base.replace(/[^\w.-]+/g, "-") || "query";
}

export function downloadTextFile(text, filename, mime = "text/csv;charset=utf-8") {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
  return anchor.download;
}

export function hitsToSubmissionRows(hits, limit) {
  const rows = [];
  const seen = new Set();

  for (const hit of hits) {
    if (hit.frame_index == null || hit.frame_index === "") continue;
    const videoId = String(hit.video_id || "").trim();
    const frameIdx = Number(hit.frame_index);
    if (!videoId || !Number.isFinite(frameIdx) || frameIdx < 0) continue;
    const key = `${videoId}|${frameIdx}`;
    if (seen.has(key)) continue;
    seen.add(key);
    rows.push([videoId, Math.trunc(frameIdx)]);
    if (limit != null && rows.length >= limit) {
      return rows.slice(0, limit);
    }
  }

  return rows;
}

/** Flatten chains to export rows in chain rank + E1…En order (no padding). */
export function hitsFromTrakeChains(chains) {
  const hits = [];
  for (const chain of chains || []) {
    for (const event of chain.events || []) {
      hits.push({
        video_id: chain.video_id,
        frame_index: event.frame_index,
        frame_id: event.frame_index,
        timestamp_sec: event.timestamp_sec,
        image_url: event.image_url,
        video_url: event.video_url || chain.video_url,
        source: `trake:${event.event_id}`,
        score: chain.score,
      });
    }
  }
  return hits;
}

/** Flatten QA chain results to export rows in chain + event order. */
export function hitsFromQaResults(results, limit = 100) {
  const hits = [];
  const seen = new Set();
  for (const item of results || []) {
    const chain = item.chain || {};
    const answer = item.answer || "";
    for (const event of chain.events || []) {
      const key = `${chain.video_id}|${event.frame_index}`;
      if (seen.has(key)) continue;
      seen.add(key);
      hits.push({
        video_id: chain.video_id,
        frame_index: event.frame_index,
        frame_id: event.frame_index,
        timestamp_sec: event.timestamp_sec,
        image_url: event.image_url,
        video_url: event.video_url || chain.video_url,
        answer,
        source: `qa:${event.event_id}`,
        score: chain.score,
      });
      if (hits.length >= limit) {
        return hits;
      }
    }
  }
  return hits;
}
