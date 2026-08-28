/** CSV export helpers for flat frame submissions. */

export const SUBMISSION_PAD_OFFSETS = [
  1, -1, 2, -2, 3, -3, 5, -5, 8, -8, 12, -12, 15, -15, 20, -20, 25, -25, 30, -30,
  40, -40, 50, -50,
];

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

export function hitsToSubmissionRows(hits, limit, padOffsets = SUBMISSION_PAD_OFFSETS) {
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

  if (limit == null) {
    return rows;
  }

  const seeds = [...rows];
  let offsetI = 0;
  while (rows.length < limit && seeds.length) {
    const offset = padOffsets[offsetI % padOffsets.length];
    const cycle = Math.floor(offsetI / padOffsets.length);
    const seed = seeds[cycle % seeds.length];
    const candidate = [seed[0], seed[1] + offset];
    offsetI += 1;
    if (candidate[1] < 0) continue;
    const key = `${candidate[0]}|${candidate[1]}`;
    if (seen.has(key)) {
      if (offsetI > limit * 200) break;
      continue;
    }
    seen.add(key);
    rows.push(candidate);
  }

  if (rows.length < limit && rows.length) {
    const last = rows[rows.length - 1];
    let nxt = last[1] + 1;
    while (rows.length < limit) {
      const key = `${last[0]}|${nxt}`;
      if (!seen.has(key)) {
        seen.add(key);
        rows.push([last[0], nxt]);
      }
      nxt += 1;
    }
  }

  return rows.slice(0, limit);
}
