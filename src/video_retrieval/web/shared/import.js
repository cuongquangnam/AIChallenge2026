/** CSV import helpers for KIS / QA / TRAKE submission files. */

import { queryIdFromFilename } from "./export.js";

function parseCsvLine(line) {
  const cells = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (inQuotes) {
      if (ch === '"') {
        if (line[i + 1] === '"') {
          current += '"';
          i += 1;
        } else {
          inQuotes = false;
        }
      } else {
        current += ch;
      }
      continue;
    }
    if (ch === '"') {
      inQuotes = true;
      continue;
    }
    if (ch === ",") {
      cells.push(current.trim());
      current = "";
      continue;
    }
    current += ch;
  }
  cells.push(current.trim());
  return cells;
}

export function parseCsvRows(text) {
  const rows = [];
  for (const raw of String(text || "").split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) continue;
    rows.push(parseCsvLine(line));
  }
  return rows;
}

/** @returns {Array<{video_id: string, frame_id: number, time_sec: number|null, answer: string}>} */
export function parseQaCsv(text) {
  const parsed = parseCsvRows(text);
  const out = [];
  for (let i = 0; i < parsed.length; i += 1) {
    const cells = parsed[i];
    if (!cells.length) continue;
    if (
      i === 0 &&
      cells[0].toLowerCase() === "video_id" &&
      /frame/.test(String(cells[1] || "").toLowerCase())
    ) {
      continue;
    }
    const videoId = String(cells[0] || "").trim();
    const frameId = Number(cells[1]);
    if (!videoId || !Number.isFinite(frameId) || frameId < 0) continue;

    let timeSec = null;
    let answer = "";
    if (cells.length >= 4) {
      const maybeTime = Number(cells[2]);
      if (Number.isFinite(maybeTime)) {
        timeSec = maybeTime;
      }
      answer = cells.slice(3).join(",").trim();
    } else if (cells.length === 3) {
      const third = cells[2];
      if (/^-?\d+(\.\d+)?$/.test(third)) {
        timeSec = Number(third);
      } else {
        answer = third;
      }
    }
    out.push({
      video_id: videoId,
      frame_id: Math.trunc(frameId),
      time_sec: timeSec,
      answer,
    });
  }
  return out;
}

/** @returns {{video_id: string, frames: number[], times: number[]}} */
export function parseTrakeCsv(text) {
  const rows = parseCsvRows(text);
  if (!rows.length) {
    throw new Error("CSV file is empty.");
  }
  const cells = rows[0];
  const videoId = String(cells[0] || "").trim();
  const numbers = cells.slice(1).map((cell) => Number(cell));
  if (!videoId) {
    throw new Error("TRAKE CSV must start with video_id.");
  }
  if (!numbers.length || numbers.some((value) => !Number.isFinite(value))) {
    throw new Error("TRAKE CSV must list numeric frame indices (and optional times).");
  }

  let frames;
  let times;
  if (numbers.length % 2 === 0) {
    const half = numbers.length / 2;
    frames = numbers.slice(0, half).map((value) => Math.trunc(value));
    times = numbers.slice(half);
  } else {
    frames = numbers.map((value) => Math.trunc(value));
    times = [];
  }
  return { video_id: videoId, frames, times };
}

async function readResponseError(response, fallback) {
  let detail = fallback;
  try {
    const body = await response.json();
    if (body?.detail) {
      detail =
        typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    }
  } catch {
    try {
      const text = await response.text();
      if (text) detail = text;
    } catch {
      /* keep fallback */
    }
  }
  return detail;
}

export function taskKindFromQueryId(queryId) {
  const id = String(queryId || "").toLowerCase();
  if (id.endsWith("-kis")) return "kis";
  if (id.endsWith("-qa")) return "qa";
  if (id.endsWith("-trake")) return "trake";
  return null;
}

export async function fetchQueryText(queryId) {
  const qid = String(queryId || "").trim();
  if (!qid) return null;
  const response = await fetch(`/api/queries/${encodeURIComponent(qid)}`);
  if (!response.ok) return null;
  const payload = await response.json();
  return String(payload.text || "").trim() || null;
}

/** Prefer textarea value; fall back to server query catalog by id. */
export async function resolveQueryText(queryId, fieldValue) {
  const direct = String(fieldValue || "").trim();
  if (direct) return direct;
  return fetchQueryText(queryId);
}

export async function runKisImport({ query, limit, queryId = "" }) {
  const response = await fetch("/kis", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, limit, query_id: queryId || "" }),
  });
  if (!response.ok) {
    throw new Error(await readResponseError(response, `KIS failed (${response.status})`));
  }
  return response.json();
}

export async function runQaImport({ question, limit }) {
  const response = await fetch("/qa", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, limit }),
  });
  if (!response.ok) {
    throw new Error(await readResponseError(response, `QA failed (${response.status})`));
  }
  return response.json();
}

export async function runTrakeImport({ query, topChains }) {
  const response = await fetch("/trake", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_chains: topChains }),
  });
  if (!response.ok) {
    throw new Error(await readResponseError(response, `TRAKE failed (${response.status})`));
  }
  return response.json();
}

export function mergeQaImportHits(framePayload, qaRows) {
  return (framePayload.hits || []).map((hit, index) => {
    const row = qaRows[index] || qaRows[0];
    return {
      ...hit,
      frame_id: hit.frame_index,
      answer: row?.answer || "",
      timestamp_sec:
        row?.time_sec != null && Number.isFinite(row.time_sec)
          ? row.time_sec
          : hit.timestamp_sec,
      source: "csv_import",
    };
  });
}

export function chainFromImportedTrakeRow(parsed, framePayload, plan) {
  const hitByFrame = new Map(
    (framePayload.hits || []).map((hit) => [hit.frame_index, hit])
  );
  const videoUrl =
    framePayload.hits?.[0]?.video_url ||
    `/media/videos/${encodeURIComponent(parsed.video_id)}`;
  const planEvents = plan?.events || [];
  const events = parsed.frames.map((frameIndex, index) => {
    const hit = hitByFrame.get(frameIndex);
    const spec = planEvents[index];
    const timeSec =
      parsed.times[index] != null && Number.isFinite(parsed.times[index])
        ? parsed.times[index]
        : hit?.timestamp_sec;
    return {
      event_id: spec?.event_id || `E${index + 1}`,
      frame_index: frameIndex,
      score: hit?.score ?? 1.0,
      timestamp_sec: timeSec,
      image_url: hit?.image_url || hit?.image_data_url || "",
      video_url: hit?.video_url || videoUrl,
      source: "csv_import",
    };
  });
  return {
    video_id: parsed.video_id,
    video_url: videoUrl,
    score: 1,
    source: "csv_import",
    events,
  };
}

export function pickMatchingChain(chains, importedChain) {
  if (!importedChain?.video_id) return null;
  const frames = (importedChain.events || []).map((event) => event.frame_index);
  for (const chain of chains || []) {
    if (chain.video_id !== importedChain.video_id) continue;
    const chainFrames = (chain.events || []).map((event) => event.frame_index);
    if (
      chainFrames.length === frames.length &&
      chainFrames.every((frame, index) => frame === frames[index])
    ) {
      return chain;
    }
  }
  return null;
}

export async function resolveSubmissionFrames({ csvText, rows, queryId = "" }) {
  const body = { query_id: queryId || "" };
  if (rows?.length) {
    body.rows = rows.map((row) => ({
      video_id: row.video_id,
      frame_index: row.frame_index,
    }));
  } else if (csvText != null) {
    body.csv_text = csvText;
  } else {
    throw new Error("Provide csvText or rows.");
  }

  const response = await fetch("/api/submission/frames", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    let detail = `Import failed (${response.status})`;
    try {
      const errBody = await response.json();
      if (errBody?.detail) {
        detail =
          typeof errBody.detail === "string"
            ? errBody.detail
            : JSON.stringify(errBody.detail);
      }
    } catch {
      /* keep default */
    }
    throw new Error(detail);
  }
  return response.json();
}

export function openCsvFilePicker(inputEl) {
  if (!inputEl) return;
  inputEl.value = "";
  inputEl.click();
}

export async function importCsvFile(file, { queryIdFrom = "filename" } = {}) {
  if (!file) return null;
  const queryId =
    queryIdFrom === "filename" ? queryIdFromFilename(file.name) : String(queryIdFrom || "");
  const csvText = await file.text();
  if (!csvText.trim()) {
    throw new Error("CSV file is empty.");
  }
  const payload = await resolveSubmissionFrames({ csvText, queryId });
  return { payload, queryId, csvText };
}
