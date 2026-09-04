import {
  eventSpecByIdFromPlan,
  renderChainCards,
  renderEventPlan,
} from "./chains.js";
import { downloadTextFile, hitsFromTrakeChains, queryIdFromFilename } from "./shared/export.js";
import {
  chainFromImportedTrakeRow,
  fetchQueryText,
  openCsvFilePicker,
  parseTrakeCsv,
  resolveSubmissionFrames,
} from "./shared/import.js";
import { joinMeta, sanitizeQueryId } from "./shared/format.js";
import { createLightboxController } from "./shared/lightbox.js";
import { createStatusController } from "./shared/status.js";

const form = document.getElementById("trake-form");
const queryEl = document.getElementById("query");
const queryIdEl = document.getElementById("query-id");
const topChainsEl = document.getElementById("top-chains");
const submitBtn = document.getElementById("submit-btn");
const exportBtn = document.getElementById("export-btn");
const importBtn = document.getElementById("import-btn");
const importCsvEl = document.getElementById("import-csv");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const resultsMetaEl = document.getElementById("results-meta");
const planEl = document.getElementById("plan");
const planBodyEl = document.getElementById("plan-body");
const editorEl = document.getElementById("editor");
const editVideoEl = document.getElementById("edit-video");
const editEventsEl = document.getElementById("edit-events");
const chainsEl = document.getElementById("chains");
const exportGridEl = document.getElementById("export-grid");

const status = createStatusController(statusEl);
const lightbox = createLightboxController({
  dialog: document.getElementById("lightbox"),
  img: document.getElementById("lightbox-img"),
  video: document.getElementById("lightbox-video"),
  meta: document.getElementById("lightbox-meta"),
  closeBtn: document.getElementById("lightbox-close"),
  openVideoBtn: document.getElementById("lightbox-open-video"),
});
lightbox.bind();

/** @type {Array<{event_id: string, frame_index: number, image_url?: string, video_url?: string, timestamp_sec?: number}>} */
let selectedEvents = [];
/** @type {Array<Record<string, unknown>>} */
let lastChains = [];
let lastPlan = null;

function syncEditorFromSelection() {
  editorEl.hidden = selectedEvents.length === 0;
  editEventsEl.replaceChildren();
  for (const event of selectedEvents) {
    const wrap = document.createElement("label");
    wrap.className = "field export-field";
    const span = document.createElement("span");
    span.textContent = event.event_id;
    const input = document.createElement("input");
    input.type = "number";
    input.min = "0";
    input.step = "1";
    input.value = String(event.frame_index);
    input.dataset.eventId = event.event_id;
    input.addEventListener("change", () => {
      const value = Number(input.value);
      if (!Number.isFinite(value) || value < 0) return;
      event.frame_index = Math.trunc(value);
    });
    wrap.append(span, input);
    editEventsEl.appendChild(wrap);
  }
}

function selectChain(chain) {
  editVideoEl.value = chain.video_id || "";
  selectedEvents = (chain.events || []).map((event) => ({
    ...event,
    video_id: chain.video_id,
  }));
  syncEditorFromSelection();
}

function renderExportGrid(hits) {
  exportGridEl.replaceChildren();
  (hits || []).forEach((hit, index) => {
    const card = document.createElement("article");
    card.className = "card";
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "thumb-btn";
    const img = document.createElement("img");
    img.alt = `${hit.video_id} ${hit.source || ""}`;
    img.src = hit.image_url || hit.image_data_url || "";
    btn.appendChild(img);
    btn.addEventListener("click", () => lightbox.open(hit));
    const body = document.createElement("div");
    body.className = "card-body";
    const title = document.createElement("h3");
    title.textContent = `#${index + 1} · ${hit.video_id || "unknown"}`;
    const detail = document.createElement("p");
    detail.textContent = joinMeta([
      hit.source,
      hit.frame_index != null ? `f${hit.frame_index}` : null,
      hit.timestamp_sec != null ? `t${hit.timestamp_sec}` : null,
    ]);
    body.append(title, detail);
    card.append(btn, body);
    exportGridEl.appendChild(card);
  });
}

function renderTrakeResults(payload) {
  resultsEl.hidden = false;
  renderEventPlan(planBodyEl, planEl, payload.plan);
  renderChains(payload);
  const exportHits =
    (payload.hits || []).length > 0
      ? payload.hits
      : hitsFromTrakeChains(payload.chains || lastChains);
  renderExportGrid(exportHits);
}

function renderChains(payload) {
  lastPlan = payload.plan || null;
  lastChains = payload.chains || [];
  renderChainCards(chainsEl, lastChains, {
    layout: "timeline",
    eventSpecById: eventSpecByIdFromPlan(lastPlan),
    showEventChannels: true,
    onEventClick: (hit) => lightbox.open(hit),
    onChainAction: (chain) => {
      selectChain(chain);
      status.set(`Selected chain ${chain.video_id}`);
    },
    chainActionLabel: "Use this chain",
  });
}

function formatTrakeRow(videoId, events) {
  const frames = events.map((event) => Number(event.frame_index));
  const times = events.map((event) => event.timestamp_sec);
  const hasTimes = times.every((value) => value != null && Number.isFinite(Number(value)));
  if (hasTimes) {
    return `${videoId},${frames.map((f) => Math.trunc(f)).join(",")},${times
      .map((value) => Number(value))
      .join(",")}`;
  }
  return `${videoId},${frames.map((f) => Math.trunc(f)).join(",")}`;
}

function exportCsv() {
  const lines = [];
  const videoId = (editVideoEl.value || "").trim();
  if (videoId && selectedEvents.length) {
    const frames = selectedEvents.map((event) => Number(event.frame_index));
    if (
      frames.every((f) => Number.isFinite(f) && f >= 0) &&
      frames.every((f, i) => i === 0 || f > frames[i - 1])
    ) {
      lines.push(formatTrakeRow(videoId, selectedEvents));
    }
  }
  if (lastChains?.length) {
    for (const chain of lastChains) {
      const row = formatTrakeRow(
        chain.video_id,
        (chain.events || []).map((event) => ({
          frame_index: event.frame_index,
          timestamp_sec: event.timestamp_sec,
        }))
      );
      if (!lines.includes(row)) lines.push(row);
    }
  }
  if (!lines.length) {
    status.set("Select or run a chain before exporting.", true);
    return;
  }
  const csv = lines.join("\n") + "\n";
  const queryId = sanitizeQueryId(queryIdEl.value);
  const filename = downloadTextFile(csv, `${queryId}.csv`);
  status.set(`Exported ${lines.length} chain rows → ${filename}`);
}

function setImportBusy(busy) {
  for (const btn of [importBtn, exportBtn, submitBtn]) {
    if (btn) btn.disabled = busy;
  }
  if (importBtn) {
    importBtn.textContent = busy ? "Importing…" : "Import CSV";
  }
}

async function importSubmissionCsv(file) {
  if (!file) return;
  setImportBusy(true);
  status.set(`Importing ${file.name}…`);
  try {
    const csvText = await file.text();
    const parsed = parseTrakeCsv(csvText);
    const queryId = queryIdFromFilename(file.name);
    const query = await fetchQueryText(queryId);
    if (query && !queryEl.value.trim()) {
      queryEl.value = query;
    }

    const rows = parsed.frames.map((frameIndex) => ({
      video_id: parsed.video_id,
      frame_index: frameIndex,
    }));

    const framePayload = await resolveSubmissionFrames({ rows, queryId });
    const importedChain = chainFromImportedTrakeRow(parsed, framePayload, null);

    queryIdEl.value = queryId;
    lastPlan = null;
    lastChains = [];
    resultsEl.hidden = false;
    planEl.hidden = true;
    chainsEl.replaceChildren();
    resultsMetaEl.textContent = joinMeta([
      `${importedChain.events?.length || 0} imported events`,
      importedChain.video_id ? `video ${importedChain.video_id}` : null,
    ]);
    renderExportGrid(framePayload.hits || []);
    if (importedChain.events?.length) {
      selectChain(importedChain);
    } else {
      editorEl.hidden = true;
      selectedEvents = [];
    }

    const errCount = (framePayload.errors || []).length;
    status.set(
      errCount
        ? `Imported CSV (${errCount} frames missing).`
        : `Imported CSV from ${file.name}.`
    );
  } catch (error) {
    status.set(error instanceof Error ? error.message : String(error), true);
  } finally {
    setImportBusy(false);
    importCsvEl.value = "";
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = queryEl.value.trim();
  if (!query) {
    status.set("Paste a TRAKE query with E1…En events.", true);
    return;
  }
  submitBtn.disabled = true;
  status.set("Running TRAKE…");
  resultsEl.hidden = true;
  try {
    const response = await fetch("/trake", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        top_chains: Number(topChainsEl.value),
      }),
    });
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(detail || `TRAKE failed (${response.status})`);
    }
    const payload = await response.json();
    resultsMetaEl.textContent = joinMeta([
      `${(payload.chains || []).length} chains`,
      `${(payload.hits || []).length} export rows`,
      payload.csv_row ? `best ${payload.csv_row}` : null,
    ]);
    renderTrakeResults(payload);
    if ((payload.chains || []).length) {
      selectChain(payload.chains[0]);
    } else {
      editorEl.hidden = true;
      selectedEvents = [];
    }
    status.set("TRAKE complete — review chains, edit frames if needed, export.");
  } catch (error) {
    resultsEl.hidden = true;
    status.set(error instanceof Error ? error.message : String(error), true);
  } finally {
    submitBtn.disabled = false;
  }
});

exportBtn.addEventListener("click", exportCsv);
importBtn.addEventListener("click", () => openCsvFilePicker(importCsvEl));
importCsvEl.addEventListener("change", () => {
  const file = importCsvEl.files?.[0];
  if (file) {
    importSubmissionCsv(file);
  }
});
