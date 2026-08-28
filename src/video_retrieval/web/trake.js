import {
  eventSpecByIdFromPlan,
  renderChainCards,
  renderEventPlan,
} from "./chains.js";
import { downloadTextFile } from "./shared/export.js";
import { joinMeta, sanitizeQueryId } from "./shared/format.js";
import { createLightboxController } from "./shared/lightbox.js";
import { createStatusController } from "./shared/status.js";

const form = document.getElementById("trake-form");
const queryEl = document.getElementById("query");
const queryIdEl = document.getElementById("query-id");
const topChainsEl = document.getElementById("top-chains");
const submitBtn = document.getElementById("submit-btn");
const exportBtn = document.getElementById("export-btn");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const resultsMetaEl = document.getElementById("results-meta");
const planEl = document.getElementById("plan");
const planBodyEl = document.getElementById("plan-body");
const editorEl = document.getElementById("editor");
const editVideoEl = document.getElementById("edit-video");
const editEventsEl = document.getElementById("edit-events");
const chainsEl = document.getElementById("chains");

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

function exportCsv() {
  const lines = [];
  const videoId = (editVideoEl.value || "").trim();
  if (videoId && selectedEvents.length) {
    const frames = selectedEvents.map((event) => Number(event.frame_index));
    if (
      frames.every((f) => Number.isFinite(f) && f >= 0) &&
      frames.every((f, i) => i === 0 || f > frames[i - 1])
    ) {
      lines.push(`${videoId},${frames.map((f) => Math.trunc(f)).join(",")}`);
    }
  }
  if (lastChains?.length) {
    for (const chain of lastChains) {
      const row = `${chain.video_id},${(chain.events || [])
        .map((event) => event.frame_index)
        .join(",")}`;
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
    resultsEl.hidden = false;
    resultsMetaEl.textContent = joinMeta([
      `${(payload.chains || []).length} chains`,
      payload.csv_row ? `best ${payload.csv_row}` : null,
    ]);
    renderEventPlan(planBodyEl, planEl, payload.plan);
    renderChains(payload);
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
