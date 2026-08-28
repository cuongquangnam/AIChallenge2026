import {
  eventSpecByIdFromPlan,
  renderChainCards,
  renderEventPlan,
} from "./chains.js";
import {
  downloadTextFile,
  hitsToSubmissionRows,
  queryIdFromFilename,
} from "./shared/export.js";
import {
  formatScore,
  formatTime,
  joinMeta,
  sanitizeQueryId,
} from "./shared/format.js";
import { createLightboxController } from "./shared/lightbox.js";
import { createStatusController } from "./shared/status.js";

const form = document.getElementById("search-form");
const queryEl = document.getElementById("query");
const limitEl = document.getElementById("limit");
const submitBtn = document.getElementById("submit-btn");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const resultsMetaEl = document.getElementById("results-meta");
const planEl = document.getElementById("plan");
const planBodyEl = document.getElementById("plan-body");
const chainsEl = document.getElementById("chains");
const gridEl = document.getElementById("grid");
const lightboxEl = document.getElementById("lightbox");
const lightboxImg = document.getElementById("lightbox-img");
const lightboxVideo = document.getElementById("lightbox-video");
const lightboxMeta = document.getElementById("lightbox-meta");
const lightboxClose = document.getElementById("lightbox-close");
const lightboxOpenVideo = document.getElementById("lightbox-open-video");
const lightboxAddFrame = document.getElementById("lightbox-add-frame");
const exportQueryIdEl = document.getElementById("export-query-id");
const exportLimitEl = document.getElementById("export-limit");
const exportBtn = document.getElementById("export-btn");
const importBtn = document.getElementById("import-btn");
const importResultsBtn = document.getElementById("import-results-btn");
const importCsvEl = document.getElementById("import-csv");
const moveFromEl = document.getElementById("move-from");
const moveToEl = document.getElementById("move-to");
const moveBtn = document.getElementById("move-btn");

const status = createStatusController(statusEl);
const lightbox = createLightboxController({
  dialog: lightboxEl,
  img: lightboxImg,
  video: lightboxVideo,
  meta: lightboxMeta,
  closeBtn: lightboxClose,
  openVideoBtn: lightboxOpenVideo,
  addFrameBtn: lightboxAddFrame,
});
lightbox.setOnVideoActivity(() => {
  const hit = lightbox.getActiveHit();
  const videoVisible = !lightboxVideo.hidden && Boolean(lightboxVideo.src);
  lightboxAddFrame.hidden = !videoVisible || !hit?.video_id;
});
lightbox.bind();

let lastPlan = null;
/** @type {Array<Record<string, unknown>>} */
let lastChains = [];
/** @type {Array<Record<string, unknown>>} */
let resultHits = [];
let hitSeq = 0;

function exportCurrentList() {
  if (!resultHits.length) {
    status.set("No frames to export.", true);
    return;
  }

  const queryId = sanitizeQueryId(exportQueryIdEl.value);
  const limitChoice = exportLimitEl.value;
  const limit = limitChoice === "exact" ? null : Number(limitChoice);
  const rows = hitsToSubmissionRows(resultHits, limit);

  if (!rows.length) {
    status.set("Export needs frames with a valid frame index.", true);
    return;
  }
  if (limit != null && rows.length < limit) {
    status.set(`Could only build ${rows.length}/${limit} rows.`, true);
    return;
  }

  const csv = rows.map(([videoId, frameIdx]) => `${videoId},${frameIdx}`).join("\n") + "\n";
  const filename = downloadTextFile(csv, `${queryId}.csv`);
  status.set(`Exported ${rows.length} rows → ${filename} (video_id,frame_idx).`);
}

function setImportBusy(busy) {
  for (const btn of [importBtn, importResultsBtn, exportBtn, submitBtn]) {
    if (!btn) continue;
    btn.disabled = busy;
  }
  if (importBtn) {
    importBtn.textContent = busy ? "Importing…" : "Import CSV";
  }
  if (importResultsBtn) {
    importResultsBtn.textContent = busy ? "Importing…" : "Import CSV";
  }
}

function openImportPicker() {
  importCsvEl.value = "";
  importCsvEl.click();
}

async function importSubmissionCsv(file) {
  if (!file) return;
  const queryId = queryIdFromFilename(file.name);
  setImportBusy(true);
  status.set(`Importing ${file.name}…`);
  try {
    const csvText = await file.text();
    if (!csvText.trim()) {
      throw new Error("CSV file is empty.");
    }
    const response = await fetch("/api/submission/frames", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ csv_text: csvText, query_id: queryId }),
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
    const payload = await response.json();
    exportQueryIdEl.value = queryId;
    renderHits(payload);
    const errCount = (payload.errors || []).length;
    const resolved = payload.resolved ?? (payload.hits || []).length;
    const total = payload.total_rows ?? resolved;
    if (!resolved) {
      status.set(
        errCount
          ? `Imported 0/${total} frames from ${file.name} (${errCount} errors).`
          : `No frames could be resolved from ${file.name}.`,
        true
      );
      return;
    }
    status.set(
      errCount
        ? `Imported ${resolved}/${total} frames from ${file.name} (${errCount} missing).`
        : `Imported ${resolved} frames from ${file.name}.`
    );
  } catch (error) {
    status.set(error instanceof Error ? error.message : String(error), true);
  } finally {
    setImportBusy(false);
    importCsvEl.value = "";
  }
}

function withHitId(hit) {
  return {
    ...hit,
    _uid: hit._uid || `hit-${++hitSeq}`,
  };
}

function openKisLightbox(hit) {
  lightbox.open(hit, [
    hit.video_id,
    hit.frame_index != null ? `frame ${hit.frame_index}` : null,
    hit.shot_index != null ? `shot ${hit.shot_index}` : null,
    hit.role ? `role ${hit.role}` : null,
    hit.source === "user_capture" ? "added by you" : null,
    formatTime(hit.timestamp_sec) ? `t=${formatTime(hit.timestamp_sec)}` : null,
    `score ${formatScore(hit.score)}`,
  ]);
}

function playFromMoment(hit) {
  if (!lightbox.playAtTime(hit)) {
    status.set("Video file not found for this result.", true);
  }
}

function renderPlan(plan) {
  renderEventPlan(planBodyEl, planEl, plan);
}

function renderChains(chains) {
  lastChains = chains || [];
  renderChainCards(chainsEl, lastChains, {
    onEventClick: (hit) => openKisLightbox(hit),
    eventSpecById: eventSpecByIdFromPlan(lastPlan),
    showEventChannels: true,
  });
}

function updateResultsMeta() {
  const added = resultHits.filter((hit) => hit.source === "user_capture").length;
  const fromCsv = resultHits.filter((hit) => hit.source === "csv_import").length;
  const eventCount = (lastPlan?.events || []).length;
  resultsMetaEl.textContent = joinMeta([
    `${resultHits.length} export frames`,
    lastChains.length ? `${lastChains.length} chains` : null,
    eventCount ? `${eventCount} events` : null,
    fromCsv ? `${fromCsv} from csv` : null,
    added ? `${added} added` : null,
    "drag or use Move frame #",
  ]);
}

function reorderHits(fromUid, toUid) {
  if (!fromUid || !toUid || fromUid === toUid) return;
  const fromIndex = resultHits.findIndex((hit) => hit._uid === fromUid);
  const toIndex = resultHits.findIndex((hit) => hit._uid === toUid);
  if (fromIndex < 0 || toIndex < 0) return;
  moveHitAtIndex(fromIndex, toIndex);
}

function moveHitAtIndex(fromIndex, toIndex) {
  if (fromIndex < 0 || toIndex < 0 || fromIndex === toIndex) return;
  if (fromIndex >= resultHits.length || toIndex >= resultHits.length) return;
  const [moved] = resultHits.splice(fromIndex, 1);
  resultHits.splice(toIndex, 0, moved);
  renderResultGrid();
  status.set(
    `Moved #${fromIndex + 1} → #${toIndex + 1}: ${moved.video_id}, f${moved.frame_index ?? "?"}`
  );
}

function moveHitByPosition() {
  if (!resultHits.length) {
    status.set("No frames to move.", true);
    return;
  }
  const fromPos = Number(moveFromEl.value);
  const toPos = Number(moveToEl.value);
  if (!Number.isInteger(fromPos) || !Number.isInteger(toPos)) {
    status.set("Enter whole numbers for both positions.", true);
    return;
  }
  const max = resultHits.length;
  if (fromPos < 1 || fromPos > max || toPos < 1 || toPos > max) {
    status.set(`Positions must be between 1 and ${max}.`, true);
    return;
  }
  if (fromPos === toPos) {
    status.set(`Frame #${fromPos} is already at that position.`);
    return;
  }
  moveHitAtIndex(fromPos - 1, toPos - 1);
  moveFromEl.value = "";
  moveToEl.value = "";
  moveFromEl.focus();
}

function removeHit(uid) {
  resultHits = resultHits.filter((hit) => hit._uid !== uid);
  if (lightbox.getActiveHit()?._uid === uid) {
    lightbox.close();
  }
  renderResultGrid();
  if (!resultHits.length) {
    status.set("All frames removed. Run a new search to refill results.");
  } else {
    status.set(`Removed a frame · ${resultHits.length} remaining.`);
  }
}

function addCapturedHit(hit) {
  const next = withHitId(hit);
  const duplicate = resultHits.some(
    (item) =>
      item.video_id === next.video_id &&
      item.frame_index != null &&
      item.frame_index === next.frame_index
  );
  if (duplicate) {
    status.set(
      `Frame ${next.video_id}, ${next.frame_index} is already in results.`,
      true
    );
    return;
  }
  resultHits = [next, ...resultHits];
  renderResultGrid();
  status.set(
    `Added ${next.video_id}, frame ${next.frame_index} @ ${formatTime(next.timestamp_sec)}.`
  );
}

async function captureCurrentFrame() {
  const activeHit = lightbox.getActiveHit();
  if (!activeHit?.video_id || lightboxVideo.hidden) {
    status.set("Open the video first, then add the current frame.", true);
    return;
  }
  const t = Number.isFinite(lightboxVideo.currentTime) ? lightboxVideo.currentTime : 0;
  lightboxAddFrame.disabled = true;
  lightboxAddFrame.textContent = "Adding…";
  try {
    const response = await fetch(
      `/media/videos/${encodeURIComponent(activeHit.video_id)}/capture`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ t }),
      }
    );
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(detail || `Capture failed (${response.status})`);
    }
    const captured = await response.json();
    addCapturedHit({
      ...captured,
      image_url: captured.image_data_url,
      role: "capture",
    });
  } catch (error) {
    status.set(error instanceof Error ? error.message : String(error), true);
  } finally {
    lightboxAddFrame.disabled = false;
    lightboxAddFrame.textContent = "Add current frame";
  }
}

function renderResultGrid() {
  gridEl.replaceChildren();
  if (!resultHits.length) {
    resultsEl.hidden = !lastPlan;
    updateResultsMeta();
    return;
  }

  resultsEl.hidden = false;
  updateResultsMeta();

  for (const [index, hit] of resultHits.entries()) {
    const card = document.createElement("article");
    card.className = "card";
    card.dataset.uid = hit._uid;
    if (hit.source === "user_capture") {
      card.classList.add("card-added");
    }

    const handle = document.createElement("button");
    handle.type = "button";
    handle.className = "drag-handle";
    handle.title = "Drag to reorder";
    handle.setAttribute("aria-label", `Drag to reorder frame ${index + 1}`);
    handle.textContent = `⋮⋮  ${index + 1}`;
    handle.draggable = true;
    handle.addEventListener("click", (event) => event.preventDefault());
    handle.addEventListener("dragstart", (event) => {
      event.dataTransfer.effectAllowed = "move";
      event.dataTransfer.setData("text/plain", hit._uid);
      card.classList.add("dragging");
    });
    handle.addEventListener("dragend", () => {
      card.classList.remove("dragging");
      for (const el of gridEl.querySelectorAll(".drag-over")) {
        el.classList.remove("drag-over");
      }
    });
    card.appendChild(handle);

    card.addEventListener("dragover", (event) => {
      event.preventDefault();
      event.dataTransfer.dropEffect = "move";
      card.classList.add("drag-over");
    });
    card.addEventListener("dragleave", (event) => {
      if (!card.contains(event.relatedTarget)) {
        card.classList.remove("drag-over");
      }
    });
    card.addEventListener("drop", (event) => {
      event.preventDefault();
      card.classList.remove("drag-over");
      const fromUid = event.dataTransfer.getData("text/plain");
      reorderHits(fromUid, hit._uid);
    });

    const openBtn = document.createElement("button");
    openBtn.type = "button";
    openBtn.className = "card-main";
    openBtn.addEventListener("click", () => openKisLightbox(hit));

    const thumbSrc = hit.image_url || hit.image_data_url;
    if (thumbSrc) {
      const img = document.createElement("img");
      img.className = "thumb";
      img.loading = "lazy";
      img.draggable = false;
      img.src = thumbSrc;
      img.alt = `${hit.video_id} keyframe`;
      img.onerror = () => {
        img.replaceWith(
          Object.assign(document.createElement("div"), {
            className: "thumb missing",
            textContent: "Image missing",
          })
        );
      };
      openBtn.appendChild(img);
    } else {
      const missing = document.createElement("div");
      missing.className = "thumb missing";
      missing.textContent = "No keyframe path";
      openBtn.appendChild(missing);
    }

    const meta = document.createElement("div");
    meta.className = "meta";
    const title = document.createElement("strong");
    title.textContent = hit.video_id || "unknown";
    const detail = document.createElement("span");
    const time = formatTime(hit.timestamp_sec);
    detail.textContent = [
      hit.frame_index != null ? `f${hit.frame_index}` : null,
      hit.shot_index != null ? `shot ${hit.shot_index}` : null,
      hit.role || null,
      hit.source === "user_capture" ? "added" : null,
      hit.source === "csv_import" ? "csv" : null,
      time,
      hit.source === "csv_import" ? null : `score ${formatScore(hit.score)}`,
    ]
      .filter(Boolean)
      .join(" · ");
    meta.append(title, detail);
    openBtn.appendChild(meta);
    card.appendChild(openBtn);

    const actions = document.createElement("div");
    actions.className = "card-actions";

    const playBtn = document.createElement("button");
    playBtn.type = "button";
    playBtn.className = "play-btn";
    if (hit.video_url) {
      playBtn.textContent = `Watch @ ${time || "0:00"}`;
      playBtn.addEventListener("click", (event) => {
        event.stopPropagation();
        openKisLightbox(hit);
        playFromMoment(hit);
      });
    } else {
      playBtn.textContent = "Video missing";
      playBtn.disabled = true;
    }

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "remove-btn";
    removeBtn.textContent = "Remove";
    removeBtn.addEventListener("click", (event) => {
      event.stopPropagation();
      removeHit(hit._uid);
    });

    actions.append(playBtn, removeBtn);
    card.appendChild(actions);
    gridEl.appendChild(card);
  }
}

function renderHits(payload) {
  lastPlan = payload.plan || null;
  lastChains = payload.chains || [];
  resultHits = (payload.hits || []).map(withHitId);

  if (!resultHits.length && !lastChains.length) {
    resultsEl.hidden = true;
    planEl.hidden = true;
    if ((payload.mode || "") !== "csv") {
      status.set("No keyframes matched this query.");
    }
    return;
  }

  resultsEl.hidden = false;
  renderPlan(lastPlan);
  renderChains(lastChains);
  renderResultGrid();
  if ((payload.mode || "") !== "csv") {
    const bits = [];
    if (lastChains.length) bits.push(`${lastChains.length} chains`);
    if (resultHits.length) bits.push(`${resultHits.length} frames`);
    status.set(bits.join(" · ") || "KIS complete.");
  }
}

lightboxAddFrame.addEventListener("click", () => {
  captureCurrentFrame();
});

exportBtn.addEventListener("click", exportCurrentList);
importBtn.addEventListener("click", openImportPicker);
if (importResultsBtn) {
  importResultsBtn.addEventListener("click", openImportPicker);
}
importCsvEl.addEventListener("change", () => {
  const file = importCsvEl.files?.[0];
  if (file) {
    importSubmissionCsv(file);
  }
});
moveBtn.addEventListener("click", moveHitByPosition);
for (const el of [moveFromEl, moveToEl]) {
  el.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      moveHitByPosition();
    }
  });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = queryEl.value.trim();
  if (!query) {
    status.set("Enter a question or scene description.", true);
    return;
  }

  submitBtn.disabled = true;
  status.set("Running KIS…");
  resultsEl.hidden = true;
  gridEl.replaceChildren();

  try {
    const response = await fetch("/kis", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        limit: Number(limitEl.value),
        query_id: (exportQueryIdEl.value || "").trim(),
      }),
    });
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(detail || `KIS search failed (${response.status})`);
    }
    const payload = await response.json();
    if (payload.query_id) {
      exportQueryIdEl.value = payload.query_id;
    }
    renderHits(payload);
  } catch (error) {
    resultsEl.hidden = true;
    status.set(error instanceof Error ? error.message : String(error), true);
  } finally {
    submitBtn.disabled = false;
  }
});
