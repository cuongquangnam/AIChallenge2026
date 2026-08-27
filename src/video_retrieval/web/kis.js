const form = document.getElementById("search-form");
const queryEl = document.getElementById("query");
const modeEl = document.getElementById("mode");
const limitEl = document.getElementById("limit");
const submitBtn = document.getElementById("submit-btn");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const resultsMetaEl = document.getElementById("results-meta");
const planEl = document.getElementById("plan");
const planBodyEl = document.getElementById("plan-body");
const gridEl = document.getElementById("grid");
const lightbox = document.getElementById("lightbox");
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

let activeHit = null;
let lastPlan = null;
let lastMode = "mixed";
/** @type {Array<Record<string, unknown>>} */
let resultHits = [];
let hitSeq = 0;

const PAD_OFFSETS = [
  1, -1, 2, -2, 3, -3, 5, -5, 8, -8, 12, -12, 15, -15, 20, -20, 25, -25, 30, -30,
  40, -40, 50, -50,
];

function setStatus(message, isError = false) {
  statusEl.textContent = message || "";
  statusEl.classList.toggle("error", Boolean(isError));
}

function formatScore(score) {
  if (typeof score !== "number" || Number.isNaN(score)) return "—";
  return score.toFixed(3);
}

function formatTime(sec) {
  if (typeof sec !== "number" || Number.isNaN(sec)) return null;
  const total = Math.max(0, Math.floor(sec));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function seekSeconds(hit) {
  if (typeof hit?.timestamp_sec === "number" && !Number.isNaN(hit.timestamp_sec)) {
    return Math.max(0, hit.timestamp_sec);
  }
  return 0;
}

function hitsToSubmissionRows(hits, limit) {
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
    const offset = PAD_OFFSETS[offsetI % PAD_OFFSETS.length];
    const cycle = Math.floor(offsetI / PAD_OFFSETS.length);
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

function exportCurrentList() {
  if (!resultHits.length) {
    setStatus("No frames to export.", true);
    return;
  }

  const queryId = (exportQueryIdEl.value || "query").trim().replace(/[^\w.-]+/g, "-");
  const limitChoice = exportLimitEl.value;
  const limit = limitChoice === "exact" ? null : Number(limitChoice);
  const rows = hitsToSubmissionRows(resultHits, limit);

  if (!rows.length) {
    setStatus("Export needs frames with a valid frame index.", true);
    return;
  }
  if (limit != null && rows.length < limit) {
    setStatus(`Could only build ${rows.length}/${limit} rows.`, true);
    return;
  }

  const csv = rows.map(([videoId, frameIdx]) => `${videoId},${frameIdx}`).join("\n") + "\n";
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${queryId || "query"}.csv`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);

  setStatus(
    `Exported ${rows.length} rows → ${anchor.download} (video_id,frame_idx).`
  );
}

function queryIdFromFilename(name) {
  const base = String(name || "")
    .replace(/^.*[\\/]/, "")
    .replace(/\.csv$/i, "")
    .trim();
  return base.replace(/[^\w.-]+/g, "-") || "query";
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
  setStatus(`Importing ${file.name}…`);
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
      setStatus(
        errCount
          ? `Imported 0/${total} frames from ${file.name} (${errCount} errors).`
          : `No frames could be resolved from ${file.name}.`,
        true
      );
      return;
    }
    setStatus(
      errCount
        ? `Imported ${resolved}/${total} frames from ${file.name} (${errCount} missing).`
        : `Imported ${resolved} frames from ${file.name}.`
    );
  } catch (error) {
    setStatus(error instanceof Error ? error.message : String(error), true);
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

function updateAddFrameButton() {
  const videoVisible = !lightboxVideo.hidden && Boolean(lightboxVideo.src);
  lightboxAddFrame.hidden = !videoVisible || !activeHit?.video_id;
}

function stopVideo() {
  lightboxVideo.pause();
  lightboxVideo.removeAttribute("src");
  lightboxVideo.load();
  lightboxVideo.hidden = true;
  lightboxImg.hidden = false;
  updateAddFrameButton();
}

function showKeyframe(hit) {
  stopVideo();
  const src = hit.image_url || hit.image_data_url;
  if (src) {
    lightboxImg.src = src;
    lightboxImg.alt = `${hit.video_id} frame ${hit.frame_index ?? ""}`;
  } else {
    lightboxImg.removeAttribute("src");
    lightboxImg.alt = "No keyframe image";
  }
  lightboxOpenVideo.hidden = !hit.video_url;
  lightboxOpenVideo.textContent = hit.video_url
    ? `Play from ${formatTime(seekSeconds(hit)) || "0:00"}`
    : "Video unavailable";
  updateAddFrameButton();
}

function playFromMoment(hit) {
  if (!hit?.video_url) {
    setStatus("Video file not found for this result.", true);
    return;
  }
  const t = seekSeconds(hit);
  lightboxImg.hidden = true;
  lightboxVideo.hidden = false;
  lightboxOpenVideo.hidden = true;
  updateAddFrameButton();

  const onReady = () => {
    try {
      lightboxVideo.currentTime = t;
    } catch {
      /* ignore seek race */
    }
    lightboxVideo.play().catch(() => {});
    updateAddFrameButton();
  };

  lightboxVideo.onloadedmetadata = onReady;
  if (lightboxVideo.src !== new URL(hit.video_url, window.location.origin).href) {
    lightboxVideo.src = hit.video_url;
    lightboxVideo.load();
  } else if (lightboxVideo.readyState >= 1) {
    onReady();
  } else {
    lightboxVideo.addEventListener("loadedmetadata", onReady, { once: true });
  }
}

function openLightbox(hit) {
  activeHit = hit;
  const bits = [
    hit.video_id,
    hit.frame_index != null ? `frame ${hit.frame_index}` : null,
    hit.shot_index != null ? `shot ${hit.shot_index}` : null,
    hit.role ? `role ${hit.role}` : null,
    hit.source === "user_capture" ? "added by you" : null,
    formatTime(hit.timestamp_sec) ? `t=${formatTime(hit.timestamp_sec)}` : null,
    `score ${formatScore(hit.score)}`,
  ].filter(Boolean);
  lightboxMeta.textContent = bits.join(" · ");
  showKeyframe(hit);
  lightbox.showModal();
}

function closeLightbox() {
  stopVideo();
  activeHit = null;
  lightbox.close();
}

function appendPlanRow(label, value) {
  const row = document.createElement("div");
  row.className = "plan-row";
  const dt = document.createElement("dt");
  dt.textContent = label;
  const dd = document.createElement("dd");
  const text = typeof value === "string" ? value.trim() : "";
  if (text) {
    dd.textContent = text;
  } else {
    dd.className = "plan-empty";
    dd.textContent = "(empty)";
  }
  row.append(dt, dd);
  planBodyEl.appendChild(row);
}

function renderPlan(plan) {
  planBodyEl.replaceChildren();
  if (!plan) {
    planEl.hidden = true;
    return;
  }

  planEl.hidden = false;
  planEl.classList.toggle("plan-fallback", plan.source === "heuristic");

  const sourceRow = document.createElement("div");
  sourceRow.className = "plan-row";
  const sourceDt = document.createElement("dt");
  sourceDt.textContent = "Source";
  const sourceDd = document.createElement("dd");
  if (plan.source === "heuristic") {
    sourceDd.innerHTML =
      "<strong>heuristic fallback</strong> — Gemini blocked or failed, so the raw query was sent to every channel with equal weights.";
  } else {
    sourceDd.textContent = plan.source || "unknown";
  }
  sourceRow.append(sourceDt, sourceDd);
  planBodyEl.appendChild(sourceRow);

  appendPlanRow("Visual", plan.visual || "");
  appendPlanRow("OCR", plan.ocr || "");
  appendPlanRow("ASR", plan.asr || "");

  const weights = plan.weights || {};
  const row = document.createElement("div");
  row.className = "plan-row";
  const dt = document.createElement("dt");
  dt.textContent = "Weights";
  const dd = document.createElement("dd");
  dd.className = "plan-weights";
  const total =
    (Number(weights.visual) || 0) + (Number(weights.ocr) || 0) + (Number(weights.asr) || 0);
  for (const key of ["visual", "ocr", "asr"]) {
    const chip = document.createElement("span");
    chip.className = "weight-chip";
    const raw = typeof weights[key] === "number" ? weights[key] : 0;
    const normalized = total > 0 ? raw / total : 0;
    chip.textContent = `${key} ${normalized.toFixed(2)}`;
    dd.appendChild(chip);
  }
  row.append(dt, dd);
  planBodyEl.appendChild(row);
}

function updateResultsMeta() {
  const added = resultHits.filter((hit) => hit.source === "user_capture").length;
  const fromCsv = resultHits.filter((hit) => hit.source === "csv_import").length;
  const planBits = [];
  if (lastPlan?.visual) planBits.push("visual");
  if (lastPlan?.ocr) planBits.push("ocr");
  if (lastPlan?.asr) planBits.push("asr");
  resultsMetaEl.textContent = [
    `${resultHits.length} frames`,
    `mode ${lastMode}`,
    planBits.length ? `channels ${planBits.join("+")}` : null,
    fromCsv ? `${fromCsv} from csv` : null,
    added ? `${added} added` : null,
    "drag or use Move frame #",
  ]
    .filter(Boolean)
    .join(" · ");
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
  setStatus(
    `Moved #${fromIndex + 1} → #${toIndex + 1}: ${moved.video_id}, f${moved.frame_index ?? "?"}`
  );
}

function moveHitByPosition() {
  if (!resultHits.length) {
    setStatus("No frames to move.", true);
    return;
  }
  const fromPos = Number(moveFromEl.value);
  const toPos = Number(moveToEl.value);
  if (!Number.isInteger(fromPos) || !Number.isInteger(toPos)) {
    setStatus("Enter whole numbers for both positions.", true);
    return;
  }
  const max = resultHits.length;
  if (fromPos < 1 || fromPos > max || toPos < 1 || toPos > max) {
    setStatus(`Positions must be between 1 and ${max}.`, true);
    return;
  }
  if (fromPos === toPos) {
    setStatus(`Frame #${fromPos} is already at that position.`);
    return;
  }
  moveHitAtIndex(fromPos - 1, toPos - 1);
  moveFromEl.value = "";
  moveToEl.value = "";
  moveFromEl.focus();
}

function removeHit(uid) {
  resultHits = resultHits.filter((hit) => hit._uid !== uid);
  if (activeHit?._uid === uid) {
    closeLightbox();
  }
  renderResultGrid();
  if (!resultHits.length) {
    setStatus("All frames removed. Run a new search to refill results.");
  } else {
    setStatus(`Removed a frame · ${resultHits.length} remaining.`);
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
    setStatus(
      `Frame ${next.video_id}, ${next.frame_index} is already in results.`,
      true
    );
    return;
  }
  resultHits = [next, ...resultHits];
  renderResultGrid();
  setStatus(
    `Added ${next.video_id}, frame ${next.frame_index} @ ${formatTime(next.timestamp_sec)}.`
  );
}

async function captureCurrentFrame() {
  if (!activeHit?.video_id || lightboxVideo.hidden) {
    setStatus("Open the video first, then add the current frame.", true);
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
    setStatus(error instanceof Error ? error.message : String(error), true);
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
    openBtn.addEventListener("click", () => openLightbox(hit));

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
        openLightbox(hit);
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
  lastMode = payload.mode || "mixed";
  resultHits = (payload.hits || []).map(withHitId);

  if (!resultHits.length) {
    resultsEl.hidden = true;
    planEl.hidden = true;
    if ((payload.mode || "") !== "csv") {
      setStatus("No keyframes matched this query.");
    }
    return;
  }

  resultsEl.hidden = false;
  renderPlan(lastPlan);
  renderResultGrid();
  if ((payload.mode || "") !== "csv") {
    setStatus(`Found ${resultHits.length} keyframe${resultHits.length === 1 ? "" : "s"}.`);
  }
}

lightboxClose.addEventListener("click", closeLightbox);
lightboxOpenVideo.addEventListener("click", () => {
  if (activeHit) playFromMoment(activeHit);
});
lightboxAddFrame.addEventListener("click", () => {
  captureCurrentFrame();
});
lightboxVideo.addEventListener("play", updateAddFrameButton);
lightboxVideo.addEventListener("loadeddata", updateAddFrameButton);
lightbox.addEventListener("close", stopVideo);
lightbox.addEventListener("click", (event) => {
  if (event.target === lightbox) closeLightbox();
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
    setStatus("Enter a question or scene description.", true);
    return;
  }

  submitBtn.disabled = true;
  setStatus("Running KIS…");
  resultsEl.hidden = true;
  gridEl.replaceChildren();

  try {
    const response = await fetch("/kis", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        mode: modeEl.value,
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
    setStatus(error instanceof Error ? error.message : String(error), true);
  } finally {
    submitBtn.disabled = false;
  }
});
