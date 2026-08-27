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
const lightbox = document.getElementById("lightbox");
const lightboxImg = document.getElementById("lightbox-img");
const lightboxVideo = document.getElementById("lightbox-video");
const lightboxMeta = document.getElementById("lightbox-meta");
const lightboxClose = document.getElementById("lightbox-close");
const lightboxOpenVideo = document.getElementById("lightbox-open-video");

/** @type {Array<{event_id: string, frame_index: number, image_url?: string, video_url?: string, timestamp_sec?: number}>} */
let selectedEvents = [];
let activeHit = null;
/** @type {Array<Record<string, unknown>>} */
let lastChains = [];

function setStatus(message, isError = false) {
  statusEl.textContent = message || "";
  statusEl.classList.toggle("error", Boolean(isError));
}

function formatTime(sec) {
  if (typeof sec !== "number" || Number.isNaN(sec)) return null;
  const total = Math.max(0, Math.floor(sec));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function stopVideo() {
  lightboxVideo.pause();
  lightboxVideo.removeAttribute("src");
  lightboxVideo.load();
  lightboxVideo.hidden = true;
  lightboxImg.hidden = false;
  lightboxOpenVideo.hidden = true;
}

function closeLightbox() {
  stopVideo();
  if (lightbox.open) lightbox.close();
}

function openLightbox(hit) {
  activeHit = hit;
  const src = hit.image_url || hit.image_data_url || "";
  lightboxImg.src = src;
  lightboxImg.hidden = !src;
  lightboxVideo.hidden = true;
  lightboxMeta.textContent = [
    hit.video_id,
    hit.event_id,
    hit.frame_index != null ? `f${hit.frame_index}` : null,
    formatTime(hit.timestamp_sec),
  ]
    .filter(Boolean)
    .join(" · ");
  lightboxOpenVideo.hidden = !hit.video_url;
  if (!lightbox.open) lightbox.showModal();
}

function playFromHit(hit) {
  if (!hit?.video_url) return;
  openLightbox(hit);
  lightboxImg.hidden = true;
  lightboxVideo.hidden = false;
  lightboxVideo.src = hit.video_url;
  const t = typeof hit.timestamp_sec === "number" ? hit.timestamp_sec : 0;
  lightboxVideo.currentTime = Math.max(0, t);
  lightboxVideo.play().catch(() => {});
}

function renderPlan(plan) {
  if (!plan) {
    planEl.hidden = true;
    return;
  }
  planEl.hidden = false;
  planBodyEl.replaceChildren();
  if (plan.context) {
    const dt = document.createElement("dt");
    dt.textContent = "context";
    const dd = document.createElement("dd");
    dd.textContent = plan.context;
    planBodyEl.append(dt, dd);
  }
  for (const event of plan.events || []) {
    const dt = document.createElement("dt");
    dt.textContent = event.event_id;
    const dd = document.createElement("dd");
    dd.textContent = event.visual || event.ocr || event.asr || "—";
    planBodyEl.append(dt, dd);
  }
}

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
  selectedEvents = (chain.events || []).map((event) => ({ ...event, video_id: chain.video_id }));
  syncEditorFromSelection();
}

function renderChains(payload) {
  lastChains = payload.chains || [];
  chainsEl.replaceChildren();
  lastChains.forEach((chain, index) => {
    const card = document.createElement("article");
    card.className = "trake-chain";
    const head = document.createElement("div");
    head.className = "trake-chain-head";
    const title = document.createElement("h3");
    title.textContent = `#${index + 1} · ${chain.video_id} · score ${Number(chain.score || 0).toFixed(3)}`;
    const useBtn = document.createElement("button");
    useBtn.type = "button";
    useBtn.className = "move-btn";
    useBtn.textContent = "Use this chain";
    useBtn.addEventListener("click", () => {
      selectChain(chain);
      setStatus(`Selected chain ${chain.video_id}`);
    });
    head.append(title, useBtn);
    card.appendChild(head);

    const timeline = document.createElement("div");
    timeline.className = "trake-timeline";
    for (const event of chain.events || []) {
      const cell = document.createElement("button");
      cell.type = "button";
      cell.className = "trake-event";
      const img = document.createElement("img");
      img.src = event.image_url || "";
      img.alt = `${event.event_id} frame`;
      const label = document.createElement("span");
      label.textContent = `${event.event_id} · f${event.frame_index}`;
      cell.append(img, label);
      const hit = { ...event, video_id: chain.video_id };
      cell.addEventListener("click", () => openLightbox(hit));
      cell.addEventListener("dblclick", () => playFromHit(hit));
      timeline.appendChild(cell);
    }
    card.appendChild(timeline);
    chainsEl.appendChild(card);
  });
}

function exportCsv() {
  const chains = document.querySelectorAll(".trake-chain");
  // Prefer edited selection; also export all returned chains as ranked rows.
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
    setStatus("Select or run a chain before exporting.", true);
    return;
  }
  const csv = lines.join("\n") + "\n";
  const queryId = (queryIdEl.value || "query").trim().replace(/[^\w.-]+/g, "-");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${queryId || "query"}.csv`;
  anchor.click();
  URL.revokeObjectURL(url);
  setStatus(`Exported ${lines.length} chain rows → ${anchor.download}`);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = queryEl.value.trim();
  if (!query) {
    setStatus("Paste a TRAKE query with E1…En events.", true);
    return;
  }
  submitBtn.disabled = true;
  setStatus("Running TRAKE…");
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
    resultsMetaEl.textContent = [
      `${(payload.chains || []).length} chains`,
      payload.csv_row ? `best ${payload.csv_row}` : null,
    ]
      .filter(Boolean)
      .join(" · ");
    renderPlan(payload.plan);
    renderChains(payload);
    if ((payload.chains || []).length) {
      selectChain(payload.chains[0]);
    } else {
      editorEl.hidden = true;
      selectedEvents = [];
    }
    setStatus("TRAKE complete — review chains, edit frames if needed, export.");
  } catch (error) {
    resultsEl.hidden = true;
    setStatus(error instanceof Error ? error.message : String(error), true);
  } finally {
    submitBtn.disabled = false;
  }
});

exportBtn.addEventListener("click", exportCsv);
lightboxClose.addEventListener("click", closeLightbox);
lightboxOpenVideo.addEventListener("click", () => {
  if (activeHit) playFromHit(activeHit);
});
lightbox.addEventListener("close", stopVideo);
lightbox.addEventListener("click", (event) => {
  if (event.target === lightbox) closeLightbox();
});
