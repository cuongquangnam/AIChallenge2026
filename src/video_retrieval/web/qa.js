const form = document.getElementById("qa-form");
const questionEl = document.getElementById("question");
const queryIdEl = document.getElementById("query-id");
const limitEl = document.getElementById("limit");
const submitBtn = document.getElementById("submit-btn");
const exportBtn = document.getElementById("export-btn");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const resultsMetaEl = document.getElementById("results-meta");
const answerVideoEl = document.getElementById("answer-video");
const answerFrameEl = document.getElementById("answer-frame");
const answerTextEl = document.getElementById("answer-text");
const descriptionsEl = document.getElementById("descriptions");
const descriptionListEl = document.getElementById("description-list");
const evidenceCardEl = document.getElementById("evidence-card");
const evidenceImgEl = document.getElementById("evidence-img");
const evidenceMetaEl = document.getElementById("evidence-meta");
const evidenceThumbBtn = document.getElementById("evidence-thumb-btn");
const playEvidenceBtn = document.getElementById("play-evidence");
const groupsEl = document.getElementById("groups");
const hitsGridEl = document.getElementById("hits-grid");
const lightbox = document.getElementById("lightbox");
const lightboxImg = document.getElementById("lightbox-img");
const lightboxVideo = document.getElementById("lightbox-video");
const lightboxMeta = document.getElementById("lightbox-meta");
const lightboxClose = document.getElementById("lightbox-close");
const lightboxOpenVideo = document.getElementById("lightbox-open-video");

let lastPayload = null;
let activeHit = null;
/** @type {Array<Record<string, unknown>>} */
let rankedHits = [];

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

function formatScore(score) {
  if (typeof score !== "number" || Number.isNaN(score)) return "—";
  return score.toFixed(3);
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
  const bits = [
    hit.video_id,
    hit.frame_id != null ? `f${hit.frame_id}` : hit.frame_index != null ? `f${hit.frame_index}` : null,
    formatTime(hit.timestamp_sec),
    hit.answer ? `ans ${hit.answer}` : null,
  ].filter(Boolean);
  lightboxMeta.textContent = bits.join(" · ");
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

function selectEvidence(hit) {
  answerVideoEl.value = hit.video_id || "";
  answerFrameEl.value = hit.frame_id ?? hit.frame_index ?? "";
  if (hit.answer) answerTextEl.value = hit.answer;
  evidenceCardEl.hidden = false;
  evidenceImgEl.src = hit.image_url || hit.image_data_url || "";
  evidenceMetaEl.textContent = [
    hit.video_id,
    hit.frame_id != null ? `f${hit.frame_id}` : `f${hit.frame_index}`,
    formatTime(hit.timestamp_sec),
  ]
    .filter(Boolean)
    .join(" · ");
  playEvidenceBtn.hidden = !hit.video_url;
  playEvidenceBtn.onclick = () => playFromHit(hit);
  evidenceThumbBtn.onclick = () => openLightbox(hit);
}

function renderHits(hits) {
  rankedHits = hits || [];
  hitsGridEl.replaceChildren();
  rankedHits.forEach((hit, index) => {
    const card = document.createElement("article");
    card.className = "card";
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "thumb-btn";
    const img = document.createElement("img");
    img.alt = `${hit.video_id} f${hit.frame_id}`;
    img.src = hit.image_url || hit.image_data_url || "";
    btn.appendChild(img);
    btn.addEventListener("click", () => {
      selectEvidence(hit);
      openLightbox(hit);
    });
    const body = document.createElement("div");
    body.className = "card-body";
    const title = document.createElement("h3");
    title.textContent = `#${index + 1} · ${hit.video_id || "unknown"}`;
    const detail = document.createElement("p");
    detail.textContent = [
      hit.frame_id != null ? `f${hit.frame_id}` : null,
      hit.answer ? `ans ${hit.answer}` : null,
      `score ${formatScore(hit.score)}`,
    ]
      .filter(Boolean)
      .join(" · ");
    const useBtn = document.createElement("button");
    useBtn.type = "button";
    useBtn.className = "move-btn";
    useBtn.textContent = "Use as answer";
    useBtn.addEventListener("click", (event) => {
      event.stopPropagation();
      selectEvidence(hit);
    });
    body.append(title, detail, useBtn);
    card.append(btn, body);
    hitsGridEl.appendChild(card);
  });
}

function renderGroups(payload) {
  groupsEl.replaceChildren();
  const groups = payload.frame_groups || [];
  for (const group of groups) {
    const section = document.createElement("section");
    section.className = "qa-group";
    const title = document.createElement("h3");
    title.textContent = `Center f${group.center_frame_id} · score ${Number(group.retrieval_score || 0).toFixed(3)}`;
    section.appendChild(title);
    const row = document.createElement("div");
    row.className = "qa-group-frames";
    for (const frame of group.frames || []) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "qa-frame-btn";
      const img = document.createElement("img");
      img.src = frame.image_url || "";
      img.alt = `frame ${frame.frame_id}`;
      const label = document.createElement("span");
      label.textContent = `f${frame.frame_id}`;
      btn.append(img, label);
      const hit = {
        video_id: payload.video_id,
        frame_id: frame.frame_id,
        frame_index: frame.frame_id,
        timestamp_sec: frame.timestamp_sec,
        image_url: frame.image_url,
        video_url: frame.video_url || payload.video_url,
        answer: answerTextEl.value || payload.answer,
      };
      btn.addEventListener("click", () => {
        selectEvidence(hit);
        openLightbox(hit);
      });
      row.appendChild(btn);
    }
    section.appendChild(row);
    groupsEl.appendChild(section);
  }
}

function renderResult(payload) {
  lastPayload = payload;
  resultsEl.hidden = false;
  answerVideoEl.value = payload.video_id || "";
  answerFrameEl.value = payload.frame_id ?? "";
  answerTextEl.value = payload.answer || "";
  resultsMetaEl.textContent = [
    payload.video_id ? `video ${payload.video_id}` : null,
    payload.frame_id != null ? `frame ${payload.frame_id}` : null,
    `${(payload.hits || []).length} ranked rows`,
  ]
    .filter(Boolean)
    .join(" · ");

  const descriptions = payload.descriptions || [];
  descriptionsEl.hidden = descriptions.length === 0;
  descriptionListEl.replaceChildren();
  for (const item of descriptions) {
    const li = document.createElement("li");
    li.textContent = item;
    descriptionListEl.appendChild(li);
  }

  if (payload.evidence_hit) {
    selectEvidence({
      ...payload.evidence_hit,
      frame_id: payload.frame_id ?? payload.evidence_hit.frame_id,
      answer: payload.answer,
    });
  } else {
    evidenceCardEl.hidden = true;
  }
  renderHits(payload.hits || []);
  renderGroups(payload);
}

function exportCsv() {
  const answer = (answerTextEl.value || "").trim();
  if (!answer) {
    setStatus("Set the answer text before exporting.", true);
    return;
  }
  const rows =
    rankedHits.length > 0
      ? rankedHits.map((hit) => [
          String(hit.video_id || "").trim(),
          Number(hit.frame_id ?? hit.frame_index),
          answer,
        ])
      : [
          [
            (answerVideoEl.value || "").trim(),
            Number(answerFrameEl.value),
            answer,
          ],
        ];
  const valid = rows.filter(
    ([videoId, frameId]) => videoId && Number.isFinite(frameId) && frameId >= 0
  );
  if (!valid.length) {
    setStatus("No valid ranked rows to export.", true);
    return;
  }
  const csv =
    valid
      .map(([videoId, frameId, ans]) => {
        const needsQuote = /[",\n]/.test(ans);
        const answerCell = needsQuote ? `"${String(ans).replace(/"/g, '""')}"` : ans;
        return `${videoId},${Math.trunc(frameId)},${answerCell}`;
      })
      .join("\n") + "\n";
  const queryId = (queryIdEl.value || "query").trim().replace(/[^\w.-]+/g, "-");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${queryId || "query"}.csv`;
  anchor.click();
  URL.revokeObjectURL(url);
  setStatus(`Exported ${valid.length} rows → ${anchor.download}`);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionEl.value.trim();
  if (!question) {
    setStatus("Enter a QA question.", true);
    return;
  }
  submitBtn.disabled = true;
  setStatus("Running QA (retrieve + answer)…");
  resultsEl.hidden = true;
  try {
    const response = await fetch("/qa", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        limit: Number(limitEl.value),
      }),
    });
    if (!response.ok) {
      let detail = await response.text();
      try {
        const parsed = JSON.parse(detail);
        detail = parsed.detail || detail;
      } catch {
        /* keep text */
      }
      throw new Error(detail || `QA failed (${response.status})`);
    }
    const payload = await response.json();
    renderResult(payload);
    setStatus(`QA complete — ${(payload.hits || []).length} ranked rows.`);
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
