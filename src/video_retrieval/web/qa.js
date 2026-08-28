import {
  eventSpecByIdFromPlan,
  renderChainCards,
  renderEventPlan,
} from "./chains.js";
import { downloadTextFile } from "./shared/export.js";
import { formatScore, joinMeta, sanitizeQueryId } from "./shared/format.js";
import { createLightboxController } from "./shared/lightbox.js";
import { createStatusController } from "./shared/status.js";

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
const planEl = document.getElementById("plan");
const planBodyEl = document.getElementById("plan-body");
const resultsChainsEl = document.getElementById("results-chains");
const hitsGridEl = document.getElementById("hits-grid");

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

let lastPayload = null;
/** @type {Array<Record<string, unknown>>} */
let rankedHits = [];

function selectResult(item, index) {
  const chain = item.chain || {};
  answerVideoEl.value = chain.video_id || "";
  answerFrameEl.value = item.questioned_frame_id ?? "";
  answerTextEl.value = item.answer || "";
  resultsMetaEl.textContent = joinMeta([
    `#${index + 1}`,
    chain.video_id,
    item.questioned_event_id,
    `f${item.questioned_frame_id}`,
    item.answer ? `ans ${item.answer}` : null,
  ]);
}

function renderResultChains(payload) {
  resultsChainsEl.replaceChildren();
  const results = payload.results || [];
  results.forEach((item, index) => {
    const wrap = document.createElement("section");
    wrap.className = "qa-result-chain";

    const head = document.createElement("div");
    head.className = "trake-chain-head";
    const title = document.createElement("h3");
    title.textContent = `#${index + 1} · ${item.chain?.video_id || "?"} · answer: ${item.answer || "(blank)"}`;
    const useBtn = document.createElement("button");
    useBtn.type = "button";
    useBtn.className = "move-btn";
    useBtn.textContent = "Use this result";
    useBtn.addEventListener("click", () => selectResult(item, index));
    head.append(title, useBtn);
    wrap.appendChild(head);

    const chainHost = document.createElement("div");
    chainHost.className = "trake-chains";
    renderChainCards(chainHost, [item.chain], {
      highlightQuestion: true,
      questionedEventId: item.questioned_event_id,
      eventSpecById: eventSpecByIdFromPlan(payload.plan),
      showEventChannels: true,
      onEventClick: (hit) => {
        selectResult(item, index);
        lightbox.open({ ...hit, answer: item.answer });
      },
    });
    wrap.appendChild(chainHost);
    resultsChainsEl.appendChild(wrap);
  });
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
    btn.addEventListener("click", () => lightbox.open(hit));
    const body = document.createElement("div");
    body.className = "card-body";
    const title = document.createElement("h3");
    title.textContent = `#${index + 1} · ${hit.video_id || "unknown"}`;
    const detail = document.createElement("p");
    detail.textContent = joinMeta([
      hit.frame_id != null ? `f${hit.frame_id}` : null,
      hit.answer ? `ans ${hit.answer}` : null,
      `score ${formatScore(hit.score)}`,
      hit.source,
    ]);
    body.append(title, detail);
    card.append(btn, body);
    hitsGridEl.appendChild(card);
  });
}

function renderResult(payload) {
  lastPayload = payload;
  resultsEl.hidden = false;
  renderEventPlan(planBodyEl, planEl, payload.plan);
  answerVideoEl.value = payload.video_id || "";
  answerFrameEl.value = payload.frame_id ?? "";
  answerTextEl.value = payload.answer || "";
  resultsMetaEl.textContent = joinMeta([
    payload.video_id ? `video ${payload.video_id}` : null,
    payload.frame_id != null ? `frame ${payload.frame_id}` : null,
    `${(payload.results || []).length} chain results`,
    `${(payload.hits || []).length} export rows`,
  ]);

  renderResultChains(payload);
  renderHits(payload.hits || []);
  if ((payload.results || []).length) {
    selectResult(payload.results[0], 0);
  }
}

function exportCsv() {
  const answer = (answerTextEl.value || "").trim();
  if (!answer) {
    status.set("Set the answer text before exporting.", true);
    return;
  }
  if (lastPayload?.csv_text) {
    const queryId = sanitizeQueryId(queryIdEl.value);
    const filename = downloadTextFile(lastPayload.csv_text, `${queryId}.csv`);
    status.set(`Exported server CSV → ${filename}`);
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
    status.set("No valid ranked rows to export.", true);
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
  const queryId = sanitizeQueryId(queryIdEl.value);
  const filename = downloadTextFile(csv, `${queryId}.csv`);
  status.set(`Exported ${valid.length} rows → ${filename}`);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionEl.value.trim();
  if (!question) {
    status.set("Enter a QA question.", true);
    return;
  }
  submitBtn.disabled = true;
  status.set("Running QA (event chains + answer)…");
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
    status.set(
      `QA complete — ${(payload.results || []).length} chains, ${(payload.hits || []).length} export rows.`
    );
  } catch (error) {
    resultsEl.hidden = true;
    status.set(error instanceof Error ? error.message : String(error), true);
  } finally {
    submitBtn.disabled = false;
  }
});

exportBtn.addEventListener("click", exportCsv);
