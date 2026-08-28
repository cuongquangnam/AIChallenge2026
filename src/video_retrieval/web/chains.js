/** Shared event-chain rendering for KIS / QA / TRAKE pages. */

const PLAN_CHANNELS = [
  ["description", "description"],
  ["visual", "visual"],
  ["ocr", "ocr"],
  ["asr", "asr"],
];

const COMPACT_CHANNELS = [
  ["visual", "visual"],
  ["ocr", "ocr"],
  ["asr", "asr"],
];

function appendPlanChannel(parent, label, text) {
  if (!text) return;
  const row = document.createElement("div");
  row.className = "plan-channel";
  const lab = document.createElement("span");
  lab.className = "plan-channel-label";
  lab.textContent = label;
  const val = document.createElement("span");
  val.className = "plan-channel-text";
  val.textContent = text;
  row.append(lab, val);
  parent.appendChild(row);
}

function appendPlanTerm(planBodyEl, label, value) {
  const dt = document.createElement("dt");
  dt.textContent = label;
  const dd = document.createElement("dd");
  dd.textContent = value;
  planBodyEl.append(dt, dd);
}

function formatGap(spec) {
  const expected = spec?.gap_from_prev_sec;
  if (typeof expected !== "number" || Number.isNaN(expected)) return "";
  const round1 = (value) =>
    Number.isInteger(value) ? String(value) : value.toFixed(1);
  const lo = spec.gap_min_sec;
  const hi = spec.gap_max_sec;
  if (typeof lo === "number" && typeof hi === "number") {
    return `~${round1(expected)}s (${round1(lo)}–${round1(hi)}s)`;
  }
  return `~${round1(expected)}s`;
}

function appendEventSpecChannels(parent, spec, { compact = false } = {}) {
  if (!parent || !spec) return;
  const channels = compact ? COMPACT_CHANNELS : PLAN_CHANNELS;
  for (const [label, key] of channels) {
    const text = spec[key];
    if (!text) continue;
    if (compact) {
      const line = document.createElement("span");
      line.className = "trake-event-spec-line";
      line.textContent = `${label}: ${text}`;
      parent.appendChild(line);
    } else {
      appendPlanChannel(parent, label, text);
    }
  }
  const gap = formatGap(spec);
  if (gap) {
    if (compact) {
      const line = document.createElement("span");
      line.className = "trake-event-spec-line";
      line.textContent = `gap: ${gap}`;
      parent.appendChild(line);
    } else {
      appendPlanChannel(parent, "gap from prev", gap);
    }
  }
}

export function eventSpecByIdFromPlan(plan) {
  const map = {};
  for (const event of plan?.events || []) {
    if (event?.event_id) {
      map[event.event_id] = event;
    }
  }
  return map;
}

export function renderEventPlan(planBodyEl, planEl, plan) {
  if (!planBodyEl || !planEl) return;
  if (!plan) {
    planEl.hidden = true;
    return;
  }
  planEl.hidden = false;
  planBodyEl.replaceChildren();

  if (plan.context) appendPlanTerm(planBodyEl, "context", plan.context);
  if (plan.question_event_id) {
    appendPlanTerm(planBodyEl, "question event", plan.question_event_id);
  }

  for (const event of plan.events || []) {
    const dt = document.createElement("dt");
    dt.textContent = event.is_question_target ? `${event.event_id} ?` : event.event_id;
    const dd = document.createElement("dd");
    dd.className = "plan-event-details";
    appendEventSpecChannels(dd, event);
    if (!dd.childElementCount) {
      dd.textContent = "—";
    }
    planBodyEl.append(dt, dd);
  }
}

function createChainEventCell(event, chain, options) {
  const {
    highlightQuestion,
    questionedEventId,
    eventSpecById,
    showEventChannels,
    onEventClick,
    chainIndex,
  } = options;

  const cell = document.createElement("button");
  cell.type = "button";
  cell.className = "trake-event";
  if (
    highlightQuestion &&
    (event.is_question_target || event.event_id === questionedEventId)
  ) {
    cell.classList.add("qa-question-event");
  }

  const img = document.createElement("img");
  img.src = event.image_url || "";
  img.alt = `${event.event_id} frame`;

  const label = document.createElement("span");
  label.textContent = `${event.event_id} · f${event.frame_index}`;
  cell.append(img, label);

  if (showEventChannels && eventSpecById) {
    const spec = eventSpecById[event.event_id];
    if (spec) {
      const specEl = document.createElement("div");
      specEl.className = "trake-event-spec";
      appendEventSpecChannels(specEl, spec, { compact: true });
      if (specEl.childElementCount) {
        cell.appendChild(specEl);
      }
    }
  }

  const hit = { ...event, video_id: chain.video_id, video_url: chain.video_url };
  cell.addEventListener("click", () => {
    if (onEventClick) onEventClick(hit, chain, chainIndex);
  });
  return cell;
}

export function renderChainCards(
  containerEl,
  chains,
  {
    onEventClick,
    highlightQuestion = false,
    questionedEventId = null,
    eventSpecById = null,
    showEventChannels = false,
    layout = "wrap",
    onChainAction = null,
    chainActionLabel = "Use this chain",
  } = {}
) {
  if (!containerEl) return;
  containerEl.replaceChildren();

  (chains || []).forEach((chain, index) => {
    const card = document.createElement("article");
    card.className = "trake-chain";

    const head = document.createElement("div");
    head.className = "trake-chain-head";
    const title = document.createElement("h3");
    title.textContent = `#${index + 1} · ${chain.video_id} · score ${Number(chain.score || 0).toFixed(3)}`;
    head.appendChild(title);

    if (onChainAction) {
      const actionBtn = document.createElement("button");
      actionBtn.type = "button";
      actionBtn.className = "move-btn";
      actionBtn.textContent = chainActionLabel;
      actionBtn.addEventListener("click", () => onChainAction(chain, index));
      head.appendChild(actionBtn);
    }

    card.appendChild(head);

    const row = document.createElement("div");
    row.className = layout === "timeline" ? "trake-timeline" : "trake-chain-events";

    for (const event of chain.events || []) {
      row.appendChild(
        createChainEventCell(event, chain, {
          highlightQuestion,
          questionedEventId,
          eventSpecById,
          showEventChannels,
          onEventClick,
          chainIndex: index,
        })
      );
    }

    card.appendChild(row);
    containerEl.appendChild(card);
  });
}
