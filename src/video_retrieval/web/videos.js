const form = document.getElementById("library-form");
const queryEl = document.getElementById("library-query");
const submitBtn = document.getElementById("library-submit");
const statusEl = document.getElementById("status");
const metaEl = document.getElementById("library-meta");
const gridEl = document.getElementById("grid");
const moreWrap = document.getElementById("library-more");
const loadMoreBtn = document.getElementById("load-more");
const folderBar = document.getElementById("folder-bar");
const folderTitle = document.getElementById("folder-title");
const backFoldersBtn = document.getElementById("back-folders");
const lightbox = document.getElementById("lightbox");
const lightboxVideo = document.getElementById("lightbox-video");
const lightboxMeta = document.getElementById("lightbox-meta");
const lightboxClose = document.getElementById("lightbox-close");

const PAGE_SIZE = 48;
let currentQuery = "";
/** @type {string | null} */
let currentSeries = null;
let offset = 0;
let total = 0;
/** @type {Array<Record<string, unknown>>} */
let loaded = [];

function setStatus(message, isError = false) {
  statusEl.textContent = message || "";
  statusEl.classList.toggle("error", Boolean(isError));
}

function formatBytes(bytes) {
  if (typeof bytes !== "number" || !Number.isFinite(bytes) || bytes < 0) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function updateFolderBar() {
  if (currentSeries) {
    folderBar.hidden = false;
    folderTitle.textContent = `${currentSeries} · ${total} video${total === 1 ? "" : "s"}`;
  } else {
    folderBar.hidden = true;
    folderTitle.textContent = "";
  }
}

function updateMeta(groupsCount = 0) {
  if (currentSeries) {
    const shown = loaded.length;
    metaEl.textContent = currentQuery
      ? `Folder ${currentSeries}: ${shown} of ${total} matching “${currentQuery}”`
      : `Folder ${currentSeries}: showing ${shown} of ${total}`;
    return;
  }
  if (!groupsCount) {
    metaEl.textContent = currentQuery
      ? `No folders match “${currentQuery}”`
      : "No videos in data/videos";
    return;
  }
  metaEl.textContent = currentQuery
    ? `${groupsCount} folder${groupsCount === 1 ? "" : "s"} · filtered “${currentQuery}”`
    : `${groupsCount} L-series folder${groupsCount === 1 ? "" : "s"}`;
}

function updateLoadMore() {
  const hasMore = Boolean(currentSeries) && loaded.length < total;
  moreWrap.hidden = !hasMore;
  loadMoreBtn.disabled = !hasMore;
}

function stopVideo() {
  lightboxVideo.pause();
  lightboxVideo.removeAttribute("src");
  lightboxVideo.load();
}

function closeLightbox() {
  if (lightbox.open) lightbox.close();
  stopVideo();
}

function openVideo(item) {
  if (!item?.video_url) {
    setStatus("Video file is missing.", true);
    return;
  }
  lightboxMeta.textContent = `${item.video_id} · ${item.filename || ""} · ${formatBytes(item.size_bytes)}`;
  lightboxVideo.src = item.video_url;
  lightboxVideo.currentTime = 0;
  if (!lightbox.open) lightbox.showModal();
  lightboxVideo.play().catch(() => {});
}

function makePoster(src, alt, emptyLabel) {
  if (src) {
    const img = document.createElement("img");
    img.className = "thumb";
    img.loading = "lazy";
    img.draggable = false;
    img.src = src;
    img.alt = alt;
    img.onerror = () => {
      img.replaceWith(
        Object.assign(document.createElement("div"), {
          className: "thumb missing",
          textContent: emptyLabel,
        })
      );
    };
    return img;
  }
  return Object.assign(document.createElement("div"), {
    className: "thumb missing",
    textContent: emptyLabel,
  });
}

function renderFolderCards(groups) {
  gridEl.replaceChildren();
  for (const group of groups) {
    const card = document.createElement("article");
    card.className = "card folder-card";

    const openBtn = document.createElement("button");
    openBtn.type = "button";
    openBtn.className = "card-main";
    openBtn.addEventListener("click", () => openSeries(group.series));

    openBtn.appendChild(
      makePoster(group.poster_url, `${group.series} folder`, "Folder")
    );

    const meta = document.createElement("div");
    meta.className = "meta";
    const title = document.createElement("strong");
    title.textContent = group.series;
    const detail = document.createElement("span");
    detail.textContent = `${group.count} video${group.count === 1 ? "" : "s"}`;
    meta.append(title, detail);
    openBtn.appendChild(meta);
    card.appendChild(openBtn);

    const actions = document.createElement("div");
    actions.className = "card-actions";
    const openFolderBtn = document.createElement("button");
    openFolderBtn.type = "button";
    openFolderBtn.className = "play-btn";
    openFolderBtn.textContent = "Open folder";
    openFolderBtn.addEventListener("click", (event) => {
      event.stopPropagation();
      openSeries(group.series);
    });
    actions.appendChild(openFolderBtn);
    card.appendChild(actions);
    gridEl.appendChild(card);
  }
}

function appendVideoCards(videos) {
  for (const item of videos) {
    const card = document.createElement("article");
    card.className = "card video-card";

    const openBtn = document.createElement("button");
    openBtn.type = "button";
    openBtn.className = "card-main";
    openBtn.addEventListener("click", () => openVideo(item));
    openBtn.appendChild(
      makePoster(item.poster_url, `${item.video_id} poster`, "No poster")
    );

    const meta = document.createElement("div");
    meta.className = "meta";
    const title = document.createElement("strong");
    title.textContent = item.video_id || "unknown";
    const detail = document.createElement("span");
    detail.textContent = [item.filename, formatBytes(item.size_bytes)].filter(Boolean).join(" · ");
    meta.append(title, detail);
    openBtn.appendChild(meta);
    card.appendChild(openBtn);

    const actions = document.createElement("div");
    actions.className = "card-actions";
    const playBtn = document.createElement("button");
    playBtn.type = "button";
    playBtn.className = "play-btn";
    playBtn.textContent = "Watch video";
    playBtn.addEventListener("click", (event) => {
      event.stopPropagation();
      openVideo(item);
    });
    actions.appendChild(playBtn);
    card.appendChild(actions);
    gridEl.appendChild(card);
  }
}

async function loadGroups() {
  currentSeries = null;
  loaded = [];
  offset = 0;
  total = 0;
  submitBtn.disabled = true;
  moreWrap.hidden = true;
  setStatus("Loading folders…");
  updateFolderBar();

  try {
    const params = new URLSearchParams();
    if (currentQuery) params.set("q", currentQuery);
    const response = await fetch(`/api/videos/groups?${params}`);
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(detail || `Request failed (${response.status})`);
    }
    const payload = await response.json();
    const groups = Array.isArray(payload.groups) ? payload.groups : [];
    renderFolderCards(groups);
    updateMeta(groups.length);
    setStatus(
      groups.length
        ? ""
        : currentQuery
          ? `No folders match “${currentQuery}”.`
          : "No videos found under data/videos."
    );
  } catch (error) {
    gridEl.replaceChildren();
    setStatus(error instanceof Error ? error.message : String(error), true);
    updateMeta(0);
  } finally {
    submitBtn.disabled = false;
    updateLoadMore();
  }
}

async function fetchVideos({ reset }) {
  if (!currentSeries) return;
  if (reset) {
    offset = 0;
    loaded = [];
    gridEl.replaceChildren();
  }

  submitBtn.disabled = true;
  loadMoreBtn.disabled = true;
  setStatus(reset ? `Loading ${currentSeries}…` : "Loading more…");

  try {
    const params = new URLSearchParams({
      series: currentSeries,
      offset: String(offset),
      limit: String(PAGE_SIZE),
    });
    if (currentQuery) params.set("q", currentQuery);

    const response = await fetch(`/api/videos?${params}`);
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(detail || `Request failed (${response.status})`);
    }
    const payload = await response.json();
    total = Number(payload.total) || 0;
    const page = Array.isArray(payload.videos) ? payload.videos : [];
    loaded = loaded.concat(page);
    offset = loaded.length;
    appendVideoCards(page);
    updateFolderBar();
    updateMeta();
    updateLoadMore();
    setStatus(
      page.length || total
        ? ""
        : currentQuery
          ? `No videos in ${currentSeries} match “${currentQuery}”.`
          : `No videos in ${currentSeries}.`
    );
  } catch (error) {
    setStatus(error instanceof Error ? error.message : String(error), true);
    updateLoadMore();
  } finally {
    submitBtn.disabled = false;
  }
}

function openSeries(series) {
  currentSeries = series;
  syncUrl();
  fetchVideos({ reset: true });
}

function applyStateFromUrl() {
  const params = new URLSearchParams(window.location.search);
  currentQuery = (params.get("q") || "").trim();
  currentSeries = (params.get("series") || "").trim() || null;
  queryEl.value = currentQuery;
}

function syncUrl() {
  const url = new URL(window.location.href);
  if (currentQuery) url.searchParams.set("q", currentQuery);
  else url.searchParams.delete("q");
  if (currentSeries) url.searchParams.set("series", currentSeries);
  else url.searchParams.delete("series");
  window.history.replaceState({}, "", url);
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  currentQuery = queryEl.value.trim();
  syncUrl();
  if (currentSeries) fetchVideos({ reset: true });
  else loadGroups();
});

backFoldersBtn.addEventListener("click", () => {
  currentSeries = null;
  syncUrl();
  loadGroups();
});

loadMoreBtn.addEventListener("click", () => fetchVideos({ reset: false }));

lightboxClose.addEventListener("click", closeLightbox);
lightbox.addEventListener("close", stopVideo);
lightbox.addEventListener("click", (event) => {
  if (event.target === lightbox) closeLightbox();
});

applyStateFromUrl();
if (currentSeries) fetchVideos({ reset: true });
else loadGroups();
