import { formatTime, hitImageSrc, seekSeconds } from "./format.js";

/**
 * Modal lightbox for keyframe preview and optional in-video playback.
 * KIS/search pages can wire add-frame capture via setOnVideoActivity.
 */
export function createLightboxController({
  dialog,
  img,
  video,
  meta,
  closeBtn,
  openVideoBtn = null,
  addFrameBtn = null,
}) {
  let activeHit = null;
  let onVideoActivity = null;

  function notifyVideoActivity() {
    if (onVideoActivity) onVideoActivity(activeHit);
  }

  function stopVideo() {
    video.pause();
    video.removeAttribute("src");
    video.load();
    video.hidden = true;
    img.hidden = false;
    if (openVideoBtn) {
      openVideoBtn.hidden = !activeHit?.video_url;
    }
    if (addFrameBtn) {
      addFrameBtn.hidden = true;
    }
    notifyVideoActivity();
  }

  function showKeyframe(hit) {
    stopVideo();
    const src = hitImageSrc(hit);
    if (src) {
      img.src = src;
      img.alt = `${hit.video_id} frame ${hit.frame_index ?? ""}`;
      img.hidden = false;
    } else {
      img.removeAttribute("src");
      img.alt = "No keyframe image";
      img.hidden = true;
    }
    if (openVideoBtn) {
      openVideoBtn.hidden = !hit.video_url;
      openVideoBtn.textContent = hit.video_url
        ? `Play from ${formatTime(seekSeconds(hit)) || "0:00"}`
        : "Video unavailable";
    }
    notifyVideoActivity();
  }

  function playAtTime(hit, timestampSec = seekSeconds(hit)) {
    if (!hit?.video_url) return false;
    img.hidden = true;
    video.hidden = false;
    if (openVideoBtn) openVideoBtn.hidden = true;

    const t = Math.max(0, timestampSec);
    const onReady = () => {
      try {
        video.currentTime = t;
      } catch {
        /* ignore seek race */
      }
      video.play().catch(() => {});
      notifyVideoActivity();
    };

    video.onloadedmetadata = onReady;
    const nextUrl = new URL(hit.video_url, window.location.origin).href;
    if (video.src !== nextUrl) {
      video.src = hit.video_url;
      video.load();
    } else if (video.readyState >= 1) {
      onReady();
    } else {
      video.addEventListener("loadedmetadata", onReady, { once: true });
    }
    notifyVideoActivity();
    return true;
  }

  function open(hit, metaParts = null) {
    activeHit = hit;
    if (metaParts) {
      meta.textContent = metaParts.filter(Boolean).join(" · ");
    } else {
      meta.textContent = [
        hit.video_id,
        hit.event_id,
        hit.frame_index != null ? `f${hit.frame_index}` : null,
        hit.frame_id != null ? `f${hit.frame_id}` : null,
        formatTime(hit.timestamp_sec),
        hit.answer ? `ans ${hit.answer}` : null,
      ]
        .filter(Boolean)
        .join(" · ");
    }
    showKeyframe(hit);
    if (!dialog.open) dialog.showModal();
    return hit;
  }

  function close() {
    stopVideo();
    activeHit = null;
    if (dialog.open) dialog.close();
  }

  function bind({ onBackdropClose = true } = {}) {
    closeBtn?.addEventListener("click", close);
    openVideoBtn?.addEventListener("click", () => {
      if (activeHit) playAtTime(activeHit);
    });
    dialog?.addEventListener("close", stopVideo);
    if (onBackdropClose && dialog) {
      dialog.addEventListener("click", (event) => {
        if (event.target === dialog) close();
      });
    }
    video?.addEventListener("play", notifyVideoActivity);
    video?.addEventListener("loadeddata", notifyVideoActivity);
  }

  return {
    open,
    close,
    playAtTime,
    stopVideo,
    getActiveHit: () => activeHit,
    setOnVideoActivity(fn) {
      onVideoActivity = fn;
    },
    bind,
  };
}
