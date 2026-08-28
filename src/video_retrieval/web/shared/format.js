/** Display helpers shared across search / KIS / QA / TRAKE pages. */

export function formatTime(sec) {
  if (typeof sec !== "number" || Number.isNaN(sec)) return null;
  const total = Math.max(0, Math.floor(sec));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

export function formatScore(score) {
  if (typeof score !== "number" || Number.isNaN(score)) return "—";
  return score.toFixed(3);
}

export function seekSeconds(hit) {
  if (typeof hit?.timestamp_sec === "number" && !Number.isNaN(hit.timestamp_sec)) {
    return Math.max(0, hit.timestamp_sec);
  }
  return 0;
}

export function hitImageSrc(hit) {
  return hit?.image_url || hit?.image_data_url || "";
}

export function joinMeta(parts) {
  return parts.filter(Boolean).join(" · ");
}

export function sanitizeQueryId(value, fallback = "query") {
  return String(value || fallback).trim().replace(/[^\w.-]+/g, "-") || fallback;
}
