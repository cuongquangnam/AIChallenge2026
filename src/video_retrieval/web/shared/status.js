/** Status line controller for task pages. */

export function createStatusController(statusEl) {
  return {
    set(message, isError = false) {
      if (!statusEl) return;
      statusEl.textContent = message || "";
      statusEl.classList.toggle("error", Boolean(isError));
    },
  };
}
