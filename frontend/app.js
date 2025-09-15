// Minimal frontend — now with explainability and responsive improvements

const apiBase = ""; // same origin

// DOM shorthand
const $ = (id) => document.getElementById(id);

// Elements
const btn = $("classifyBtn");
const input = $("textInput");
const result = $("resultBox");
const badge = $("resultBadge");
const toast = $("toast");
const spinner = $("spinner");
const darkToggle = $("darkToggle");
const modelSelect = $("modelSelect");
const footerVersion = $("footerVersion");
const toggleExplain = $("toggleExplain");
const explainBox = $("explainBox");

// Show a temporary toast message
function showToast(msg) {
  toast.textContent = msg;
  toast.classList.remove("hidden");
  setTimeout(() => toast.classList.add("hidden"), 3500);
}

// Set the result badge style and text
function setBadge(label, confidence) {
  badge.classList.remove("hidden");
  const pct = (confidence * 100).toFixed(1);
  const common = "inline-flex items-center gap-2 px-3 py-1 rounded-xl text-sm font-semibold border";
  if (label === "AI") {
    badge.className = `${common} bg-sky-50 dark:bg-sky-200 border-sky-200 dark:border-sky-300 text-sky-700`;
    badge.innerHTML = `AI <span class="text-slate-500 dark:text-slate-400">(${pct}%)</span>`;
  } else if (label === "Human") {
    badge.className = `${common} bg-emerald-50 dark:bg-emerald-200 border-emerald-200 dark:border-emerald-300 text-emerald-700`;
    badge.innerHTML = `Human <span class="text-slate-500 dark:text-slate-400">(${pct}%)</span>`;
  } else {
    badge.className = `${common} bg-amber-50 dark:bg-amber-200 border-amber-200 dark:border-amber-300 text-amber-700`;
    badge.textContent = "Unsure";
  }
}

// Escape HTML special characters for safe rendering
function escapeHTML(str) {
  return str.replace(/[&<>"']/g, (m) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[m]));
}

// Convert a weight value into a CSS rgba string.
// Positive values favour AI and are tinted red; negative favour Human and are tinted blue.
function weightToRGBA(w) {
  const alpha = Math.min(0.85, Math.max(0.1, Math.abs(w)));
  if (w >= 0) {
    return `rgba(255, 0, 0, ${alpha})`;
  }
  return `rgba(0, 128, 255, ${alpha})`;
}

// Render highlighted spans into explainBox
function renderHighlights(text, spans) {
  let html = "";
  let cursor = 0;
  // sort spans by start position
  spans.sort((a, b) => a.start - b.start);
  for (const s of spans) {
    const start = Math.max(0, Math.min(text.length, s.start));
    const end = Math.max(start, Math.min(text.length, s.end));
    if (start > cursor) {
      html += escapeHTML(text.slice(cursor, start));
    }
    const seg = text.slice(start, end);
    const color = weightToRGBA(s.weight);
    html += `<span style="background:${color}" class="rounded"">${escapeHTML(seg)}</span>`;
    cursor = end;
  }
  if (cursor < text.length) {
    html += escapeHTML(text.slice(cursor));
  }
  explainBox.innerHTML = html;
  explainBox.classList.remove("hidden");
}

// Fetch explanation data from the backend
async function fetchExplanation(text, backend) {
  const res = await fetch(`${apiBase}/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, backend }),
  });
  if (!res.ok) {
    throw new Error(`Explain request failed with ${res.status}`);
  }
  const data = await res.json();
  return data;
}

// Main classify action
async function classify() {
  const text = (input.value || "").trim();
  if (!text) return showToast("Please paste some text.");
  if (text.length < 20) showToast("Very short inputs may be unreliable.");

  const backend = (modelSelect?.value || "auto");

  btn.disabled = true;
  btn.textContent = "Classifying...";
  spinner.classList.remove("hidden");
  result.textContent = "";
  badge.classList.add("hidden");
  explainBox.classList.add("hidden");

  try {
    const res = await fetch(`${apiBase}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, backend }),
    });
    const payload = await res.json().catch(() => ({}));
    if (!res.ok) return showToast(`Error ${res.status}: ${payload.detail || "Prediction failed"}`);

    const {
      prediction,
      confidence,
      probabilities,
      model,
      backend: usedBackend,
      runtime_seconds,
      timestamp,
    } = payload;
    const formattedProbs = {};
      for (const key in probabilities) {
        const val = probabilities[key];
        formattedProbs[key] = Number(val.toFixed(2));
  }
    setBadge(prediction, confidence);
    result.textContent =
`Prediction  : ${prediction}
Confidence  : ${(confidence * 100).toFixed(1)}%
Probabilities: ${JSON.stringify(formattedProbs)}
Model       : ${model?.hf_model_name || "-"}
Backend     : ${usedBackend || backend}
Runtime     : ${runtime_seconds}s
When        : ${timestamp}`;

    // fetch and display explanation if toggled
    if (toggleExplain && toggleExplain.checked) {
      try {
        const exp = await fetchExplanation(text, backend);
        if (Array.isArray(exp.spans) && exp.spans.length > 0) {
          renderHighlights(text, exp.spans);
        } else {
          explainBox.classList.add("hidden");
        }
      } catch (e) {
        console.warn("Explanation failed:", e);
        explainBox.classList.add("hidden");
      }
    }
  } catch (e) {
    showToast(`Network/JS error: ${String(e)}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "Classify";
    spinner.classList.add("hidden");
  }
}

// Initialise dark mode toggler
function initDarkMode() {
  const saved = localStorage.getItem("dark");
  if (saved === "1") document.documentElement.classList.add("dark");
  darkToggle.addEventListener("click", () => {
    const on = document.documentElement.classList.toggle("dark");
    localStorage.setItem("dark", on ? "1" : "0");
  });
}

// Show version info in footer
async function showVersion() {
  try {
    const r = await fetch(`${apiBase}/version`, { cache: "no-store" });
    if (r.ok) {
      const j = await r.json();
      footerVersion.textContent = `Model loaded: ${j.clf_loaded ? "Yes" : "No"} · ${j.hf_model_name || "-"}`;
    }
  } catch {}
}

// Set up event listeners
btn.addEventListener("click", classify);
// When toggling explanation checkbox, re-run explanation if prediction exists
if (toggleExplain) {
  toggleExplain.addEventListener("change", async () => {
    if (toggleExplain.checked) {
      const text = (input.value || "").trim();
      const backend = (modelSelect?.value || "auto");
      if (text) {
        try {
          const exp = await fetchExplanation(text, backend);
          if (Array.isArray(exp.spans) && exp.spans.length > 0) {
            renderHighlights(text, exp.spans);
          } else {
            explainBox.classList.add("hidden");
          }
        } catch {
          explainBox.classList.add("hidden");
        }
      }
    } else {
      explainBox.classList.add("hidden");
    }
  });
}

window.addEventListener("load", async () => {
  try { await fetch(`${apiBase}/health`); } catch {}
  initDarkMode();
  showVersion();
});