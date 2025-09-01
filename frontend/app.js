// Customized frontend/app.js for improved UX
// This script drives the classification UI, handles dark mode toggling,
// displays prediction results and manages the recent history panel.

const apiBase = ""; // same origin

// DOM helpers
const $ = (id) => document.getElementById(id);
const btn = $("classifyBtn");
const historyBtn = $("historyBtn");
const input = $("textInput");
const result = $("resultBox");
const badge = $("resultBadge");
const toast = $("toast");
const spinner = $("spinner");
const historyList = $("historyList");
const footerVersion = $("footerVersion");
const darkToggle = $("darkToggle");
const body = $("body");
const historyPanel = $("historyPanel");

// Show toast notification
function showToast(msg) {
  toast.textContent = msg;
  toast.classList.remove("hidden");
  setTimeout(() => toast.classList.add("hidden"), 3500);
}

// Render the badge based on label and confidence
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

// Classification function
async function classify() {
  const text = (input.value || "").trim();
  if (!text) return showToast("Please paste some text.");

  btn.disabled = true; btn.textContent = "Classifying...";
  spinner.classList.remove("hidden");
  result.textContent = ""; badge.classList.add("hidden");

  try {
    const res = await fetch(`${apiBase}/predict`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ text })
    });
    const payload = await res.json().catch(() => ({}));
    if (!res.ok) return showToast(`Error ${res.status}: ${payload.detail || "Prediction failed"}`);

    const { prediction, confidence, probabilities, model, runtime_seconds, timestamp } = payload;
    // Use an "unsure" label if confidence is near the boundary
    const label = (confidence >= 0.55) ? prediction : "Unsure";
    setBadge(label, confidence);
    result.textContent =
`Prediction  : ${prediction}
Confidence  : ${(confidence*100).toFixed(1)}%
Probabilities: ${JSON.stringify(probabilities)}
Model       : ${model?.hf_model_name || "-"}
Runtime     : ${runtime_seconds}s
When        : ${timestamp}`;
    // Update history panel if visible
    if (historyPanel.style.display !== "none") {
      await refreshHistory();
    }
  } catch (e) {
    showToast(`Network/JS error: ${String(e)}`);
  } finally {
    btn.disabled = false; btn.textContent = "Classify";
    spinner.classList.add("hidden");
  }
}

// Fetch and render recent history
async function refreshHistory() {
  try {
    const res = await fetch(`${apiBase}/history?limit=20`);
    const data = await res.json().catch(() => ({history: []}));
    historyList.innerHTML = "";
    (data.history || []).forEach(item => {
      const li = document.createElement("li");
      li.className = "border border-slate-200 dark:border-slate-600 rounded-xl p-3";
      li.innerHTML = `<div class="text-xs text-slate-500 dark:text-slate-400">${item.timestamp}</div>
        <div class="font-semibold">${item.prediction} <span class="text-slate-500 dark:text-slate-400">(${(item.confidence*100).toFixed(1)}%)</span></div>
        <div class="text-slate-700 dark:text-slate-300 mt-1 line-clamp-3">${item.input_preview}</div>`;
      historyList.appendChild(li);
    });
  } catch (e) {
    console.error(e);
  }
}

// Toggle history panel visibility
function toggleHistory() {
  if (historyPanel.style.display === "none" || !historyPanel.style.display) {
    historyPanel.style.display = "block";
    refreshHistory();
  } else {
    historyPanel.style.display = "none";
  }
}

// Initialize dark mode and toggle handler
function initDarkMode() {
  const saved = localStorage.getItem("dark");
  if (saved === "1") {
    document.documentElement.classList.add("dark");
  }
  darkToggle.addEventListener("click", () => {
    const on = document.documentElement.classList.toggle("dark");
    localStorage.setItem("dark", on ? "1" : "0");
  });
}

// Show model version info (optional /version endpoint)
async function showVersion() {
  try {
    const r = await fetch(`${apiBase}/version`);
    if (r.ok) {
      const j = await r.json();
      footerVersion.textContent = `Model loaded: ${j.clf_loaded ? "Yes" : "No"} · ${j.hf_model_name || "-"}`;
    }
  } catch {}
}

// Event listeners
btn.addEventListener("click", classify);
historyBtn.addEventListener("click", toggleHistory);
window.addEventListener("load", async () => {
  try { await fetch(`${apiBase}/health`); } catch {}
  initDarkMode();
  showVersion();
});