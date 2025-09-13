// minimal frontend — classify only
const apiBase = ""; // same origin

// DOM
const $ = (id) => document.getElementById(id);
const btn = $("classifyBtn");
const input = $("textInput");
const result = $("resultBox");
const badge = $("resultBadge");
const toast = $("toast");
const spinner = $("spinner");
const darkToggle = $("darkToggle");
const modelSelect = $("modelSelect");
const footerVersion = $("footerVersion");

function showToast(msg) {
  toast.textContent = msg;
  toast.classList.remove("hidden");
  setTimeout(() => toast.classList.add("hidden"), 3500);
}

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

  try {
    const res = await fetch(`${apiBase}/predict`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ text, backend })
    });

    const payload = await res.json().catch(() => ({}));
    if (!res.ok) return showToast(`Error ${res.status}: ${payload.detail || "Prediction failed"}`);

    const { prediction, confidence, probabilities, model, backend: usedBackend, runtime_seconds, timestamp } = payload;

    setBadge(prediction, confidence);
    result.textContent =
`Prediction  : ${prediction}
Confidence  : ${(confidence*100).toFixed(1)}%
Probabilities: ${JSON.stringify(probabilities)}
Model       : ${model?.hf_model_name || "-"}
Backend     : ${usedBackend || backend}
Runtime     : ${runtime_seconds}s
When        : ${timestamp}`;
  } catch (e) {
    showToast(`Network/JS error: ${String(e)}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "Classify";
    spinner.classList.add("hidden");
  }
}

function initDarkMode(){
  const saved = localStorage.getItem("dark");
  if (saved === "1") document.documentElement.classList.add("dark");
  darkToggle.addEventListener("click", () => {
    const on = document.documentElement.classList.toggle("dark");
    localStorage.setItem("dark", on ? "1" : "0");
  });
}

async function showVersion() {
  try {
    const r = await fetch(`${apiBase}/version`, { cache: "no-store" });
    if(r.ok) {
      const j = await r.json();
      footerVersion.textContent = `Model loaded: ${j.clf_loaded ? "Yes" : "No"} · ${j.hf_model_name||"-"}`;
    }
  } catch {}
}

btn.addEventListener("click", classify);
window.addEventListener("load", async () => {
  try { await fetch(`${apiBase}/health`); } catch {}
  initDarkMode();
  showVersion();
});
