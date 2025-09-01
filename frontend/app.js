// frontend/app.js
const apiBase = ""; // same origin

const $ = (id) => document.getElementById(id);
const btn = $("classifyBtn");
const historyBtn = $("historyBtn");
const input = $("textInput");
const result = $("resultBox");
const badge = $("resultBadge");
const toast = $("toast");
const spinner = $("spinner");
const historyList = $("historyList");
const kpiTotal = $("kpiTotal");
const kpiAI = $("kpiAI");
const kpiHuman = $("kpiHuman");
const footerVersion = $("footerVersion");
const darkToggle = $("darkToggle");
const body = $("body");

let lineChart, barChart, histChart;

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
    badge.className = `${common} bg-sky-50 border-sky-200 text-sky-700`;
    badge.innerHTML = `AI <span class="text-slate-500">(${pct}%)</span>`;
  } else if (label === "Human") {
    badge.className = `${common} bg-emerald-50 border-emerald-200 text-emerald-700`;
    badge.innerHTML = `Human <span class="text-slate-500">(${pct}%)</span>`;
  } else {
    badge.className = `${common} bg-amber-50 border-amber-200 text-amber-700`;
    badge.textContent = "Unsure";
  }
}

function asPct(n) {
  return `${(n * 100).toFixed(0)}%`;
}

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

    // Near-boundary = Unsure
    const label = (confidence >= 0.55) ? prediction : "Unsure";
    setBadge(label, confidence);

    result.textContent =
`Prediction  : ${prediction}
Confidence  : ${(confidence*100).toFixed(1)}%
Probabilities: ${JSON.stringify(probabilities)}
Model       : ${model?.hf_model_name || "-"}
Runtime     : ${runtime_seconds}s
When        : ${timestamp}`;

    await refreshHistory();
    await refreshAnalytics();
  } catch (e) {
    showToast(`Network/JS error: ${String(e)}`);
  } finally {
    btn.disabled = false; btn.textContent = "Classify";
    spinner.classList.add("hidden");
  }
}

async function refreshHistory() {
  const res = await fetch(`${apiBase}/history?limit=20`);
  const data = await res.json().catch(() => ({history: []}));
  historyList.innerHTML = "";
  (data.history || []).forEach(item => {
    const li = document.createElement("li");
    li.className = "border border-slate-200 rounded-xl p-3";
    li.innerHTML = `<div class="text-xs text-slate-500">${item.timestamp}</div>
      <div class="font-semibold">${item.prediction} <span class="text-slate-500">(${(item.confidence*100).toFixed(1)}%)</span></div>
      <div class="text-slate-700 mt-1 line-clamp-3">${item.input_preview}</div>`;
    historyList.appendChild(li);
  });
}

function buildOrUpdateCharts(analytics) {
  const ctxL = document.getElementById("lineChart").getContext("2d");
  const ctxB = document.getElementById("barChart").getContext("2d");
  const ctxH = document.getElementById("histChart").getContext("2d");

  const labels = (analytics.last_50 || []).map(x => x.ts);
  const aiSeries = (analytics.last_50 || []).map(x => x.prediction === "AI" ? x.confidence : 0);
  const humanSeries = (analytics.last_50 || []).map(x => x.prediction === "Human" ? x.confidence : 0);

  if (lineChart) lineChart.destroy();
  lineChart = new Chart(ctxL, {
    type: "line",
    data: { labels, datasets: [{ label: "AI (last 50)", data: aiSeries }, { label: "Human (last 50)", data: humanSeries }] },
    options: { responsive: true, maintainAspectRatio: false }
  });

  if (barChart) barChart.destroy();
  barChart = new Chart(ctxB, {
    type: "bar",
    data: {
      labels: Object.keys(analytics.avg_confidence_by_label || {}),
      datasets: [{ label: "Avg confidence", data: Object.values(analytics.avg_confidence_by_label || {}) }]
    },
    options: { responsive: true, maintainAspectRatio: false }
  });

  if (histChart) histChart.destroy();
  histChart = new Chart(ctxH, {
    type: "bar",
    data: {
      labels: ["0–0.1","0.1–0.2","0.2–0.3","0.3–0.4","0.4–0.5","0.5–0.6","0.6–0.7","0.7–0.8","0.8–0.9","0.9–1.0"],
      datasets: [{ label: "Confidence histogram", data: analytics.confidence_histogram_bins || [] }]
    },
    options: { responsive: true, maintainAspectRatio: false }
  });

  // KPIs
  const totals = analytics.totals_by_label || {};
  const total = (totals.AI || 0) + (totals.Human || 0);
  kpiTotal.textContent = total;
  kpiAI.textContent = total ? asPct((totals.AI || 0)/total) : "0%";
  kpiHuman.textContent = total ? asPct((totals.Human || 0)/total) : "0%";
}

async function refreshAnalytics() {
  const res = await fetch(`${apiBase}/analytics`);
  const data = await res.json().catch(() => ({}));
  buildOrUpdateCharts(data);
}

function initDarkMode() {
  const saved = localStorage.getItem("dark");
  if (saved === "1") {
    document.documentElement.classList.add("dark");
    body.classList.replace("bg-slate-50","bg-slate-900");
    body.classList.replace("text-slate-900","text-slate-100");
  }
  darkToggle.addEventListener("click", () => {
    const on = document.documentElement.classList.toggle("dark");
    if (on) {
      body.classList.replace("bg-slate-50","bg-slate-900");
      body.classList.replace("text-slate-900","text-slate-100");
      localStorage.setItem("dark","1");
    } else {
      body.classList.replace("bg-slate-900","bg-slate-50");
      body.classList.replace("text-slate-100","text-slate-900");
      localStorage.setItem("dark","0");
    }
  });
}

async function showVersion() {
  try {
    const r = await fetch(`${apiBase}/version`);
    if (r.ok) {
      const j = await r.json();
      footerVersion.textContent = `Model loaded: ${j.clf_loaded ? "Yes" : "No"} · ${j.hf_model_name || "-"}`;
    }
  } catch {}
}

btn.addEventListener("click", classify);
historyBtn.addEventListener("click", refreshHistory);
window.addEventListener("load", async () => {
  try { await fetch(`${apiBase}/health`); } catch {}
  initDarkMode();
  await refreshHistory();
  await refreshAnalytics();
  await showVersion(); // optional if /version added
});
