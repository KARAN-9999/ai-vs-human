// Drives UI, dark mode, prediction, history, and analytics charts.

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
const historyPanel = $("historyPanel");

// KPI
const kpiTotal = $("kpiTotal");
const kpiAI = $("kpiAI");
const kpiHuman = $("kpiHuman");

// Charts
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
    const label = (confidence >= 0.55) ? prediction : "Unsure";
    setBadge(label, confidence);
    result.textContent =
`Prediction  : ${prediction}
Confidence  : ${(confidence*100).toFixed(1)}%
Probabilities: ${JSON.stringify(probabilities)}
Model       : ${model?.hf_model_name || "-"}
Runtime     : ${runtime_seconds}s
When        : ${timestamp}`;

    // Always refresh analytics; history only if visible
    await refreshAnalytics();
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

async function refreshHistory() {
  try {
    const res = await fetch(`${apiBase}/history?limit=20`, { cache: "no-store" });
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

function toggleHistory() {
  if (historyPanel.style.display === "none" || !historyPanel.style.display) {
    historyPanel.style.display = "block";
    refreshHistory();
    refreshAnalytics();
  } else {
    historyPanel.style.display = "none";
  }
}

function initDarkMode() {
  const saved = localStorage.getItem("dark");
  if (saved === "1") document.documentElement.classList.add("dark");
  $("darkToggle").addEventListener("click", () => {
    const on = document.documentElement.classList.toggle("dark");
    localStorage.setItem("dark", on ? "1" : "0");
  });
}

function buildOrUpdateCharts(analytics) {
  const ctxL = document.getElementById("lineChart").getContext("2d");
  const ctxB = document.getElementById("barChart").getContext("2d");
  const ctxH = document.getElementById("histChart").getContext("2d");

  // friendlier HH:MM:SS labels
  const labels = (analytics.last_50 || []).map(x =>
    (x.ts.includes("T") ? x.ts.split("T")[1].replace("Z","") : x.ts).slice(0,8)
  );
  const aiSeries    = (analytics.last_50 || []).map(x => x.prediction === "AI" ? x.confidence : 0);
  const humanSeries = (analytics.last_50 || []).map(x => x.prediction === "Human" ? x.confidence : 0);

  if (lineChart) lineChart.destroy();
  lineChart = new Chart(ctxL, {
    type: "line",
    data: { labels, datasets: [
      { label: "AI (last 50)", data: aiSeries, borderColor: "#0ea5e9", backgroundColor: "rgba(14,165,233,0.3)", tension: 0.25 },
      { label: "Human (last 50)", data: humanSeries, borderColor: "#10b981", backgroundColor: "rgba(16,185,129,0.3)", tension: 0.25 }
    ]},
    options: { responsive: true, maintainAspectRatio: false, animation: false }
  });

  if (barChart) barChart.destroy();
  barChart = new Chart(ctxB, {
    type: "bar",
    data: {
      labels: Object.keys(analytics.avg_confidence_by_label || {}),
      datasets: [{ label: "Avg confidence", data: Object.values(analytics.avg_confidence_by_label || {}), backgroundColor: ["#0ea5e9", "#10b981"] }]
    },
    options: { responsive: true, maintainAspectRatio: false, animation: false, scales: { y: { beginAtZero: true, max: 1 } } }
  });

  if (histChart) histChart.destroy();
  histChart = new Chart(ctxH, {
    type: "bar",
    data: {
      labels: ["0–0.1","0.1–0.2","0.2–0.3","0.3–0.4","0.4–0.5","0.5–0.6","0.6–0.7","0.7–0.8","0.8–0.9","0.9–1.0"],
      datasets: [{ label: "Confidence histogram", data: analytics.confidence_histogram_bins || [], backgroundColor: "#6366f1" }]
    },
    options: { responsive: true, maintainAspectRatio: false, animation: false, scales: { y: { beginAtZero: true } } }
  });

  const totals = analytics.totals_by_label || {};
  const total = (totals.AI || 0) + (totals.Human || 0);
  kpiTotal.textContent = total;
  kpiAI.textContent = total ? `${Math.round((totals.AI || 0)/total*100)}%` : "0%";
  kpiHuman.textContent = total ? `${Math.round((totals.Human || 0)/total*100)}%` : "0%";
}

async function refreshAnalytics() {
  try {
    const res = await fetch(`${apiBase}/analytics`, { cache: "no-store" });
    const data = await res.json().catch(() => ({}));
    buildOrUpdateCharts(data);
  } catch (e) {
    console.error(e);
  }
}

async function showVersion() {
  try {
    const r = await fetch(`${apiBase}/version`, { cache: "no-store" });
    if (r.ok) {
      const j = await r.json();
      footerVersion.textContent = `Model loaded: ${j.clf_loaded ? "Yes" : "No"} · ${j.hf_model_name || "-"}`;
    }
  } catch {}
}

btn.addEventListener("click", classify);
historyBtn.addEventListener("click", toggleHistory);

window.addEventListener("load", async () => {
  try { await fetch(`${apiBase}/health`); } catch {}
  initDarkMode();
  await refreshHistory();
  await refreshAnalytics();
  showVersion();
});

// Poll every 5s
setInterval(() => {
  refreshAnalytics();
  if (historyPanel.style.display !== "none") refreshHistory();
}, 5000);
