// frontend/app.js
const btn = document.getElementById("btnClassify");
const btnHistory = document.getElementById("btnHistory");
const inputEl = document.getElementById("inputText");
const output = document.getElementById("output");
const resTitle = document.getElementById("resTitle");
const meta = document.getElementById("meta");
const probs = document.getElementById("probs");
const historyCard = document.getElementById("historyCard");
const historyList = document.getElementById("historyList");

btn.addEventListener("click", async () => {
  const text = inputEl.value.trim();
  if (!text) return alert("Add some text first");
  output.style.display = "none";
  resTitle.textContent = "Classifying…";
  try {
    const r = await fetch("/predict", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({text})
    });
    const data = await r.json();
    if (!r.ok) {
      alert(JSON.stringify(data));
      return;
    }
    output.style.display = "block";
    resTitle.textContent = `${data.prediction} (confidence ${Math.round(data.confidence*10000)/100}%)`;
    meta.innerHTML = `Model: ${data.model?.hf_model_name} • runtime: ${data.runtime_seconds}s • ${data.timestamp}`;
    probs.textContent = JSON.stringify(data.probabilities, null, 2);
  } catch (err) {
    alert("Request failed: " + err);
  }
});

btnHistory.addEventListener("click", async () => {
  try {
    const r = await fetch("/history?limit=20");
    const json = await r.json();
    historyCard.style.display = "block";
    historyList.innerHTML = "";
    json.history.forEach(item => {
      const d = document.createElement("div");
      d.style.marginBottom = "8px";
      d.innerHTML = `<strong>${item.prediction}</strong> • ${item.confidence.toFixed(3)} • ${item.timestamp}<div style="font-size:12px; color:#374151;">${item.input_preview}</div>`;
      historyList.appendChild(d);
    });
  } catch (err) {
    alert("Failed to load history: " + err);
  }
});
