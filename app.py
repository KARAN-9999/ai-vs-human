"""
FastAPI backend for AI vs Human classifier — minimal edition (no analytics/history).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple, Optional
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib

import src.inference as inf  # for transformer or LR fallback


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

UNSURE_THRESHOLD = float(os.getenv("UNSURE_THRESHOLD", "0.60"))

# ---- Load optional models (best-effort) ----
robust_lr_model = None
try:
    robust_lr_model = joblib.load(MODELS_DIR / "robust_baseline_model.joblib")
except Exception as e:
    print(f"[warn] Could not load robust logistic regression model: {e}")

old_lr_model = None
try:
    old_lr_model = joblib.load(MODELS_DIR / "lr_v1/model.joblib")
except Exception as e:
    print(f"[warn] Could not load original logistic regression model: {e}")

# ---- App setup ----
app = FastAPI(title="AI vs Human — API (minimal)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "frontend")), name="static")


class PredictIn(BaseModel):
    text: str
    # "auto", "lr" (robust), "old_lr", "transformer"
    backend: Optional[str] = "auto"


def predict_with_model(text: str, backend: str) -> Tuple[str, Dict[str, float], Dict]:
    """Route to the requested backend with graceful fallback."""
    start_time = time.time()
    backend = (backend or "auto").lower()

    # old LR (legacy)
    if backend == "old_lr" and old_lr_model is not None:
        probs = old_lr_model.predict_proba([text])[0]
        labels = old_lr_model.classes_
        prob_dict = {str(labels[i]): float(probs[i]) for i in range(len(labels))}
        label = "AI" if prob_dict.get("AI", 0.0) >= prob_dict.get("Human", 0.0) else "Human"
        meta = {"model_name": "original_lr_model", "backend": "old_lr"}

    # transformer
    elif backend == "transformer":
        label, prob_dict, meta = inf.predict_text(text, backend="transformer")

    # robust LR (preferred)
    elif backend in ("lr", "auto"):
        if robust_lr_model is not None:
            probs = robust_lr_model.predict_proba([text])[0]
            labels = robust_lr_model.classes_
            prob_dict = {str(labels[i]): float(probs[i]) for i in range(len(labels))}
            label = "AI" if prob_dict.get("AI", 0.0) >= prob_dict.get("Human", 0.0) else "Human"
            meta = {"model_name": "robust_baseline_model", "backend": "lr"}
        else:
            # fall back to inference auto-resolution (may choose transformer/LR)
            label, prob_dict, meta = inf.predict_text(text, backend="auto")

    else:
        # unknown backend: fall back to auto
        label, prob_dict, meta = inf.predict_text(text, backend="auto")

    meta["runtime_seconds"] = round(time.time() - start_time, 4)
    return label, prob_dict, meta


# ---------------- Routes ----------------
@app.get("/")
def root():
    return FileResponse(str(BASE_DIR / "frontend" / "index.html"))

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}

@app.get("/version")
def version():
    clf_loaded = robust_lr_model is not None
    hf_dir = os.getenv("TRANSFORMER_MODEL_DIR", "").strip()
    if hf_dir and Path(hf_dir).is_dir():
        hf_name = Path(hf_dir).name
    else:
        auto = next(iter([p for p in MODELS_DIR.glob("finetuned_*") if p.is_dir()]), None)
        hf_name = auto.name if auto else "-"
    return {"clf_loaded": clf_loaded, "hf_model_name": hf_name}

@app.post("/predict")
def predict(inp: PredictIn):
    txt = (inp.text or "").strip()
    if not txt:
        raise HTTPException(400, "Empty text")

    label, probs, meta = predict_with_model(txt, inp.backend or "auto")

    # normalize keys to always include AI/Human
    probs_norm = {"AI": float(probs.get("AI", 0.0)), "Human": float(probs.get("Human", 0.0))}
    confidence = float(max(probs_norm.values()))
    display_label = label if confidence >= UNSURE_THRESHOLD else "Unsure"
    now_iso = datetime.now(timezone.utc).isoformat()

    return {
        "prediction": display_label,
        "confidence": confidence,
        "probabilities": probs_norm,
        "model": {"hf_model_name": meta.get("model_name", "-")},
        "backend": meta.get("backend", inp.backend or "auto"),
        "runtime_seconds": meta.get("runtime_seconds", 0.0),
        "timestamp": now_iso,
    }
