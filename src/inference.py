# src/inference.py
from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

import joblib
import numpy as np

# -------------------------------
# Optional transformer imports
# -------------------------------
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    _TRANS_AVAILABLE = True
except Exception:
    _TRANS_AVAILABLE = False

# -------------------------------
# Config
# -------------------------------
MODELS_DIR = Path("models")

# LR artifacts
PIPELINE_PATH = MODELS_DIR / "tfidf_logreg.joblib"   # sklearn Pipeline
LR_DIR = MODELS_DIR / "lr_v1"
VEC_PATH = LR_DIR / "tfidf.joblib"
CLF_PATH = LR_DIR / "model.joblib"
LAB_PATH = LR_DIR / "labels.joblib"

# Transformer artifacts
TRANS_DIR_ENV = os.getenv("TRANSFORMER_MODEL_DIR", "")
MAX_LEN = int(os.getenv("TRANSFORMER_MAX_LEN", "256"))

# Backend selection: "auto", "lr", "transformer"
BACKEND_ENV = os.getenv("MODEL_BACKEND", "auto").strip().lower()

# -------------------------------
# Globals
# -------------------------------
_PIPE = None
_VEC = None
_CLF = None
_LABEL = None

_TOKENIZER = None
_TRANS_MODEL = None
_DEVICE = None
_TRANS_PATH = None

# -------------------------------
# Helpers
# -------------------------------
def _latest_finetuned_dir() -> Optional[Path]:
    """Pick the most recent models/finetuned_* directory."""
    candidates = [p for p in MODELS_DIR.glob("finetuned_*") if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

def _resolve_backend() -> str:
    if BACKEND_ENV in {"lr", "transformer"}:
        return BACKEND_ENV
    if _TRANS_AVAILABLE and (Path(TRANS_DIR_ENV).is_dir() or _latest_finetuned_dir()):
        return "transformer"
    if PIPELINE_PATH.exists() or (VEC_PATH.exists() and CLF_PATH.exists() and LAB_PATH.exists()):
        return "lr"
    return "lr"

# -------------------------------
# Loaders
# -------------------------------
def _load_lr():
    """Load LR pipeline or legacy artifacts."""
    global _PIPE, _VEC, _CLF, _LABEL
    if _PIPE is None and _VEC is None:
        if PIPELINE_PATH.exists():
            _PIPE = joblib.load(PIPELINE_PATH)
        else:
            _VEC = joblib.load(VEC_PATH)
            _CLF = joblib.load(CLF_PATH)
            _LABEL = joblib.load(LAB_PATH)

def _load_transformer():
    """Load Hugging Face transformer model + tokenizer."""
    global _TOKENIZER, _TRANS_MODEL, _DEVICE, _TRANS_PATH
    if _TOKENIZER is not None:
        return
    if not _TRANS_AVAILABLE:
        raise RuntimeError("transformers/torch not installed; cannot use transformer backend")

    if TRANS_DIR_ENV and Path(TRANS_DIR_ENV).is_dir():
        model_dir = Path(TRANS_DIR_ENV)
    else:
        model_dir = _latest_finetuned_dir()
    if not model_dir or not model_dir.exists():
        raise FileNotFoundError(
            "No finetuned_* transformer directory found in models/. "
            "Set TRANSFORMER_MODEL_DIR env var or run fine-tuning."
        )
    _TRANS_PATH = model_dir
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _TOKENIZER = AutoTokenizer.from_pretrained(str(model_dir))
    _TRANS_MODEL = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    _TRANS_MODEL.to(_DEVICE)
    _TRANS_MODEL.eval()

# -------------------------------
# Predictors
# -------------------------------
def _predict_lr(text: str) -> Tuple[str, Dict[str, float], Dict]:
    _load_lr()
    t0 = time.time()

    if _PIPE is not None:
        if hasattr(_PIPE, "predict_proba"):
            proba = _PIPE.predict_proba([text])[0]
            classes = list(_PIPE.classes_)
            probs_map = {str(c): float(proba[i]) for i, c in enumerate(classes)}
            ai_p = probs_map.get("AI", 0.0)
            human_p = probs_map.get("Human", 0.0)
            label = "AI" if ai_p >= human_p else "Human"
            probs = {"AI": ai_p, "Human": human_p}
        else:
            pred = str(_PIPE.predict([text])[0])
            other = "AI" if pred == "Human" else "Human"
            label, probs = pred, {pred: 0.9, other: 0.1}
        meta = {
            "backend": "lr",
            "variant": "pipeline",
            "model_path": str(PIPELINE_PATH),
            "runtime_seconds": round(time.time() - t0, 4),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        return label, probs, meta

    # legacy
    X = _VEC.transform([text])
    if hasattr(_CLF, "predict_proba"):
        proba = _CLF.predict_proba(X)[0]
        cls_ids = np.array(_CLF.classes_)
        label_names = _LABEL.inverse_transform(cls_ids)
        probs_map = {str(name): float(proba[i]) for i, name in enumerate(label_names)}
        ai_p = probs_map.get("AI", 0.0)
        human_p = probs_map.get("Human", 0.0)
        label = "AI" if ai_p >= human_p else "Human"
        probs = {"AI": ai_p, "Human": human_p}
    else:
        enc_pred = int(_CLF.predict(X)[0])
        pred_name = str(_LABEL.inverse_transform([enc_pred])[0])
        other = "AI" if pred_name == "Human" else "Human"
        label, probs = pred_name, {pred_name: 0.9, other: 0.1}

    meta = {
        "backend": "lr",
        "variant": "legacy",
        "artifact_dir": str(LR_DIR),
        "runtime_seconds": round(time.time() - t0, 4),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    return label, probs, meta

def _predict_transformer(text: str) -> Tuple[str, Dict[str, float], Dict]:
    _load_transformer()
    t0 = time.time()

    enc = _TOKENIZER(
        text,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(_DEVICE)

    with torch.no_grad():
        out = _TRANS_MODEL(**enc)
        logits = out.logits.squeeze(0)
        probs_t = torch.softmax(logits, dim=-1).detach().cpu().numpy()

    labels = ["Human", "AI"]  # consistent with training: 0=Human, 1=AI
    probs = {lbl: float(probs_t[i]) for i, lbl in enumerate(labels)}
    label = "AI" if probs["AI"] >= probs["Human"] else "Human"

    meta = {
        "backend": "transformer",
        "model_dir": str(_TRANS_PATH),
        "device": str(_DEVICE),
        "max_len": MAX_LEN,
        "runtime_seconds": round(time.time() - t0, 4),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    return label, probs, meta

# -------------------------------
# Public API
# -------------------------------
def predict_text(
    text: str,
    backend: Optional[str] = None,
) -> Tuple[str, Dict[str, float], Dict]:
    """
    Return (label, {'AI': p_ai, 'Human': p_human}, meta)
    - backend: 'lr', 'transformer', or 'auto' (default/ENV).
    """
    if not text or not text.strip():
        raise ValueError("Empty text")

    chosen = (backend or BACKEND_ENV or "auto").strip().lower()
    if chosen == "auto":
        chosen = _resolve_backend()

    if chosen == "lr":
        return _predict_lr(text)
    if chosen == "transformer":
        return _predict_transformer(text)

    return _predict_lr(text)
