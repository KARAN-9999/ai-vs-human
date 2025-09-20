# src/inference.py
from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List

import joblib
import numpy as np

# Optional transformer support
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from huggingface_hub import snapshot_download
    _TRANS_AVAILABLE = True
except Exception:
    _TRANS_AVAILABLE = False

# -----------------------------------------------------------------------------
# Paths & Config
# -----------------------------------------------------------------------------
MODELS_DIR = Path("models")

# Robust LR (single pipeline)
ROBUST_PIPE_CANDIDATES = [
    MODELS_DIR / "robust_baseline_model.joblib",
    MODELS_DIR / "tfidf_logreg.joblib",  # legacy fallback
]

# Old LR artifacts
LRV1_DIR = MODELS_DIR / "lr_v1"
LRV1_VEC = LRV1_DIR / "tfidf.joblib"
LRV1_CLF = LRV1_DIR / "model.joblib"
LRV1_LAB = LRV1_DIR / "labels.joblib"
LRV1_CAL = LRV1_DIR / "calibration.joblib"

# Transformer model ID
HUGGINGFACE_MODEL_ID = "Karan-09/ai-vs-human-transformer"
MAX_LEN = int(os.getenv("TRANSFORMER_MAX_LEN", "256"))

# Backend: 'auto' | 'lr' | 'old_lr' | 'transformer' | 'ensemble'
BACKEND_ENV = os.getenv("MODEL_BACKEND", "auto").strip().lower()

# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------
_PIPE_ROBUST: Optional[Any] = None
_V1_VEC: Optional[Any] = None
_V1_CLF: Optional[Any] = None
_V1_LAB: Optional[Any] = None
_V1_CAL: Optional[Any] = None

_TOKENIZER = None
_TRANS_MODEL = None
_DEVICE = None

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _classes_to_probs(proba_vec, classes_like) -> Dict[str, float]:
    probs = {"AI": 0.0, "Human": 0.0}
    for i, c in enumerate(list(classes_like)):
        name = str(c)
        if name.lower() == "ai":
            probs["AI"] = float(proba_vec[i])
        elif name.lower() == "human":
            probs["Human"] = float(proba_vec[i])
    return probs


def _decode_label_idx(idx: int, label_obj) -> str:
    try:
        if hasattr(label_obj, "inverse_transform"):
            return str(label_obj.inverse_transform([idx])[0])
        if isinstance(label_obj, (list, tuple, np.ndarray)):
            return str(label_obj[idx])
    except Exception:
        pass
    return "Human"


def _choose_label(probs: Dict[str, float]) -> str:
    return "AI" if probs.get("AI", 0.0) >= probs.get("Human", 0.0) else "Human"


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------
def _load_robust_pipeline():
    global _PIPE_ROBUST
    if _PIPE_ROBUST is not None:
        return
    for p in ROBUST_PIPE_CANDIDATES:
        if p.exists():
            _PIPE_ROBUST = joblib.load(p)
            return


def _load_old_lr():
    global _V1_VEC, _V1_CLF, _V1_LAB, _V1_CAL
    if _V1_VEC is not None:
        return
    if LRV1_VEC.exists() and LRV1_CLF.exists() and LRV1_LAB.exists():
        _V1_VEC = joblib.load(LRV1_VEC)
        _V1_CLF = joblib.load(LRV1_CLF)
        _V1_LAB = joblib.load(LRV1_LAB)
        if LRV1_CAL.exists():
            try:
                _V1_CAL = joblib.load(LRV1_CAL)
            except Exception:
                _V1_CAL = None


def _load_transformer(prewarm: bool = False):
    """Load transformer from Hugging Face Hub (cached in Render disk)."""
    global _TOKENIZER, _TRANS_MODEL, _DEVICE
    if _TOKENIZER is not None:
        return
    if not _TRANS_AVAILABLE:
        raise RuntimeError("transformers/torch not installed; cannot use transformer backend")

    # ensure cached snapshot (avoids download every deploy)
    snapshot_download(repo_id=HUGGINGFACE_MODEL_ID, local_dir="models/hf-cache", ignore_patterns=["*.msgpack"])

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _TOKENIZER = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL_ID, cache_dir="models/hf-cache")
    _TRANS_MODEL = AutoModelForSequenceClassification.from_pretrained(
        HUGGINGFACE_MODEL_ID, cache_dir="models/hf-cache"
    )
    _TRANS_MODEL.to(_DEVICE)
    _TRANS_MODEL.eval()

    # Optional prewarm to avoid first-request slowness
    if prewarm:
        dummy = _TOKENIZER("warmup", return_tensors="pt").to(_DEVICE)
        with torch.no_grad():
            _ = _TRANS_MODEL(**dummy)


# -----------------------------------------------------------------------------
# Predictors
# -----------------------------------------------------------------------------
def _predict_robust_lr(text: str) -> Tuple[str, Dict[str, float], Dict]:
    _load_robust_pipeline()
    if _PIPE_ROBUST is None:
        raise RuntimeError("Robust LR pipeline not found.")
    t0 = time.time()
    if hasattr(_PIPE_ROBUST, "predict_proba"):
        proba = _PIPE_ROBUST.predict_proba([text])[0]
        classes = getattr(_PIPE_ROBUST, "classes_", ["Human", "AI"])
        probs = _classes_to_probs(proba, classes)
    else:
        pred = str(_PIPE_ROBUST.predict([text])[0])
        other = "AI" if pred == "Human" else "Human"
        probs = {pred: 0.9, other: 0.1}
    label = _choose_label(probs)
    return label, probs, {"backend": "lr", "runtime_seconds": round(time.time() - t0, 4), "timestamp": _now_iso()}


def _predict_old_lr(text: str) -> Tuple[str, Dict[str, float], Dict]:
    _load_old_lr()
    if _V1_VEC is None:
        raise RuntimeError("Old LR artifacts not found in models/lr_v1.")
    t0 = time.time()
    X = _V1_VEC.transform([text])
    if hasattr(_V1_CLF, "predict_proba"):
        proba = _V1_CLF.predict_proba(X)[0]
        classes = getattr(_V1_CLF, "classes_", [0, 1])
        probs = _classes_to_probs(proba, classes)
    else:
        enc_pred = int(_V1_CLF.predict(X)[0])
        pred_name = _decode_label_idx(enc_pred, _V1_LAB)
        other = "AI" if pred_name == "Human" else "Human"
        probs = {pred_name: 0.9, other: 0.1}
    label = _choose_label(probs)
    return label, probs, {"backend": "old_lr", "runtime_seconds": round(time.time() - t0, 4), "timestamp": _now_iso()}


def _predict_transformer(text: str) -> Tuple[str, Dict[str, float], Dict]:
    _load_transformer()
    t0 = time.time()
    enc = _TOKENIZER(text, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(_DEVICE)
    with torch.no_grad():
        out = _TRANS_MODEL(**enc)
        logits = out.logits.squeeze(0)
        probs_t = torch.softmax(logits, dim=-1).cpu().numpy()
    labels = ["Human", "AI"]  # index 0=Human, 1=AI
    probs = {lbl: float(probs_t[i]) for i, lbl in enumerate(labels)}
    label = _choose_label(probs)
    return label, probs, {"backend": "transformer", "device": str(_DEVICE), "runtime_seconds": round(time.time() - t0, 4), "timestamp": _now_iso()}


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def _resolve_backend_auto() -> str:
    if _TRANS_AVAILABLE:
        return "transformer"
    for p in ROBUST_PIPE_CANDIDATES:
        if p.exists():
            return "lr"
    if LRV1_VEC.exists() and LRV1_CLF.exists() and LRV1_LAB.exists():
        return "old_lr"
    return "lr"


def predict_text(text: str, backend: Optional[str] = None) -> Tuple[str, Dict[str, float], Dict]:
    if not text or not text.strip():
        raise ValueError("Empty text")
    chosen = (backend or BACKEND_ENV or "auto").strip().lower()
    if chosen == "auto":
        chosen = _resolve_backend_auto()
    if chosen in ("lr", "robust_lr"):
        return _predict_robust_lr(text)
    if chosen in ("old_lr", "lr_v1"):
        return _predict_old_lr(text)
    if chosen == "transformer":
        return _predict_transformer(text)
    return _predict_robust_lr(text)


# -----------------------------------------------------------------------------
# Warmup transformer at startup
# -----------------------------------------------------------------------------
if BACKEND_ENV in ("transformer", "auto", "ensemble") and _TRANS_AVAILABLE:
    try:
        _load_transformer(prewarm=True)
        print("[inference] Transformer model preloaded & warmed up")
    except Exception as e:
        print(f"[inference] Failed to preload transformer: {e}")
