# src/inference.py
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np

# Where you saved the TF-IDF artifacts
MODEL_DIR = Path("models/lr_v1")
VEC_PATH = MODEL_DIR / "tfidf.joblib"
CLF_PATH = MODEL_DIR / "model.joblib"
LAB_PATH = MODEL_DIR / "labels.joblib"

# Lazy-loaded globals
_VEC = None
_CLF = None
_LABEL = None

def _load():
    global _VEC, _CLF, _LABEL
    if _VEC is None:
        _VEC = joblib.load(VEC_PATH)
    if _CLF is None:
        _CLF = joblib.load(CLF_PATH)
    if _LABEL is None:
        _LABEL = joblib.load(LAB_PATH)

def predict_text(text: str) -> Tuple[str, Dict[str, float], Dict]:
    """
    Return (label, {'AI': p_ai, 'Human': p_human}, meta)
    Pure TF-IDF + LogisticRegression (or similar scikit-learn classifier).
    """
    if not text or not text.strip():
        raise ValueError("Empty text")

    _load()
    t0 = time.time()
    X = _VEC.transform([text])

    # Predict probs if available
    if hasattr(_CLF, "predict_proba"):
        proba = _CLF.predict_proba(X)[0]  # aligned to _CLF.classes_ (encoded ints)
        cls_ids = np.array(_CLF.classes_)  # e.g., [0, 1] or [1, 0]
        # Map encoded ints back to string labels in the SAME order as proba columns
        label_names = _LABEL.inverse_transform(cls_ids)
        probs_map = {str(name): float(proba[i]) for i, name in enumerate(label_names)}
        # Ensure both keys exist
        ai_p = float(probs_map.get("AI", 0.0))
        human_p = float(probs_map.get("Human", 0.0))
        probs = {"AI": ai_p, "Human": human_p}
        label = "AI" if ai_p >= human_p else "Human"
    else:
        # Fallback for non-probabilistic models
        enc_pred = int(_CLF.predict(X)[0])
        pred_name = str(_LABEL.inverse_transform([enc_pred])[0])
        other = "AI" if pred_name == "Human" else "Human"
        probs = {pred_name: 0.9, other: 0.1}
        label = pred_name

    meta = {
        "model_name": "tfidf_lr_v1",
        "artifact_dir": str(MODEL_DIR),
        "runtime_seconds": round(time.time() - t0, 4),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    return label, probs, meta
