"""
Inference utilities for AI vs Human classification.

This script exposes a function ``predict_text`` that accepts a text
string and returns a predicted label, probability distribution and
metadata. It mirrors the upstream implementation but adjusts the
default classifier path to point to the combined dataset model
(``logreg_transformer_emb_best.joblib``) if present. If you train
additional classifiers and wish to use them instead, set the
``MODELS_DIR`` or ``CLF_PATH`` environment variables accordingly.

The embedding extraction uses a HuggingFace transformer; the model name
can be overridden with the ``HF_MODEL_NAME`` environment variable.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Dict, Tuple

try:
    import torch
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
    import joblib
except Exception:  # noqa: BLE001
    # If heavy deps are missing inference will fall back to defaults
    torch = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    AutoModel = None  # type: ignore
    AutoModelForSequenceClassification = None  # type: ignore
    joblib = None  # type: ignore


# --- Config ---
# Allow overriding of model names and paths via environment variables
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "distilroberta-base")
MODEL_DIR = os.getenv("MODELS_DIR", "models")
# Default classifier path for sklearn model (frozen embeddings)
DEFAULT_CLF = "logreg_transformer_emb_best.joblib"
CLF_PATH = os.path.join(MODEL_DIR, os.getenv("CLF_FILENAME", DEFAULT_CLF))
HF_CLASSIFIER_DIR = os.getenv("HF_CLASSIFIER_DIR")  # directory of a fine‑tuned HF model

EMB_CACHE: dict[str, object] = {}
LABELS = ["AI", "Human"]  # order matters for probability mapping

# Cache for the Hugging Face classifier model and tokenizer
HF_CLASSIFIER_CACHE: Dict[str, object] = {}


def _load_hf() -> Tuple[object | None, object | None]:
    """Load the HuggingFace tokenizer and model once and cache them."""
    if "tok" in EMB_CACHE and "hf" in EMB_CACHE:
        return EMB_CACHE["tok"], EMB_CACHE["hf"]
    if torch is None or AutoTokenizer is None or AutoModel is None:
        return None, None
    tok = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    hf = AutoModel.from_pretrained(HF_MODEL_NAME)
    hf.eval()
    EMB_CACHE["tok"], EMB_CACHE["hf"] = tok, hf
    return tok, hf


def _mean_pool(last_hidden_state, attention_mask):
    # Mean pooling over valid tokens
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    sum_vec = masked.sum(dim=1)
    lengths = mask.sum(dim=1).clamp(min=1e-9)
    return (sum_vec / lengths).detach().cpu().numpy()


def _embed(text: str):
    """Embed a single input string using the transformer encoder."""
    tok, hf = _load_hf()
    if tok is None or hf is None:
        return None
    enc = tok(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )
    with torch.no_grad():
        out = hf(**enc).last_hidden_state  # [B, T, H]
    emb = _mean_pool(out, enc["attention_mask"])  # [B, H]
    return emb  # numpy array


def _load_clf():
    """Load the sklearn classifier if available."""
    if joblib is None:
        return None
    if os.path.exists(CLF_PATH):
        try:
            return joblib.load(CLF_PATH)
        except Exception:
            return None
    return None


_CLF = _load_clf()


def _load_hf_classifier():
    """Load a fine‑tuned Hugging Face sequence classification model, if configured.

    If the environment variable ``HF_CLASSIFIER_DIR`` points to a directory containing
    ``config.json`` and ``pytorch_model.bin`` the model and tokenizer are loaded and
    cached. Returns (tokenizer, model) or (None, None) if unavailable.
    """
    if not HF_CLASSIFIER_DIR:
        return None, None
    if HF_CLASSIFIER_DIR in HF_CLASSIFIER_CACHE:
        return HF_CLASSIFIER_CACHE[HF_CLASSIFIER_DIR]
    if torch is None or AutoModelForSequenceClassification is None or AutoTokenizer is None:
        return None, None
    if not os.path.isdir(HF_CLASSIFIER_DIR):
        return None, None
    try:
        tok = AutoTokenizer.from_pretrained(HF_CLASSIFIER_DIR)
        clf = AutoModelForSequenceClassification.from_pretrained(HF_CLASSIFIER_DIR)
        clf.eval()
        HF_CLASSIFIER_CACHE[HF_CLASSIFIER_DIR] = (tok, clf)
        return tok, clf
    except Exception:
        return None, None


def predict_text(text: str) -> Tuple[str, Dict[str, float], Dict]:
    """Predict whether the input text is AI‑generated or Human.

    Returns a tuple containing the predicted label, a probability
    distribution over labels and a metadata dictionary.

    If the classifier or transformer model is not available the
    function falls back to a deterministic default distribution and
    returns "AI".
    """
    if not text or not text.strip():
        raise ValueError("Empty text")
    start = time.time()
    label = "AI"
    probs: Dict[str, float] = {"AI": 0.5, "Human": 0.5}
    # Attempt to use a fine‑tuned HF classifier if configured
    tok_hf_clf, clf_hf = _load_hf_classifier()
    if tok_hf_clf is not None and clf_hf is not None:
        try:
            enc = tok_hf_clf(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256,
            )
            with torch.no_grad():
                logits = clf_hf(**enc).logits  # [1, num_labels]
            prob_tensor = torch.nn.functional.softmax(logits, dim=-1)[0]
            # Determine index mapping based on model config (id2label)
            id2label = getattr(clf_hf.config, "id2label", {0: "Human", 1: "AI"})
            inv_map = {v: k for k, v in id2label.items()}
            ai_idx = inv_map.get("AI", 1)
            human_idx = inv_map.get("Human", 0)
            ai_p = float(prob_tensor[ai_idx])
            human_p = float(prob_tensor[human_idx])
            probs = {"AI": ai_p, "Human": human_p}
            label = "AI" if ai_p >= human_p else "Human"
        except Exception:
            # fallback to using sklearn classifier if HF classifier fails
            tok_hf_clf = None
            clf_hf = None

    if tok_hf_clf is None or clf_hf is None:
        # Use sklearn classifier on frozen embeddings if available
        if _CLF is not None:
            emb = _embed(text)
            if emb is not None:
                try:
                    if hasattr(_CLF, "predict_proba"):
                        p = _CLF.predict_proba(emb)[0]
                        # Map probabilities using classifier's classes_ attribute
                        if hasattr(_CLF, "classes_"):
                            cls_to_idx = {c: i for i, c in enumerate(_CLF.classes_)}
                            ai_p = float(p[cls_to_idx.get("AI", 1 if "AI" not in cls_to_idx else 0)])
                            human_p = float(p[cls_to_idx.get("Human", 0)])
                        else:
                            ai_p, human_p = float(p[1]), float(p[0])
                    else:
                        pred = int(_CLF.predict(emb)[0])
                        ai_p = 0.9 if pred == 1 else 0.1
                        human_p = 1 - ai_p
                    probs = {"AI": ai_p, "Human": human_p}
                    label = "AI" if probs["AI"] >= probs["Human"] else "Human"
                except Exception:
                    probs = {"AI": 0.6, "Human": 0.4}
                    label = "AI"
            else:
                # no embedding available (transformers not installed)
                probs = {"AI": 0.6, "Human": 0.4}
                label = "AI"
        else:
            # no classifier found yet; default fallback
            probs = {"AI": 0.55, "Human": 0.45}
            label = "AI"
    meta = {
        "model_name": HF_CLASSIFIER_DIR or HF_MODEL_NAME,
        "runtime_seconds": round(time.time() - start, 4),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    return label, probs, meta