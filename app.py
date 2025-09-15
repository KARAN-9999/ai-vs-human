# """
# FastAPI backend for AI vs Human classifier — minimal edition (no analytics/history).
# """

# from __future__ import annotations

# import os
# from datetime import datetime, timezone
# from pathlib import Path
# from typing import Dict, Tuple, Optional
# import time

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import FileResponse
# from pydantic import BaseModel
# import joblib

# import src.inference as inf  # for transformer or LR fallback


# BASE_DIR = Path(__file__).resolve().parent
# MODELS_DIR = BASE_DIR / "models"

# UNSURE_THRESHOLD = float(os.getenv("UNSURE_THRESHOLD", "0.60"))

# # ---- Load optional models (best-effort) ----
# robust_lr_model = None
# try:
#     robust_lr_model = joblib.load(MODELS_DIR / "robust_baseline_model.joblib")
# except Exception as e:
#     print(f"[warn] Could not load robust logistic regression model: {e}")

# old_lr_model = None
# try:
#     old_lr_model = joblib.load(MODELS_DIR / "lr_v1/model.joblib")
# except Exception as e:
#     print(f"[warn] Could not load original logistic regression model: {e}")

# # ---- App setup ----
# app = FastAPI(title="AI vs Human — API (minimal)")
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # tighten for production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# app.mount("/static", StaticFiles(directory=str(BASE_DIR / "frontend")), name="static")


# class PredictIn(BaseModel):
#     text: str
#     # "auto", "lr" (robust), "old_lr", "transformer"
#     backend: Optional[str] = "auto"


# def predict_with_model(text: str, backend: str) -> Tuple[str, Dict[str, float], Dict]:
#     """Route to the requested backend with graceful fallback."""
#     start_time = time.time()
#     backend = (backend or "auto").lower()

#     # old LR (legacy)
#     if backend == "old_lr" and old_lr_model is not None:
#         probs = old_lr_model.predict_proba([text])[0]
#         labels = old_lr_model.classes_
#         prob_dict = {str(labels[i]): float(probs[i]) for i in range(len(labels))}
#         label = "AI" if prob_dict.get("AI", 0.0) >= prob_dict.get("Human", 0.0) else "Human"
#         meta = {"model_name": "original_lr_model", "backend": "old_lr"}

#     # transformer
#     elif backend == "transformer":
#         label, prob_dict, meta = inf.predict_text(text, backend="transformer")

#     # robust LR (preferred)
#     elif backend in ("lr", "auto"):
#         if robust_lr_model is not None:
#             probs = robust_lr_model.predict_proba([text])[0]
#             labels = robust_lr_model.classes_
#             prob_dict = {str(labels[i]): float(probs[i]) for i in range(len(labels))}
#             label = "AI" if prob_dict.get("AI", 0.0) >= prob_dict.get("Human", 0.0) else "Human"
#             meta = {"model_name": "robust_baseline_model", "backend": "lr"}
#         else:
#             # fall back to inference auto-resolution (may choose transformer/LR)
#             label, prob_dict, meta = inf.predict_text(text, backend="auto")

#     else:
#         # unknown backend: fall back to auto
#         label, prob_dict, meta = inf.predict_text(text, backend="auto")

#     meta["runtime_seconds"] = round(time.time() - start_time, 4)
#     return label, prob_dict, meta


# # ---------------- Routes ----------------
# @app.get("/")
# def root():
#     return FileResponse(str(BASE_DIR / "frontend" / "index.html"))

# @app.get("/health")
# def health():
#     return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}

# @app.get("/version")
# def version():
#     clf_loaded = robust_lr_model is not None
#     hf_dir = os.getenv("TRANSFORMER_MODEL_DIR", "").strip()
#     if hf_dir and Path(hf_dir).is_dir():
#         hf_name = Path(hf_dir).name
#     else:
#         auto = next(iter([p for p in MODELS_DIR.glob("finetuned_*") if p.is_dir()]), None)
#         hf_name = auto.name if auto else "-"
#     return {"clf_loaded": clf_loaded, "hf_model_name": hf_name}

# @app.post("/predict")
# def predict(inp: PredictIn):
#     txt = (inp.text or "").strip()
#     if not txt:
#         raise HTTPException(400, "Empty text")

#     label, probs, meta = predict_with_model(txt, inp.backend or "auto")

#     # normalize keys to always include AI/Human
#     probs_norm = {"AI": float(probs.get("AI", 0.0)), "Human": float(probs.get("Human", 0.0))}
#     confidence = float(max(probs_norm.values()))
#     display_label = label if confidence >= UNSURE_THRESHOLD else "Unsure"
#     now_iso = datetime.now(timezone.utc).isoformat()

#     return {
#         "prediction": display_label,
#         "confidence": confidence,
#         "probabilities": probs_norm,
#         "model": {"hf_model_name": meta.get("model_name", "-")},
#         "backend": meta.get("backend", inp.backend or "auto"),
#         "runtime_seconds": meta.get("runtime_seconds", 0.0),
#         "timestamp": now_iso,
#     }
# """
# FastAPI backend for AI vs Human classifier — extended edition with explainability.
# This module exposes prediction and explanation endpoints that serve a simple
# frontend for classifying text as AI‑generated or human‑written.

# The `/predict` route returns the predicted label and probability scores for the
# requested backend (logistic regression, transformer, or auto selection).  The
# new `/explain` route computes token‑level importance scores using either LIME
# for linear models or Integrated Gradients via Captum for transformer models.

# Weights are normalized to lie in the range −1..1 so the frontend can map them
# to colour intensities.  Positive values indicate evidence for the “AI” class
# (see `polarity`), while negative values indicate evidence for the “Human” class.
# The response includes a list of character spans and corresponding weights to
# highlight text on the client side.
# """

# from __future__ import annotations

# import os
# from datetime import datetime, timezone
# from pathlib import Path
# from typing import Dict, Tuple, Optional, List, Any
# import time

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import FileResponse
# from pydantic import BaseModel
# import joblib

# import src.inference as inf  # for transformer or LR fallback

# # Base paths
# BASE_DIR = Path(__file__).resolve().parent
# MODELS_DIR = BASE_DIR / "models"

# UNSURE_THRESHOLD = float(os.getenv("UNSURE_THRESHOLD", "0.60"))

# # ---- Load optional models (best‑effort) ----
# robust_lr_model = None
# try:
#     robust_lr_model = joblib.load(MODELS_DIR / "robust_baseline_model.joblib")
# except Exception as e:
#     print(f"[warn] Could not load robust logistic regression model: {e}")

# old_lr_model = None
# try:
#     old_lr_model = joblib.load(MODELS_DIR / "lr_v1/model.joblib")
# except Exception as e:
#     print(f"[warn] Could not load original logistic regression model: {e}")

# # ---- App setup ----
# app = FastAPI(title="AI vs Human — API (extended)")
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # tighten for production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# app.mount("/static", StaticFiles(directory=str(BASE_DIR / "frontend")), name="static")


# class PredictIn(BaseModel):
#     text: str
#     # "auto", "lr" (robust), "old_lr", "transformer"
#     backend: Optional[str] = "auto"


# def predict_with_model(text: str, backend: str) -> Tuple[str, Dict[str, float], Dict[str, Any]]:
#     """Route to the requested backend with graceful fallback."""
#     start_time = time.time()
#     backend = (backend or "auto").lower()

#     # old LR (legacy)
#     if backend == "old_lr" and old_lr_model is not None:
#         probs = old_lr_model.predict_proba([text])[0]
#         labels = old_lr_model.classes_
#         prob_dict = {str(labels[i]): float(probs[i]) for i in range(len(labels))}
#         label = "AI" if prob_dict.get("AI", 0.0) >= prob_dict.get("Human", 0.0) else "Human"
#         meta = {"model_name": "original_lr_model", "backend": "old_lr"}

#     # transformer
#     elif backend == "transformer":
#         label, prob_dict, meta = inf.predict_text(text, backend="transformer")

#     # robust LR (preferred)
#     elif backend in ("lr", "auto"):
#         if robust_lr_model is not None:
#             probs = robust_lr_model.predict_proba([text])[0]
#             labels = robust_lr_model.classes_
#             prob_dict = {str(labels[i]): float(probs[i]) for i in range(len(labels))}
#             label = "AI" if prob_dict.get("AI", 0.0) >= prob_dict.get("Human", 0.0) else "Human"
#             meta = {"model_name": "robust_baseline_model", "backend": "lr"}
#         else:
#             # fall back to inference auto‑resolution (may choose transformer/LR)
#             label, prob_dict, meta = inf.predict_text(text, backend="auto")

#     else:
#         # unknown backend: fall back to auto
#         label, prob_dict, meta = inf.predict_text(text, backend="auto")

#     meta["runtime_seconds"] = round(time.time() - start_time, 4)
#     return label, prob_dict, meta


# class Span(BaseModel):
#     """A highlighted span with start/end indices and a weight."""
#     start: int
#     end: int
#     weight: float


# class ExplainIn(BaseModel):
#     """Request body for explanation endpoint."""
#     text: str
#     backend: Optional[str] = "auto"


# class ExplainOut(BaseModel):
#     """Response body for explanation endpoint."""
#     pred_label: str
#     confidence: float
#     spans: List[Span]
#     polarity: str
#     meta: Dict[str, Any]


# @app.get("/")
# def root() -> FileResponse:
#     return FileResponse(str(BASE_DIR / "frontend" / "index.html"))


# @app.get("/health")
# def health():
#     return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}


# @app.get("/version")
# def version():
#     clf_loaded = robust_lr_model is not None
#     hf_dir = os.getenv("TRANSFORMER_MODEL_DIR", "").strip()
#     if hf_dir and Path(hf_dir).is_dir():
#         hf_name = Path(hf_dir).name
#     else:
#         auto = next(iter([p for p in MODELS_DIR.glob("finetuned_*") if p.is_dir()]), None)
#         hf_name = auto.name if auto else "-"
#     return {"clf_loaded": clf_loaded, "hf_model_name": hf_name}


# @app.post("/predict")
# def predict(inp: PredictIn):
#     txt = (inp.text or "").strip()
#     if not txt:
#         raise HTTPException(400, "Empty text")

#     label, probs, meta = predict_with_model(txt, inp.backend or "auto")

#     # normalize keys to always include AI/Human
#     probs_norm = {"AI": float(probs.get("AI", 0.0)), "Human": float(probs.get("Human", 0.0))}
#     confidence = float(max(probs_norm.values()))
#     display_label = label if confidence >= UNSURE_THRESHOLD else "Unsure"
#     now_iso = datetime.now(timezone.utc).isoformat()

#     return {
#         "prediction": display_label,
#         "confidence": confidence,
#         "probabilities": probs_norm,
#         "model": {"hf_model_name": meta.get("model_name", "-")},
#         "backend": meta.get("backend", inp.backend or "auto"),
#         "runtime_seconds": meta.get("runtime_seconds", 0.0),
#         "timestamp": now_iso,
#     }


# @app.post("/explain", response_model=ExplainOut)
# def explain(inp: ExplainIn) -> ExplainOut:
#     """Return token‑level attributions to explain the model's prediction.

#     Depending on the backend, this function computes LIME explanations for
#     logistic regression models or Integrated Gradients for transformer models.
#     In case of failure (e.g. missing dependencies), an empty list of spans is
#     returned so the frontend can gracefully hide the explanation.
#     """
#     txt = (inp.text or "").strip()
#     if not txt:
#         raise HTTPException(400, "Empty text")
#     # run prediction first to determine backend and label
#     label, probs, meta = predict_with_model(txt, inp.backend or "auto")
#     confidence = float(max(probs.values()))
#     chosen_backend = meta.get("backend", inp.backend or "auto").lower()
#     spans: List[Span] = []
#     polarity = "ai-positive"  # positive weights favour the AI class
#     # Try to compute attributions
#     try:
#         if chosen_backend in ("lr", "old_lr", "auto"):
#             # LIME explanation for linear models
#             from lime.lime_text import LimeTextExplainer  # type: ignore
#             import numpy as _np  # local alias to avoid global import collision

#             # Local wrapper to produce probabilities in the order [Human, AI]
#             def predict_proba(texts: List[str]):
#                 res = []
#                 for s in texts:
#                     _lab, _probs, _ = predict_with_model(s, chosen_backend)
#                     res.append([_probs.get("Human", 0.0), _probs.get("AI", 0.0)])
#                 return _np.array(res)

#             explainer = LimeTextExplainer(class_names=["Human", "AI"])
#             exp = explainer.explain_instance(txt, predict_proba, num_features=20)

#             # LIME returns list of (token, weight); map tokens back to character spans.
#             offsets_used = [False] * len(txt)
#             for token, weight in exp.as_list():
#                 token_str = str(token)
#                 search_pos = 0
#                 # search for occurrences of token; mark only unused positions
#                 while True:
#                     idx = txt.lower().find(token_str.lower(), search_pos)
#                     if idx == -1:
#                         break
#                     end = idx + len(token_str)
#                     # ensure this span is not overlapping previously used characters
#                     if not any(offsets_used[idx:end]):
#                         for i in range(idx, end):
#                             offsets_used[i] = True
#                         spans.append(Span(start=idx, end=end, weight=float(weight)))
#                         break
#                     else:
#                         search_pos = end
#             # Normalise weights
#             max_abs = max((abs(s.weight) for s in spans), default=1.0)
#             spans = [Span(start=s.start, end=s.end, weight=(s.weight / max_abs)) for s in spans]

#         elif chosen_backend == "transformer":
#             # Integrated Gradients explanation for transformers
#             import torch  # type: ignore
#             from captum.attr import IntegratedGradients  # type: ignore
#             # Ensure model and tokenizer are loaded
#             inf._load_transformer()
#             tokenizer = inf._TOKENIZER
#             model = inf._TRANS_MODEL
#             device = inf._DEVICE
#             model.eval()
#             encoding = tokenizer(
#                 txt,
#                 return_tensors="pt",
#                 return_offsets_mapping=True,
#                 truncation=True,
#                 max_length=inf.MAX_LEN,
#             )
#             input_ids = encoding["input_ids"].to(device)
#             attention_mask = encoding["attention_mask"].to(device)
#             offsets = encoding["offset_mapping"][0].tolist()
#             # Determine target index: 1 for AI, 0 for Human
#             pred_idx = 1 if label == "AI" else 0
#             # Define forward function for Captum
#             def forward_func(in_ids, mask):
#                 out = model(in_ids, attention_mask=mask)
#                 return out.logits[:, pred_idx]
#             ig = IntegratedGradients(forward_func)
#             attributions = ig.attribute(
#                 inputs=input_ids,
#                 additional_forward_args=(attention_mask,),
#                 n_steps=32,
#             )
#             # Sum attributions across embedding dimensions
#             token_attr = attributions.sum(dim=-1).squeeze().detach().cpu().numpy()
#             spans_tmp: List[Span] = []
#             for (start, end), w in zip(offsets, token_attr):
#                 if end > start:
#                     spans_tmp.append(Span(start=int(start), end=int(end), weight=float(w)))
#             max_abs = max((abs(s.weight) for s in spans_tmp), default=1.0)
#             spans = [Span(start=s.start, end=s.end, weight=(s.weight / max_abs)) for s in spans_tmp]
#             polarity = "ai-positive"

#     except Exception as exc:
#         # Explanation failed; log warning and return no spans
#         print(f"[warn] explanation failed: {type(exc).__name__}: {exc}")
#         spans = []

#     return ExplainOut(
#         pred_label=label,
#         confidence=confidence,
#         spans=spans,
#         polarity=polarity,
#         meta={"backend": chosen_backend, "runtime_seconds": meta.get("runtime_seconds", 0.0)},
#     )
"""
FastAPI backend for AI vs Human classifier — extended edition with explainability.
This module exposes prediction and explanation endpoints that serve a simple
frontend for classifying text as AI‑generated or human‑written.

The `/predict` route returns the predicted label and probability scores for the
requested backend (logistic regression, transformer, or auto selection).  The
new `/explain` route computes token‑level importance scores using either LIME
for linear models or Integrated Gradients via Captum for transformer models.

Weights are normalized to lie in the range −1..1 so the frontend can map them
to colour intensities.  Positive values indicate evidence for the “AI” class
(see `polarity`), while negative values indicate evidence for the “Human” class.
The response includes a list of character spans and corresponding weights to
highlight text on the client side.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib

import src.inference as inf  # for transformer or LR fallback

# Base paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

UNSURE_THRESHOLD = float(os.getenv("UNSURE_THRESHOLD", "0.60"))

# ---- Load optional models (best‑effort) ----
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
app = FastAPI(title="AI vs Human — API (extended)")
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


def predict_with_model(text: str, backend: str) -> Tuple[str, Dict[str, float], Dict[str, Any]]:
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
            # fall back to inference auto‑resolution (may choose transformer/LR)
            label, prob_dict, meta = inf.predict_text(text, backend="auto")

    else:
        # unknown backend: fall back to auto
        label, prob_dict, meta = inf.predict_text(text, backend="auto")

    meta["runtime_seconds"] = round(time.time() - start_time, 4)
    return label, prob_dict, meta


class Span(BaseModel):
    """A highlighted span with start/end indices and a weight."""
    start: int
    end: int
    weight: float


class ExplainIn(BaseModel):
    """Request body for explanation endpoint."""
    text: str
    backend: Optional[str] = "auto"


class ExplainOut(BaseModel):
    """Response body for explanation endpoint."""
    pred_label: str
    confidence: float
    spans: List[Span]
    polarity: str
    meta: Dict[str, Any]


@app.get("/")
def root() -> FileResponse:
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


@app.post("/explain", response_model=ExplainOut)
def explain(inp: ExplainIn) -> ExplainOut:
    """Return token‑level attributions to explain the model's prediction.

    Depending on the backend, this function computes LIME explanations for
    logistic regression models or Integrated Gradients for transformer models.
    In case of failure (e.g. missing dependencies), an empty list of spans is
    returned so the frontend can gracefully hide the explanation.
    """
    txt = (inp.text or "").strip()
    if not txt:
        raise HTTPException(400, "Empty text")
    # run prediction first to determine backend and label
    label, probs, meta = predict_with_model(txt, inp.backend or "auto")
    confidence = float(max(probs.values()))
    chosen_backend = meta.get("backend", inp.backend or "auto").lower()
    spans: List[Span] = []
    polarity = "ai-positive"  # positive weights favour the AI class
    # Try to compute attributions
    try:
        if chosen_backend in ("lr", "old_lr", "auto"):
            # LIME explanation for linear models
            from lime.lime_text import LimeTextExplainer  # type: ignore
            import numpy as _np  # local alias to avoid global import collision

            # Local wrapper to produce probabilities in the order [Human, AI]
            def predict_proba(texts: List[str]):
                res = []
                for s in texts:
                    _lab, _probs, _ = predict_with_model(s, chosen_backend)
                    res.append([_probs.get("Human", 0.0), _probs.get("AI", 0.0)])
                return _np.array(res)

            explainer = LimeTextExplainer(class_names=["Human", "AI"])
            exp = explainer.explain_instance(txt, predict_proba, num_features=20)

            # LIME returns list of (token, weight); map tokens back to character spans.
            offsets_used = [False] * len(txt)
            for token, weight in exp.as_list():
                token_str = str(token)
                search_pos = 0
                # search for occurrences of token; mark only unused positions
                while True:
                    idx = txt.lower().find(token_str.lower(), search_pos)
                    if idx == -1:
                        break
                    end = idx + len(token_str)
                    # ensure this span is not overlapping previously used characters
                    if not any(offsets_used[idx:end]):
                        for i in range(idx, end):
                            offsets_used[i] = True
                        spans.append(Span(start=idx, end=end, weight=float(weight)))
                        break
                    else:
                        search_pos = end
            # Normalise weights
            max_abs = max((abs(s.weight) for s in spans), default=1.0)
            spans = [Span(start=s.start, end=s.end, weight=(s.weight / max_abs)) for s in spans]

        elif chosen_backend == "transformer":
            # Integrated Gradients explanation for transformers
            import torch  # type: ignore
            # Use LayerIntegratedGradients to compute attributions on the embedding layer
            from captum.attr import LayerIntegratedGradients  # type: ignore
            # Ensure model and tokenizer are loaded
            inf._load_transformer()
            tokenizer = inf._TOKENIZER
            model = inf._TRANS_MODEL
            device = inf._DEVICE
            model.eval()
            # Tokenize with offsets for mapping token indices back to character spans
            encoding = tokenizer(
                txt,
                return_tensors="pt",
                return_offsets_mapping=True,
                truncation=True,
                max_length=inf.MAX_LEN,
            )
            # input_ids and attention_mask should remain integer (long) tensors
            input_ids = encoding["input_ids"].to(device).long()
            attention_mask = encoding["attention_mask"].to(device).long()
            offsets = encoding["offset_mapping"][0].tolist()
            # Determine target index: 1 for AI, 0 for Human
            pred_idx = 1 if label == "AI" else 0
            # Define forward function for Captum
            def forward_func(in_ids, mask):
                out = model(in_ids, attention_mask=mask)
                return out.logits[:, pred_idx]
            # Instantiate LayerIntegratedGradients on the model's input embedding layer
            lig = LayerIntegratedGradients(forward_func, model.get_input_embeddings())
            # Baseline of zeros (same shape) for IG path
            baselines = torch.zeros_like(input_ids)
            attributions = lig.attribute(
                inputs=input_ids,
                baselines=baselines,
                additional_forward_args=(attention_mask,),
                n_steps=32,
            )
            # Sum attributions across embedding dimensions to get per-token scores
            token_attr = attributions.sum(dim=-1).squeeze().detach().cpu().numpy()
            spans_tmp: List[Span] = []
            for (start, end), w in zip(offsets, token_attr):
                if end > start:
                    spans_tmp.append(Span(start=int(start), end=int(end), weight=float(w)))
            max_abs = max((abs(s.weight) for s in spans_tmp), default=1.0)
            spans = [Span(start=s.start, end=s.end, weight=(s.weight / max_abs)) for s in spans_tmp]
            polarity = "ai-positive"

    except Exception as exc:
        # Explanation failed; log warning and return no spans
        print(f"[warn] explanation failed: {type(exc).__name__}: {exc}")
        spans = []

    return ExplainOut(
        pred_label=label,
        confidence=confidence,
        spans=spans,
        polarity=polarity,
        meta={"backend": chosen_backend, "runtime_seconds": meta.get("runtime_seconds", 0.0)},
    )