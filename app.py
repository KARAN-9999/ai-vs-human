# """
# FastAPI backend for the AI vs Human classifier (TF-IDF or Transformer).

# Endpoints:
# - GET  /            -> serves frontend/index.html
# - GET  /health
# - POST /predict     -> { prediction, confidence, probabilities, model, backend, runtime_seconds, timestamp }
# - GET  /history     -> { count, history: [...] }
# - GET  /analytics   -> { totals_by_label, avg_confidence_by_label, confidence_histogram_bins, last_50 }
# - GET  /version     -> { clf_loaded, hf_model_name }

# Stores predictions in SQLite at data/app.db. Serves static frontend at /static.
# """

# from __future__ import annotations

# import os
# import json
# import sqlite3
# from datetime import datetime, timezone
# from pathlib import Path
# from typing import Dict, List, Tuple, Optional

# from fastapi import FastAPI, HTTPException, Query, Request
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import FileResponse
# from pydantic import BaseModel

# import src.inference as inf

# # -----------------------------
# # Config & paths
# # -----------------------------
# BASE_DIR = Path(__file__).resolve().parent
# DB_DIR   = BASE_DIR / "data"
# DB_DIR.mkdir(parents=True, exist_ok=True)
# DB_PATH  = str(DB_DIR / "app.db")

# # Confidence below this becomes "Unsure"
# UNSURE_THRESHOLD = float(os.getenv("UNSURE_THRESHOLD", "0.60"))

# # -----------------------------
# # SQLite helpers
# # -----------------------------
# def _conn():
#     return sqlite3.connect(DB_PATH, check_same_thread=False)

# def init_db():
#     with _conn() as con:
#         con.execute(
#             """
#             CREATE TABLE IF NOT EXISTS predictions(
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 text TEXT NOT NULL,
#                 prediction TEXT NOT NULL,
#                 confidence REAL NOT NULL,
#                 probs_json TEXT NOT NULL,
#                 model_name TEXT NOT NULL,
#                 runtime_s REAL NOT NULL,
#                 ts TEXT NOT NULL
#             )
#             """
#         )
#         con.execute("CREATE INDEX IF NOT EXISTS idx_ts ON predictions(ts)")
#         con.commit()

# def insert_prediction(row: Dict) -> None:
#     with _conn() as con:
#         con.execute(
#             """
#             INSERT INTO predictions(text, prediction, confidence, probs_json,
#                                     model_name, runtime_s, ts)
#             VALUES(?,?,?,?,?,?,?)
#             """,
#             (
#                 row["text"],
#                 row["prediction"],
#                 row["confidence"],
#                 json.dumps(row["probabilities"]),
#                 row["model_name"],
#                 row["runtime_seconds"],
#                 row["timestamp"],
#             ),
#         )
#         con.commit()

# def fetch_history(limit: int = 20) -> List[Dict]:
#     with _conn() as con:
#         rows = con.execute(
#             """
#             SELECT id, text, prediction, confidence, ts
#             FROM predictions
#             ORDER BY id DESC LIMIT ?
#             """,
#             (limit,),
#         ).fetchall()
#     out: List[Dict] = []
#     for rid, text, pred, conf, ts in rows:
#         preview = (text or "")[:140].replace("\n", " ")
#         out.append(
#             {
#                 "id": rid,
#                 "input_preview": preview,
#                 "prediction": pred,
#                 "confidence": float(conf),
#                 "timestamp": ts,
#             }
#         )
#     return out

# def label_stats() -> Tuple[Dict, Dict]:
#     with _conn() as con:
#         totals = dict(
#             con.execute(
#                 "SELECT prediction, COUNT(*) FROM predictions GROUP BY prediction"
#             ).fetchall()
#         )
#         avgs = dict(
#             con.execute(
#                 "SELECT prediction, AVG(confidence) FROM predictions GROUP BY prediction"
#             ).fetchall()
#         )
#     return totals, {k: float(v) for k, v in avgs.items()}

# def confidence_list() -> List[float]:
#     with _conn() as con:
#         return [float(x[0]) for x in con.execute("SELECT confidence FROM predictions")]

# def last_50() -> List[Tuple[str, str, float]]:
#     with _conn() as con:
#         return con.execute(
#             """
#             SELECT ts, prediction, confidence FROM predictions
#             ORDER BY id DESC LIMIT 50
#             """
#         ).fetchall()

# # -----------------------------
# # FastAPI app
# # -----------------------------
# app = FastAPI(title="AI vs Human — API")
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # tighten to your frontend domain in production if needed
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Serve frontend static files
# app.mount("/static", StaticFiles(directory=str(BASE_DIR / "frontend")), name="static")

# @app.on_event("startup")
# def _startup() -> None:
#     init_db()

# class PredictIn(BaseModel):
#     text: str
#     backend: Optional[str] = None   # "lr", "transformer", or None/"auto"

# @app.middleware("http")
# async def add_nocache_headers(request: Request, call_next):
#     response = await call_next(request)
#     response.headers["Cache-Control"] = "no-store, must-revalidate, max-age=0"
#     response.headers["Pragma"] = "no-cache"
#     response.headers["Expires"] = "0"
#     return response

# @app.get("/")
# def root() -> FileResponse:
#     return FileResponse(str(BASE_DIR / "frontend" / "index.html"))

# @app.get("/health")
# def health() -> Dict[str, str]:
#     return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}

# @app.post("/predict")
# def predict(inp: PredictIn) -> Dict:
#     txt = (inp.text or "").strip()
#     if not txt:
#         raise HTTPException(status_code=400, detail="Empty text")

#     backend = (inp.backend or "").strip().lower() or None
#     # src.inference.predict_text MUST return: (label, probs_dict, meta_dict)
#     label, probs, meta = inf.predict_text(txt, backend=backend)

#     # Normalize keys to "AI" and "Human"
#     if "AI" not in probs or "Human" not in probs:
#         # Best-effort remap if model returned lowercase or other variants
#         norm = {k.strip().title(): float(v) for k, v in probs.items()}
#         probs = {"AI": float(norm.get("Ai", norm.get("AI", 0.0))),
#                  "Human": float(norm.get("Human", norm.get("Human", 0.0)))}

#     confidence = float(max(probs.values()))
#     display_label = label if confidence >= UNSURE_THRESHOLD else "Unsure"
#     now_iso = datetime.now(timezone.utc).isoformat()

#     row = {
#         "text": txt,
#         "prediction": display_label,   # store display label (AI/Human/Unsure)
#         "confidence": confidence,
#         "probabilities": probs,
#         "model_name": meta.get("model_name", "tfidf_lr_v1"),
#         "runtime_seconds": meta.get("runtime_seconds", 0.0),
#         "timestamp": now_iso,
#     }
#     insert_prediction(row)

#     return {
#         "prediction": display_label,
#         "confidence": confidence,
#         "probabilities": probs,
#         "model": {"hf_model_name": meta.get("model_name", "tfidf_lr_v1")},
#         "backend": meta.get("backend", backend or "auto"),
#         "runtime_seconds": meta.get("runtime_seconds", 0.0),
#         "timestamp": now_iso,
#     }

# @app.get("/history")
# def history(limit: int = Query(20, ge=1, le=200)) -> Dict:
#     items = fetch_history(limit)
#     return {"count": len(items), "history": items}

# @app.get("/analytics")
# def analytics() -> Dict:
#     """Return analytics for recent predictions. Always safe defaults."""
#     try:
#         totals, avg_conf = label_stats()
#         confs = confidence_list()
#         # 10 bins [0.0–0.1, ..., 0.9–1.0]
#         bins = [0] * 10
#         for c in confs:
#             try:
#                 idx = min(9, int(float(c) * 10))
#             except Exception:
#                 idx = 0
#             bins[idx] += 1

#         series = [
#             {"ts": ts, "prediction": pred, "confidence": float(conf)}
#             for ts, pred, conf in last_50()
#         ]

#         return {
#             "totals_by_label": {
#                 "AI": int(totals.get("AI", 0)),
#                 "Human": int(totals.get("Human", 0)),
#                 "Unsure": int(totals.get("Unsure", 0)),
#             },
#             "avg_confidence_by_label": {
#                 "AI": float(avg_conf.get("AI", 0.0)),
#                 "Human": float(avg_conf.get("Human", 0.0)),
#                 "Unsure": float(avg_conf.get("Unsure", 0.0)),
#             },
#             "confidence_histogram_bins": bins,
#             "last_50": series,
#         }
#     except Exception:
#         return {
#             "totals_by_label": {"AI": 0, "Human": 0, "Unsure": 0},
#             "avg_confidence_by_label": {"AI": 0.0, "Human": 0.0, "Unsure": 0.0},
#             "confidence_histogram_bins": [0] * 10,
#             "last_50": [],
#         }

# @app.get("/version")
# def version() -> Dict:
#     """Model availability info."""
#     model_dir = Path("models/lr_v1")
#     have_vec = (model_dir / "tfidf.joblib").exists()
#     have_clf = (model_dir / "model.joblib").exists()
#     have_lab = (model_dir / "labels.joblib").exists()
#     clf_loaded = bool(have_vec and have_clf and have_lab)

#     hf_dir = os.getenv("TRANSFORMER_MODEL_DIR", "").strip()
#     if hf_dir and Path(hf_dir).is_dir():
#         hf_name = Path(hf_dir).name
#     else:
#         auto = next(iter([p for p in Path("models").glob("finetuned_*") if p.is_dir()]), None)
#         hf_name = auto.name if auto else "-"

#     return {"clf_loaded": clf_loaded, "hf_model_name": hf_name}

# # Run locally:
# # uvicorn app:app --reload --port 8000
"""
FastAPI backend for AI vs Human classifier supporting multiple models and robust analytics.
"""

from __future__ import annotations

import os
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import time

import src.inference as inf  # For transformer or fallback


BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "data"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = str(DB_DIR / "app.db")
MODELS_DIR = BASE_DIR / "models"

UNSURE_THRESHOLD = float(os.getenv("UNSURE_THRESHOLD", "0.60"))

# Load models
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

# app and DB setup
app = FastAPI(title="AI vs Human — API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "frontend")), name="static")


def _conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS predictions(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                probs_json TEXT NOT NULL,
                model_name TEXT NOT NULL,
                runtime_s REAL NOT NULL,
                ts TEXT NOT NULL
            )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_ts ON predictions(ts)")
        con.commit()

def insert_prediction(row: Dict) -> None:
    with _conn() as con:
        con.execute("""
            INSERT INTO predictions(text, prediction, confidence, probs_json,
                                    model_name, runtime_s, ts)
            VALUES(?,?,?,?,?,?,?)
        """, (
            row["text"], row["prediction"], row["confidence"], 
            json.dumps(row["probabilities"]), row["model_name"],
            row["runtime_seconds"], row["timestamp"]
        ))
        con.commit()

def fetch_history(limit: int = 20) -> List[Dict]:
    with _conn() as con:
        rows = con.execute("""
            SELECT id, text, prediction, confidence, ts
            FROM predictions
            ORDER BY id DESC LIMIT ?
        """, (limit,)).fetchall()
    res: List[Dict] = []
    for rid, text, pred, conf, ts in rows:
        preview = (text or "")[:140].replace("\n", " ")
        res.append({
            "id": rid, "input_preview": preview,
            "prediction": pred, "confidence": float(conf), "timestamp": ts
        })
    return res

def label_stats() -> Tuple[Dict, Dict]:
    with _conn() as con:
        totals = dict(con.execute("SELECT prediction, COUNT(*) FROM predictions GROUP BY prediction").fetchall())
        avgs = dict(con.execute("SELECT prediction, AVG(confidence) FROM predictions GROUP BY prediction").fetchall())
    return totals, {k: float(v) for k, v in avgs.items()}

def confidence_list() -> List[float]:
    with _conn() as con:
        return [float(x[0]) for x in con.execute("SELECT confidence FROM predictions")]

def last_50() -> List[Tuple[str, str, float]]:
    with _conn() as con:
        return con.execute("""
            SELECT ts, prediction, confidence FROM predictions
            ORDER BY id DESC LIMIT 50
        """).fetchall()


class PredictIn(BaseModel):
    text: str
    backend: Optional[str] = None  # "lr" (robust), "old_lr", "transformer"

@app.on_event("startup")
def startup_event():
    init_db()

def predict_with_model(text: str, backend: str) -> Tuple[str, Dict[str, float], Dict]:
    start_time = time.time()
    backend = backend.lower()
    if backend == "old_lr" and old_lr_model:
        probs = old_lr_model.predict_proba([text])[0]
        labels = old_lr_model.classes_
        prob_dict = {labels[i]: float(probs[i]) for i in range(len(labels))}
        label = "AI" if prob_dict.get("AI", 0) >= prob_dict.get("Human", 0) else "Human"
        meta = {"model_name": "original_lr_model", "backend": "old_lr"}
    elif backend == "transformer":
        label, prob_dict, meta = inf.predict_text(text, backend="transformer")
    else:  # Default robust logistic regression
        if robust_lr_model:
            probs = robust_lr_model.predict_proba([text])[0]
            labels = robust_lr_model.classes_
            prob_dict = {labels[i]: float(probs[i]) for i in range(len(labels))}
            label = "AI" if prob_dict.get("AI", 0) >= prob_dict.get("Human", 0) else "Human"
            meta = {"model_name": "robust_baseline_model", "backend": "lr"}
        else:
            label, prob_dict, meta = inf.predict_text(text, backend="auto")
    meta["runtime_seconds"] = round(time.time() - start_time, 4)
    return label, prob_dict, meta

@app.middleware("http")
async def no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.get("/")
def root():
    return FileResponse(str(BASE_DIR / "frontend" / "index.html"))

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}

@app.post("/predict")
def predict(inp: PredictIn):
    txt = (inp.text or "").strip()
    if not txt:
        raise HTTPException(400, "Empty text")

    backend = (inp.backend or "lr").lower()
    label, probs, meta = predict_with_model(txt, backend)
    if "AI" not in probs or "Human" not in probs:
        norm = {k.strip().title(): float(v) for k, v in probs.items()}
        probs = {
            "AI": float(norm.get("Ai", norm.get("AI", 0.0))),
            "Human": float(norm.get("Human", norm.get("Human", 0.0))),
        }
    confidence = float(max(probs.values()))
    display_label = label if confidence >= UNSURE_THRESHOLD else "Unsure"
    now_iso = datetime.now(timezone.utc).isoformat()

    row = {
        "text": txt,
        "prediction": display_label,
        "confidence": confidence,
        "probabilities": probs,
        "model_name": meta.get("model_name", "-"),
        "runtime_seconds": meta.get("runtime_seconds", 0.0),
        "timestamp": now_iso,
    }
    insert_prediction(row)

    return {
        "prediction": display_label,
        "confidence": confidence,
        "probabilities": probs,
        "model": {"hf_model_name": meta.get("model_name", "-")},
        "backend": meta.get("backend", backend),
        "runtime_seconds": meta.get("runtime_seconds", 0.0),
        "timestamp": now_iso,
    }

@app.get("/history")
def history(limit: int = Query(20, ge=1, le=200)):
    items = fetch_history(limit)
    return {"count": len(items), "history": items}

@app.get("/analytics")
def analytics():
    try:
        totals, avg_conf = label_stats()
        confs = confidence_list()
        bins = [0] * 10
        for c in confs:
            try:
                idx = min(9, int(float(c) * 10))
            except Exception:
                idx = 0
            bins[idx] += 1

        series = [
            {"ts": ts, "prediction": pred, "confidence": float(conf)}
            for ts, pred, conf in last_50()
        ]

        return {
            "totals_by_label": {
                "AI": int(totals.get("AI", 0)),
                "Human": int(totals.get("Human", 0)),
                "Unsure": int(totals.get("Unsure", 0)),
            },
            "avg_confidence_by_label": {
                "AI": float(avg_conf.get("AI", 0.0)),
                "Human": float(avg_conf.get("Human", 0.0)),
                "Unsure": float(avg_conf.get("Unsure", 0.0)),
            },
            "confidence_histogram_bins": bins,
            "last_50": series,
        }
    except Exception:
        return {
            "totals_by_label": {"AI": 0, "Human": 0, "Unsure": 0},
            "avg_confidence_by_label": {"AI": 0.0, "Human": 0.0, "Unsure": 0.0},
            "confidence_histogram_bins": [0] * 10,
            "last_50": [],
        }

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
