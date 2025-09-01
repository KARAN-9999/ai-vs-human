"""
FastAPI backend for the AI vs Human classifier (TF-IDF only).

Endpoints:
- GET  /            -> serves frontend/index.html
- GET  /health
- POST /predict     -> { prediction, confidence, probabilities, model, runtime_seconds, timestamp }
- GET  /history     -> { count, history: [...] }
- GET  /analytics   -> { totals_by_label, avg_confidence_by_label, confidence_histogram_bins, last_50 }
- GET  /version     -> { clf_loaded, hf_model_name }

Stores predictions in SQLite at data/app.db. Serves static frontend at /static.
"""

from __future__ import annotations

import os
import math
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import src.inference as inf  # TF-IDF-only predict_text

BASE_DIR = Path(__file__).resolve().parent
DB_DIR   = BASE_DIR / "data"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH  = str(DB_DIR / "app.db")   # <<< absolute path now

def _conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    with _conn() as con:
        con.execute(
            """
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
            """
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_ts ON predictions(ts)")
        con.commit()

def insert_prediction(row: Dict) -> None:
    with _conn() as con:
        con.execute(
            """
            INSERT INTO predictions(text, prediction, confidence, probs_json,
                                    model_name, runtime_s, ts)
            VALUES(?,?,?,?,?,?,?)
            """,
            (
                row["text"],
                row["prediction"],
                row["confidence"],
                json.dumps(row["probabilities"]),
                row["model_name"],
                row["runtime_seconds"],
                row["timestamp"],
            ),
        )
        con.commit()

def fetch_history(limit: int = 20) -> List[Dict]:
    with _conn() as con:
        rows = con.execute(
            """
            SELECT id, text, prediction, confidence, ts
            FROM predictions
            ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    out: List[Dict] = []
    for rid, text, pred, conf, ts in rows:
        preview = (text or "")[:140].replace("\n", " ")
        out.append(
            {
                "id": rid,
                "input_preview": preview,
                "prediction": pred,
                "confidence": float(conf),
                "timestamp": ts,
            }
        )
    return out

def label_stats() -> Tuple[Dict, Dict]:
    with _conn() as con:
        totals = dict(
            con.execute(
                "SELECT prediction, COUNT(*) FROM predictions GROUP BY prediction"
            ).fetchall()
        )
        avgs = dict(
            con.execute(
                "SELECT prediction, AVG(confidence) FROM predictions GROUP BY prediction"
            ).fetchall()
        )
    return totals, {k: float(v) for k, v in avgs.items()}

def confidence_list() -> List[float]:
    with _conn() as con:
        return [float(x[0]) for x in con.execute("SELECT confidence FROM predictions")]

def last_50() -> List[Tuple[str, str, float]]:
    with _conn() as con:
        return con.execute(
            """
            SELECT ts, prediction, confidence FROM predictions
            ORDER BY id DESC LIMIT 50
            """
        ).fetchall()

# --- FastAPI app ---
app = FastAPI(title="AI vs Human — API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "frontend")), name="static")

@app.on_event("startup")
def _startup() -> None:
    init_db()

class PredictIn(BaseModel):
    text: str

@app.middleware("http")
async def add_nocache_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.get("/")
def root() -> FileResponse:
    return FileResponse(str(BASE_DIR / "frontend" / "index.html"))

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}

@app.post("/predict")
def predict(inp: PredictIn) -> Dict:
    txt = (inp.text or "").strip()
    if not txt:
        raise HTTPException(status_code=400, detail="Empty text")

    label, probs, meta = inf.predict_text(txt)
    confidence = float(max(probs.values()))
    now_iso = datetime.now(timezone.utc).isoformat()

    row = {
        "text": txt,
        "prediction": label,
        "confidence": confidence,
        "probabilities": probs,
        "model_name": meta.get("model_name", "tfidf_lr_v1"),
        "runtime_seconds": meta.get("runtime_seconds", 0.0),
        "timestamp": now_iso,
    }
    insert_prediction(row)

    return {
        "prediction": label,
        "confidence": confidence,
        "probabilities": probs,
        "model": {"hf_model_name": meta.get("model_name", "tfidf_lr_v1")},
        "runtime_seconds": meta.get("runtime_seconds", 0.0),
        "timestamp": now_iso,
    }

@app.get("/history")
def history(limit: int = Query(20, ge=1, le=200)) -> Dict:
    items = fetch_history(limit)
    return {"count": len(items), "history": items}

@app.get("/analytics")
def analytics() -> Dict:
    """Return analytics for the last 50 predictions."""
    totals, avg_conf = label_stats()
    confs = confidence_list()
    bins = [0] * 10
    for c in confs:
        idx = min(9, int(math.floor(float(c) * 10)))
        bins[idx] += 1
    series = [
        {"ts": ts, "prediction": pred, "confidence": float(conf)}
        for ts, pred, conf in last_50()
    ]
    return {
        "totals_by_label": totals,
        "avg_confidence_by_label": avg_conf,
        "confidence_histogram_bins": bins,
        "last_50": series,
    }

@app.get("/version")
def version() -> Dict:
    """Model availability info (no HF)."""
    model_dir = Path("models/lr_v1")
    have_vec = (model_dir / "tfidf.joblib").exists()
    have_clf = (model_dir / "model.joblib").exists()
    have_lab = (model_dir / "labels.joblib").exists()
    clf_loaded = bool(have_vec and have_clf and have_lab)
    return {"clf_loaded": clf_loaded, "hf_model_name": "-"}

# Run: uvicorn app:app --reload --port 8000
