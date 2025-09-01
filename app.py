# app.py
from __future__ import annotations
import os
import math
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.inference import predict_text

# --- DB setup (SQLite in data/app.db) ---
DB_PATH = os.getenv("DB_PATH", "data/app.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

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

def insert_prediction(row: Dict):
    with _conn() as con:
        con.execute("""
        INSERT INTO predictions(text, prediction, confidence, probs_json, model_name, runtime_s, ts)
        VALUES(?,?,?,?,?,?,?)
        """, (
            row["text"], row["prediction"], row["confidence"],
            json.dumps(row["probabilities"]),
            row["model_name"], row["runtime_seconds"], row["timestamp"]
        ))
        con.commit()

def fetch_history(limit: int = 20) -> List[Dict]:
    with _conn() as con:
        rows = con.execute("""
        SELECT id, text, prediction, confidence, ts
        FROM predictions
        ORDER BY id DESC LIMIT ?
        """, (limit,)).fetchall()
    out = []
    for rid, text, pred, conf, ts in rows:
        preview = (text or "")[:140].replace("\n", " ")
        out.append({
            "id": rid,
            "input_preview": preview,
            "prediction": pred,
            "confidence": float(conf),
            "timestamp": ts
        })
    return out

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
        SELECT ts, prediction, confidence FROM predictions ORDER BY id DESC LIMIT 50
        """).fetchall()

# --- FastAPI app ---
app = FastAPI(title="AI vs Human — API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# Serve frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.on_event("startup")
def _startup():
    init_db()

class PredictIn(BaseModel):
    text: str

@app.get("/")
def root():
    # Serve the UI
    return FileResponse("frontend/index.html")

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}

@app.post("/predict")
def predict(inp: PredictIn):
    txt = (inp.text or "").strip()
    if not txt:
        raise HTTPException(status_code=400, detail="Empty text")

    label, probs, meta = predict_text(txt)
    confidence = float(max(probs.values()))

    # persist
    row = {
        "text": txt,
        "prediction": label,
        "confidence": confidence,
        "probabilities": probs,
        "model_name": meta["model_name"],
        "runtime_seconds": meta["runtime_seconds"],
        "timestamp": meta["timestamp"],
    }
    insert_prediction(row)

    return {
        "prediction": label,
        "confidence": confidence,
        "probabilities": probs,
        "model": {"hf_model_name": meta["model_name"]},
        "runtime_seconds": meta["runtime_seconds"],
        "timestamp": meta["timestamp"],
    }

@app.get("/history")
def history(limit: int = Query(20, ge=1, le=200)):
    items = fetch_history(limit)
    return {"count": len(items), "history": items}

@app.get("/analytics")
def analytics():
    totals, avg_conf = label_stats()
    confs = confidence_list()
    bins = [0] * 10
    for c in confs:
        idx = min(9, int(math.floor(c * 10)))
        bins[idx] += 1
    series = [{"ts": ts, "prediction": pred, "confidence": float(conf)} for ts, pred, conf in last_50()]
    return {
        "totals_by_label": totals,
        "avg_confidence_by_label": avg_conf,
        "confidence_histogram_bins": bins,
        "last_50": series
    }
