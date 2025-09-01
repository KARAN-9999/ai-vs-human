# src/db.py
import os, sqlite3, json
from typing import List, Dict

DB_PATH = os.getenv("DB_PATH", "data/app.db")

def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
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
        )""")
        con.execute("CREATE INDEX IF NOT EXISTS idx_ts ON predictions(ts)")
        con.commit()

def insert_prediction(row: Dict):
    with _conn() as con:
        con.execute("""
        INSERT INTO predictions(text, prediction, confidence, probs_json, model_name, runtime_s, ts)
        VALUES(?,?,?,?,?,?,?)
        """, (row["text"], row["prediction"], row["confidence"],
              json.dumps(row["probabilities"]), row["model_name"],
              row["runtime_seconds"], row["timestamp"]))
        con.commit()

def get_history(limit: int = 20) -> List[Dict]:
    with _conn() as con:
        rows = con.execute("""
        SELECT id, text, prediction, confidence, ts
        FROM predictions ORDER BY id DESC LIMIT ?
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

def get_all_confidences():
    with _conn() as con:
        return [float(x[0]) for x in con.execute("SELECT confidence FROM predictions")]
        
def get_label_stats():
    with _conn() as con:
        totals = dict(con.execute("SELECT prediction, COUNT(*) FROM predictions GROUP BY prediction").fetchall())
        avgs = dict(con.execute("SELECT prediction, AVG(confidence) FROM predictions GROUP BY prediction").fetchall())
    return totals, {k: float(v) for k, v in avgs.items()}

def last_50():
    with _conn() as con:
        return con.execute("""
        SELECT ts, prediction, confidence FROM predictions
        ORDER BY id DESC LIMIT 50
        """).fetchall()
