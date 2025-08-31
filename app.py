
# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import logging
from pathlib import Path

# import from package
from src import predict_text, get_history, get_predictor

app = FastAPI(title="AI vs Human Classifier API", version="1.0.0")

# allow local frontend (adjust origins in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_INDEX = Path("frontend/index.html")

class TextInput(BaseModel):
    text: str

@app.on_event("startup")
def startup_event():
    # optional warm-up: instantiate predictor lazily so the first request isn't slowed by imports.
    try:
        _ = get_predictor()
        logging.info("Predictor instance created (lazy load complete).")
    except Exception as e:
        # keep server up even if models missing, errors will throw on predict
        logging.warning(f"Predictor failed to initialize at startup: {e}")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: TextInput):
    if not payload.text or not isinstance(payload.text, str):
        raise HTTPException(status_code=422, detail="Invalid text payload")
    try:
        res = predict_text(payload.text)
        return JSONResponse(content=res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def history(limit: int = 100):
    try:
        data = get_history(limit)
        return {"count": len(data), "history": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# serve simple frontend index if exists
@app.get("/")
def index():
    if FRONTEND_INDEX.exists():
        return FileResponse(FRONTEND_INDEX)
    return {"info": "AI vs Human Classifier API. See /docs for OpenAPI UI."}

