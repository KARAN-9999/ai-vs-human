# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from inference import predict_text

app = FastAPI(title="AI vs Human Classifier API", version="1.0.0")

class TextInput(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: TextInput):
    return predict_text(payload.text)
