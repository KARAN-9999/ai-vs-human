# src/inference.py
import os
import time
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

import joblib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from .utils import clean_text_simple, softmax, tail_jsonl

# Default model files (adjust if your filenames are different)
CLASSIFIER_PATH = Path("models/logreg_transformer_emb_best.joblib")
LABEL_ENCODER_PATH = Path("models/label_encoder_transformer.joblib")
HF_MODEL_NAME = "distilroberta-base"
MAX_LEN = 256
BATCH_SIZE = 32
HISTORY_PATH = Path("reports/prediction_history.jsonl")
HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

_PREDICTOR = None  # module-level lazy singleton


class Predictor:
    def __init__(
        self,
        classifier_path: Path = CLASSIFIER_PATH,
        label_encoder_path: Path = LABEL_ENCODER_PATH,
        hf_model_name: str = HF_MODEL_NAME,
        max_len: int = MAX_LEN,
        device: Optional[torch.device] = None,
    ):
        self.classifier_path = Path(classifier_path)
        self.label_encoder_path = Path(label_encoder_path)
        self.hf_model_name = hf_model_name
        self.max_len = max_len
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # lazy placeholders
        self._tokenizer = None
        self._hf_model = None
        self._clf = None
        self._label_encoder = None
        self._hidden_dim = None

        self._ensure_model_files_exist()
        self._load_classifier_components()

    def _ensure_model_files_exist(self):
        if not self.classifier_path.exists():
            raise FileNotFoundError(f"Classifier file not found: {self.classifier_path}")
        if not self.label_encoder_path.exists():
            raise FileNotFoundError(f"Label encoder file not found: {self.label_encoder_path}")

    def _load_tokenizer_and_hf(self):
        if self._tokenizer is None or self._hf_model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
            self._hf_model = AutoModel.from_pretrained(self.hf_model_name).to(self.device)
            self._hf_model.eval()

    def _load_classifier_components(self):
        # load classifier and encoder right away (cheap)
        self._clf = joblib.load(self.classifier_path)
        self._label_encoder = joblib.load(self.label_encoder_path)

        # try to infer hidden dims from classifier.coef_
        try:
            self._hidden_dim = int(self._clf.coef_.shape[1])
        except Exception:
            self._hidden_dim = None

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Return mean-pooled embeddings (numpy array shape [n, hidden_dim]).
        Uses batching to avoid OOM for long lists.
        """
        self._load_tokenizer_and_hf()
        if isinstance(texts, str):
            texts = [texts]

        all_embs = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            enc = self._tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self._hf_model(**enc).last_hidden_state  # [bsz, seq, hidden]
                mask = enc["attention_mask"].unsqueeze(-1).to(out.dtype)  # [bsz, seq, 1]
                summed = (out * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                mean = (summed / counts).cpu().numpy()  # [bsz, hidden]
                all_embs.append(mean)
        return np.vstack(all_embs)

    def predict_proba_from_emb(self, emb: np.ndarray) -> np.ndarray:
        """Return probabilities array shape [n, n_classes]."""
        if hasattr(self._clf, "predict_proba"):
            probs = self._clf.predict_proba(emb)
        elif hasattr(self._clf, "decision_function"):
            logits = self._clf.decision_function(emb)
            # if binary, decision_function returns shape (n,) -> make (n,2)
            if logits.ndim == 1:
                logits = np.vstack([-logits, logits]).T
            probs = softmax(logits)
        else:
            # fallback: call predict and give 1.0 for predicted class (not ideal)
            preds = self._clf.predict(emb)
            probs = np.zeros((len(preds), len(self._label_encoder.classes_)))
            for i, p in enumerate(preds):
                probs[i, int(p)] = 1.0
        return probs

    def predict_text(self, text: str) -> Dict[str, Any]:
        """
        Main convenience function: returns dict with prediction,
        probabilities, confidence, timestamp, model info etc.
        Also appends to history file.
        """
        t0 = time.time()
        raw_text = clean_text_simple(text)
        emb = self.embed([raw_text])  # shape (1, hidden)
        probs = self.predict_proba_from_emb(emb)[0]  # (n_classes,)
        idx = int(np.argmax(probs))
        label = str(self._label_encoder.inverse_transform([idx])[0])
        confidence = float(probs[idx])
        classes = [str(c) for c in list(self._label_encoder.classes_)]

        result = {
            "prediction": label,
            "confidence": confidence,
            "probabilities": {cls: float(p) for cls, p in zip(classes, probs)},
            "model": {
                "classifier_path": str(self.classifier_path),
                "hf_model_name": self.hf_model_name,
            },
            "input_preview": raw_text[:512],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "runtime_seconds": round(time.time() - t0, 4),
        }

        # save history (append jsonl)
        try:
            with HISTORY_PATH.open("a", encoding="utf-8") as f:
                json.dump({**result, "input_full": raw_text}, f)
                f.write("\n")
        except Exception as exc:
            # don't fail inference because of logging
            result["_history_write_error"] = str(exc)

        return result

    def get_history(self, n: int = 100) -> List[Dict[str, Any]]:
        return tail_jsonl(HISTORY_PATH, n)


def get_predictor() -> Predictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = Predictor()
    return _PREDICTOR


# convenience module-level functions
def predict_text(text: str) -> Dict[str, Any]:
    return get_predictor().predict_text(text)


def get_history(n: int = 100) -> List[Dict[str, Any]]:
    return get_predictor().get_history(n)
