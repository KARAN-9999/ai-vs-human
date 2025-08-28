# inference.py
import torch
import joblib
import numpy as np
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "distilroberta-base"
MAX_LEN = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ——— Load HF backbone (for embeddings)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
hf_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
hf_model.eval()

# ——— Load your trained classifier + label encoder
# Make sure these files exist in ./models/
model = joblib.load("models/logreg_transformer_emb_best.joblib")
label_encoder = joblib.load("models/label_encoder_transformer.joblib")

def get_embedding(text: str) -> np.ndarray:
    """Mean-pooled transformer embedding."""
    enc = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN
    ).to(device)

    with torch.no_grad():
        out = hf_model(**enc).last_hidden_state            # [1, seq, hidden]
        mask = enc["attention_mask"].unsqueeze(-1).to(out.dtype)
        mean = (out * mask).sum(1) / mask.sum(1)           # [1, hidden]
    return mean.cpu().numpy()

def predict_text(text: str) -> dict:
    """Return label + class probabilities."""
    emb = get_embedding(text)                              # [1, hidden]
    probs = model.predict_proba(emb)[0]                    # [num_classes]
    pred_idx = int(np.argmax(probs))
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    return {
        "prediction": pred_label,
        "probabilities": {
            label: float(p) for label, p in zip(label_encoder.classes_, probs)
        }
    }

if __name__ == "__main__":
    sample = "This is likely written by an AI assistant."
    print(predict_text(sample))
