# inference.py (root)
from src.inference import predict_text

if __name__ == "__main__":
    sample = "This is likely written by an AI assistant."
    print(predict_text(sample))
