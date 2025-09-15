import gdown, os, zipfile, pathlib, hashlib

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
DEST_DIR = MODELS_DIR / "finetuned_bert-base-uncased"

URL = os.getenv("MODEL_ARCHIVE_URL", "").strip()
SHA256 = os.getenv("MODEL_SHA256", "").strip()
OUTPUT = MODELS_DIR / "finetuned_bert-base-uncased.zip"

def sha256sum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_model():
    if DEST_DIR.exists() and any(DEST_DIR.iterdir()):
        print("[fetch_model] Model already exists.")
        return

    MODELS_DIR.mkdir(exist_ok=True)
    print(f"[fetch_model] Downloading from {URL} ...")
    gdown.download(URL, str(OUTPUT), quiet=False)

    if SHA256:
        digest = sha256sum(OUTPUT)
        if digest.lower() != SHA256.lower():
            raise RuntimeError(f"SHA256 mismatch: got {digest}, expected {SHA256}")
        print("[fetch_model] SHA256 verified")

    print("[fetch_model] Extracting ...")
    with zipfile.ZipFile(OUTPUT, "r") as z:
        z.extractall(DEST_DIR)
    print(f"[fetch_model] Model ready at {DEST_DIR}")

if __name__ == "__main__":
    ensure_model()
