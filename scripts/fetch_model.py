import os, pathlib, requests, zipfile, hashlib, tempfile

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
DEST_DIR = MODELS_DIR / "finetuned_bert-base-uncased"

URL = os.getenv("MODEL_ARCHIVE_URL", "").strip()
SHA256 = os.getenv("MODEL_SHA256", "").strip()

def sha256sum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_model():
    if DEST_DIR.exists() and any(DEST_DIR.iterdir()):
        print("[fetch_model] Model already exists.")
        return

    if not URL:
        raise RuntimeError("MODEL_ARCHIVE_URL not set.")

    MODELS_DIR.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        print(f"[fetch_model] Downloading {URL} ...")
        with requests.get(URL, stream=True) as r:
            r.raise_for_status()
            for chunk in r.iter_content(1024 * 1024):
                tmp.write(chunk)
        tmp_path = pathlib.Path(tmp.name)

    if SHA256:
        digest = sha256sum(tmp_path)
        if digest.lower() != SHA256.lower():
            raise RuntimeError(f"SHA256 mismatch: got {digest}, expected {SHA256}")
        print("[fetch_model] SHA256 verified")

    print("[fetch_model] Extracting ...")
    with zipfile.ZipFile(tmp_path, "r") as z:
        z.extractall(DEST_DIR)

    os.environ.setdefault("TRANSFORMER_MODEL_DIR", str(DEST_DIR))
    print(f"[fetch_model] Model ready at {DEST_DIR}")

if __name__ == "__main__":
    ensure_model()
