# src/utils.py
import numpy as np
import json
from pathlib import Path
from typing import List

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def clean_text_simple(text: str) -> str:
    # lightweight normalize/cleanup: trim and collapse spaces.
    if not isinstance(text, str):
        return ""
    s = text.strip()
    s = " ".join(s.split())
    return s

def tail_jsonl(path: Path, n: int = 100) -> List[dict]:
    """Return last n JSON objects from a jsonl file (path may not exist)."""
    out = []
    if not path.exists():
        return out
    with path.open("rb") as f:
        # seek from end, read blocks - simple approach
        try:
            f.seek(0, 2)
            filesize = f.tell()
            blocksize = 4096
            data = b""
            pointer = filesize
            while pointer > 0 and len(out) < n:
                read_size = blocksize if pointer - blocksize > 0 else pointer
                pointer -= read_size
                f.seek(pointer)
                chunk = f.read(read_size)
                data = chunk + data
                lines = data.splitlines()
                # parse from the end
                out = []
                for line in reversed(lines):
                    try:
                        out.append(json.loads(line.decode("utf-8")))
                        if len(out) >= n:
                            break
                    except Exception:
                        continue
                # if not enough lines, continue loop
        except Exception:
            # fallback simple read
            f.seek(0)
            for line in f:
                try:
                    out.append(json.loads(line.decode("utf-8")))
                except Exception:
                    continue
    return list(reversed(out))[:n]
