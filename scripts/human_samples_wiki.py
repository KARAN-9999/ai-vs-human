# scripts/sample_human_wiki.py
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterable, Optional

from datasets import load_dataset

OUT = Path("data/augmented_human.csv")
MIN_LEN = 200
MAX_LEN = 1200
TARGET  = 1500  # approx rows to collect
SEED    = 42

# (Optional) silence the Windows symlink warning from huggingface_hub
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _pick_text_from_row(row: dict) -> Optional[str]:
    """
    Return a usable text string from a HF row with unknown schema.
    Tries common fields in order; returns None if nothing usable.
    """
    # Prefer a single long field if present
    for key in ("text", "content", "article", "paragraph"):
        if key in row and isinstance(row[key], str) and row[key].strip():
            return row[key].strip()

    # Common pair for older AG News
    if "title" in row or "description" in row:
        title = row.get("title", "") or ""
        desc  = row.get("description", "") or ""
        combined = f"{title} {desc}".strip()
        if combined:
            return combined

    # Some datasets use 'body'
    if "body" in row and isinstance(row["body"], str) and row["body"].strip():
        return row["body"].strip()

    return None


def _collect_stream(stream: Iterable[dict], limit: int) -> list[str]:
    out: list[str] = []
    for row in stream:
        text = _pick_text_from_row(row)
        if not text:
            continue
        t = text.strip()
        if MIN_LEN <= len(t) <= MAX_LEN:
            out.append(t)
            if len(out) >= limit:
                break
    return out


def try_wikimedia_wikipedia() -> list[str]:
    """
    Preferred modern Wikipedia dataset:
      - dataset: 'wikimedia/wikipedia'
      - a recent config like '20231101.en' (English dump)
    We stream 'train' to avoid full downloads.
    """
    # Try a couple of recent dumps in order
    for cfg in ("20240501.en", "20231101.en", "20230701.en"):
        try:
            ds = load_dataset("wikimedia/wikipedia", cfg, split="train", streaming=True)
            return _collect_stream(ds, TARGET)
        except Exception:
            continue
    raise RuntimeError("wikimedia/wikipedia not available")


def try_wikitext() -> list[str]:
    """
    Wikitext (human authored) – lighter than Wikipedia.
    Use the 'wikitext-103-raw-v1' train split and treat each line as a paragraph.
    """
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    out: list[str] = []
    for row in ds:
        t = (row.get("text") or "").strip()
        if MIN_LEN <= len(t) <= MAX_LEN:
            out.append(t)
            if len(out) >= TARGET:
                break
    return out


def try_ag_news() -> list[str]:
    """
    AG News as a fallback. Newer releases often have a single 'text' field.
    """
    ds = load_dataset("ag_news", split="train")
    out: list[str] = []
    for row in ds:
        t = _pick_text_from_row(row)
        if not t:
            continue
        t = t.strip()
        if MIN_LEN <= len(t) <= MAX_LEN:
            out.append(t)
            if len(out) >= TARGET:
                break
    return out


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Try modern Wikipedia → Wikitext → AG News
    source = None
    paras: list[str] = []
    try:
        paras = try_wikimedia_wikipedia()
        source = "wikimedia_wikipedia"
    except Exception as e:
        print(f"[warn] wikimedia/wikipedia failed ({e}); trying wikitext.")
        try:
            paras = try_wikitext()
            source = "wikitext-103-raw-v1"
        except Exception as e2:
            print(f"[warn] wikitext failed ({e2}); using ag_news.")
            paras = try_ag_news()
            source = "ag_news"

    rows = [{"text": p, "label": "Human", "source": source} for p in paras]

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label", "source"])
        w.writeheader()
        w.writerows(rows)

    print(f"[ok] Wrote {len(rows)} Human rows to {OUT} (source={source})")


if __name__ == "__main__":
    main()
