# scripts/generate_ai_samples.py
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

OUT = Path("data/augmented_ai.csv")
MODEL_NAME = "distilgpt2"     # small & fast
SEED = 42

# ---- Generation knobs (tweak if needed) ----
PROMPTS: List[str] = [
    "Write a concise explanation of how transformers work in simple terms.",
    "Summarize the benefits and risks of using AI in education.",
    "Draft a professional email requesting feedback on a report.",
    "Compose a short product description for a new smartwatch.",
    "Explain the difference between correlation and causation with examples.",
    "Provide a balanced argument for and against remote work.",
    "Give an overview of climate change mitigation strategies.",
    "Provide a casual social caption about a weekend trip.",
    "Answer helpfully: How do I reduce CPU usage in Python?",
    "Write a short news brief about a technology conference.",
]
NUM_SAMPLES_PER_PROMPT = 30   # total per prompt (reduce on CPU if slow)
BATCH_SIZE = 5                # how many sequences to sample at once
MAX_NEW_TOKENS = 80
TEMPERATURE = 0.9
TOP_P = 0.95
# --------------------------------------------


def main():
    torch.manual_seed(SEED)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    is_new = not OUT.exists()
    f = OUT.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=["text", "label", "source", "prompt"])
    if is_new:
        writer.writeheader()

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # GPT-2 family has no PAD; use EOS as PAD and make sure the model knows.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl.config.pad_token_id = tok.pad_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl.to(device).eval()

    total_written = 0
    try:
        for p_idx, prompt in enumerate(PROMPTS, 1):
            input_ids = tok.encode(prompt, return_tensors="pt").to(device)
            # Use ones since we're not padding the prompt here.
            attn_mask = torch.ones_like(input_ids, device=device)

            remaining = NUM_SAMPLES_PER_PROMPT
            rounds = math.ceil(remaining / BATCH_SIZE)
            for r in range(rounds):
                cur_bs = min(BATCH_SIZE, remaining)

                # repeat the prompt 'cur_bs' times for batch sampling
                batch_input = input_ids.repeat(cur_bs, 1)
                batch_mask  = attn_mask.repeat(cur_bs, 1)

                with torch.inference_mode():
                    out_ids = mdl.generate(
                        input_ids=batch_input,
                        attention_mask=batch_mask,
                        do_sample=True,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        max_new_tokens=MAX_NEW_TOKENS,
                        pad_token_id=tok.pad_token_id,
                        eos_token_id=tok.eos_token_id,
                        use_cache=True,
                    )

                for i in range(cur_bs):
                    text = tok.decode(out_ids[i], skip_special_tokens=True).strip()
                    # (Optional) ensure we don’t store just the raw prompt,
                    # keep only the continuation part if you prefer:
                    # if text.startswith(prompt): text = text[len(prompt):].lstrip()

                    writer.writerow({
                        "text": text,
                        "label": "AI",
                        "source": MODEL_NAME,
                        "prompt": prompt
                    })
                    total_written += 1

                f.flush()  # incremental flush so you keep progress
                remaining -= cur_bs
                print(f"[{p_idx}/{len(PROMPTS)}] Prompt batch {r+1}/{rounds} → total rows: {total_written}")

        print(f"[ok] Wrote {total_written} AI samples to {OUT}")

    except KeyboardInterrupt:
        print(f"[partial] Interrupted. Saved {total_written} rows so far at {OUT}")
    finally:
        f.close()


if __name__ == "__main__":
    main()
