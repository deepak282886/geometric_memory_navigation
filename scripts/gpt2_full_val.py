"""GPT-2 small on the FULL 10,042-example validation split (run in background).

~24 minutes on CPU, so this is designed to be launched detached: it writes
per-example predictions to ``experiments/gpt2_val_full_predictions.json`` and
progress to stdout, and ``scripts/full_val_eval.py`` picks the file up when it
exists.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.evaluate import ending_log_prob_stats  # noqa: E402

VAL_FULL_PATH = PROJECT_ROOT / "data" / "val_full.json"
OUTPUT_PATH = PROJECT_ROOT / "experiments" / "gpt2_val_full_predictions.json"
GPT2_MODEL_NAME = "gpt2"
PROGRESS_EVERY = 250


def main() -> None:
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    with VAL_FULL_PATH.open(encoding="utf-8") as handle:
        val_data: List[Dict[str, Any]] = json.load(handle)

    print(f"[gpt2] {len(val_data)} validation examples", flush=True)
    tokenizer = GPT2TokenizerFast.from_pretrained(GPT2_MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(GPT2_MODEL_NAME)
    model.eval()

    predictions: List[int] = []
    labels: List[int] = []
    correct_sum = 0
    correct_normalized = 0
    started = time.perf_counter()

    for index, example in enumerate(val_data, start=1):
        sums, counts = ending_log_prob_stats(
            model, tokenizer, example["ctx"], example["endings"]
        )
        normalized = [(s / c) if c else s for s, c in zip(sums, counts)]
        label = int(example["label"])
        predictions.append(int(np.argmax(sums)))
        labels.append(label)
        correct_sum += int(int(np.argmax(sums)) == label)
        correct_normalized += int(int(np.argmax(normalized)) == label)

        if index % PROGRESS_EVERY == 0:
            elapsed = time.perf_counter() - started
            print(
                f"[gpt2] {index}/{len(val_data)} | sum acc {correct_sum / index:.4f} "
                f"| norm acc {correct_normalized / index:.4f} "
                f"| {elapsed:.0f}s elapsed, ~{elapsed / index * (len(val_data) - index):.0f}s left",
                flush=True,
            )

    # write every 1,000 examples? no - single write at the end, plus incremental
    payload = {
        "model_name": GPT2_MODEL_NAME,
        "scoring": "sum of ending-token log probs",
        "num_examples": len(val_data),
        "accuracy": correct_sum / max(len(val_data), 1),
        "accuracy_length_normalized": correct_normalized / max(len(val_data), 1),
        "seconds": time.perf_counter() - started,
        "predictions": predictions,
        "labels": labels,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    print(
        f"[gpt2] done: sum acc {payload['accuracy']:.4f} | "
        f"norm acc {payload['accuracy_length_normalized']:.4f} | "
        f"{payload['seconds']:.0f}s -> {OUTPUT_PATH}",
        flush=True,
    )


if __name__ == "__main__":
    main()
