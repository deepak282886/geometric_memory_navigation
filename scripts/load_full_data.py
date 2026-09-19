"""Load the FULL HellaSwag train split (39,905 examples) -> data/train_full.json.

Same record schema as ``data/train.json`` from TASK 1 (``ctx``, ``endings``,
``label``, ``activity_label``), just the whole split instead of the first 2000.

Feeds the bigger-memory experiment: more memories give the sampler more
directions to choose from, at zero extra parameters and no retraining.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import load_data  # sibling script: reuse the exact TASK 1 record-building logic  # noqa: E402

OUTPUT_PATH = PROJECT_ROOT / "data" / "train_full.json"


def main() -> None:
    started = time.perf_counter()

    print("Loading the FULL HellaSwag train split ...")
    train_split = load_data.load_split(load_data.TRAIN_SPLIT)
    total = len(train_split)
    print(f"  full train split: {total} examples")

    print(f"\nConverting all {total} examples ...")
    records = load_data.build_records(train_split, total)

    print("\nWriting JSON:")
    load_data.save_json(records, OUTPUT_PATH)

    print(f"\nFirst 2 {load_data.TRAIN_SPLIT} examples:")
    for index in range(2):
        load_data.print_example(index, records[index])

    load_data.print_label_distribution(records, "train_full")

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(
        f"\nSaved {len(records)} examples to {OUTPUT_PATH} ({size_mb:.1f} MB) "
        f"in {time.perf_counter() - started:.1f}s"
    )


if __name__ == "__main__":
    main()
