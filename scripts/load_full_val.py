"""Load the FULL HellaSwag validation split (10,042 examples) -> data/val_full.json.

Same record schema as ``data/val.json`` (TASK 1), just the whole split. Needed to
resolve whether the navigation component helps: at n=500 the unpaired standard
error is 2.11 points, which cannot separate a 4-point effect. At n=10,042 it drops
to ~0.47 points.
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

OUTPUT_PATH = PROJECT_ROOT / "data" / "val_full.json"


def main() -> None:
    started = time.perf_counter()

    print("Loading the FULL HellaSwag validation split ...")
    val_split = load_data.load_split(load_data.VAL_SPLIT)
    total = len(val_split)
    print(f"  full validation split: {total} examples")

    records = load_data.build_records(val_split, total)

    print("\nWriting JSON:")
    load_data.save_json(records, OUTPUT_PATH)
    load_data.print_label_distribution(records, "val_full")

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(
        f"\nSaved {len(records)} examples to {OUTPUT_PATH} ({size_mb:.1f} MB) "
        f"in {time.perf_counter() - started:.1f}s"
    )


if __name__ == "__main__":
    main()
