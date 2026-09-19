"""Download HellaSwag and persist a compact JSON version used by the GMN pipeline.

TASK 1 output:
    data/train.json  -> first 2000 examples of the train split
    data/val.json    -> first 500 examples of the validation split

Every record has the shape::

    {
        "ctx": str,              # context the ending must continue
        "endings": [str, str, str, str],
        "label": int,            # index (0-3) of the correct ending
        "activity_label": str,   # e.g. "Removing ice from car"
    }
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datasets import load_dataset

try:  # tqdm is optional, the script must still run without it
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    def tqdm(iterable: Iterable[Any], **_kwargs: Any) -> Iterable[Any]:
        return iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# "Rowan/hellaswag" is the canonical Hub repo. Newer `datasets` releases (>= 5)
# require a "namespace/name" id, so the bare "hellaswag" alias is kept only as a
# fallback for older environments.
DATASET_CANDIDATES = ("Rowan/hellaswag", "hellaswag")
TRAIN_SPLIT = "train"
VAL_SPLIT = "validation"

TRAIN_SIZE = 2000
VAL_SIZE = 500

NUM_ENDINGS = 4


def load_split(split_name: str):
    """Load ``split_name`` from the first dataset repo id that resolves."""
    last_error: Optional[Exception] = None
    for dataset_name in DATASET_CANDIDATES:
        try:
            split = load_dataset(dataset_name, split=split_name)
            print(f"  using dataset repo: {dataset_name}")
            return split
        except Exception as exc:  # noqa: BLE001 - try the next candidate
            last_error = exc
            print(f"  could not load {dataset_name!r} ({type(exc).__name__}), trying next ...")
    raise RuntimeError(
        f"unable to load HellaSwag split {split_name!r} from {DATASET_CANDIDATES}"
    ) from last_error


def build_records(split, limit: int) -> List[Dict[str, Any]]:
    """Convert the first ``limit`` rows of a HellaSwag split into plain dicts."""
    records: List[Dict[str, Any]] = []
    # HellaSwag is shuffle-free and deterministic; the first ``limit`` rows are
    # simply the head of the split, so slicing is exact and reproducible.
    for row in tqdm(split.select(range(limit)), total=limit, desc="converting"):
        endings = list(row["endings"])
        label = int(row["label"])  # HF stores the label as a string ("0".."3")

        if len(endings) != NUM_ENDINGS:
            raise ValueError(
                f"expected {NUM_ENDINGS} endings, got {len(endings)} "
                f"for source_id={row.get('source_id')!r}"
            )
        if not 0 <= label < NUM_ENDINGS:
            raise ValueError(f"label {label} out of range for {row.get('source_id')!r}")

        records.append(
            {
                "ctx": str(row["ctx"]),
                "endings": endings,
                "label": label,
                "activity_label": str(row["activity_label"]),
            }
        )
    return records


def save_json(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    print(f"  saved {len(records):>5d} examples -> {path}")


def print_example(index: int, record: Dict[str, Any]) -> None:
    """Pretty-print one full record so the structure can be eyeballed."""
    print(f"\n  [example {index}]")
    print(f"    activity_label : {record['activity_label']}")
    print(f"    label          : {record['label']}")
    print(f"    ctx            : {record['ctx']}")
    for i, ending in enumerate(record["endings"]):
        marker = " <-- correct" if i == record["label"] else ""
        print(f"    ending[{i}]      : {ending}{marker}")


def print_label_distribution(records: List[Dict[str, Any]], name: str) -> None:
    counts = Counter(record["label"] for record in records)
    total = len(records)
    print(f"\n  {name} label distribution (n={total}):")
    for label in range(NUM_ENDINGS):
        count = counts.get(label, 0)
        pct = 100.0 * count / total if total else 0.0
        bar = "#" * int(round(pct / 2))
        print(f"    label {label}: {count:>5d}  ({pct:5.1f}%)  {bar}")


def main() -> None:
    print(f"Loading HellaSwag from HuggingFace datasets {DATASET_CANDIDATES} ...")
    train_split = load_split(TRAIN_SPLIT)
    val_split = load_split(VAL_SPLIT)
    print(f"  full train split      : {len(train_split)} examples")
    print(f"  full validation split : {len(val_split)} examples")

    print(f"\nKeeping first {TRAIN_SIZE} train / {VAL_SIZE} validation examples ...")
    train_records = build_records(train_split, TRAIN_SIZE)
    val_records = build_records(val_split, VAL_SIZE)

    print("\nWriting JSON:")
    save_json(train_records, DATA_DIR / "train.json")
    save_json(val_records, DATA_DIR / "val.json")

    print(f"\nFirst 2 {TRAIN_SPLIT} examples:")
    for index in range(2):
        print_example(index, train_records[index])

    print_label_distribution(train_records, "train")
    print_label_distribution(val_records, "val")

    print("\nData loading complete.")


if __name__ == "__main__":
    main()
