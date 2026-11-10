"""ARC-Challenge + ConceptNet preparation for the V3 controlled experiment (TASK 18).

Produces the four data artifacts both models share::

    environment/data/arc/train.json           ~1117 records  <- GPT-2 + GMN train
    environment/data/arc/val.json              ~295 records  <- model selection
    environment/data/arc/test.json            ~1165 records  <- held-out reporting
    environment/data/conceptnet/triples.json  100000 triples <- relational signal

ARC records have the schema fixed by TASKS_V3.md::

    {"question": str, "choices": [str, str, str, str], "label": int}

ConceptNet records::

    [{"subject": str, "relation": str, "object": str}]

Sources
-------
ARC
    Hub repo ``allenai/ai2_arc``, config ``ARC-Challenge``.  The bare alias
    ``ai2_arc`` that older docs use no longer resolves under ``datasets >= 5``
    (they require a ``namespace/name`` repo id), so the namespaced id is tried
    first.  ARC has a handful of 3-choice and 5-choice questions; those are
    dropped because the record schema for this experiment is a fixed 4-way
    choice.  ARC answer keys come in two flavours -- ``A``-``D`` *and* the
    legacy NYSED ``1``-``4`` -- so the answer is located by its position in
    ``choices["label"]``, which handles both without a special case.

ConceptNet
    ``conceptnet-assertions-5.7.0.csv.gz`` (the build the task names).  The
    gzip is 475 MB, so it is streamed to a local cache under
    ``environment/data/conceptnet/`` (resumable via HTTP ``Range``) and then
    parsed line by line -- it never has to fit in memory.  Kept triples are
    English, carry one of the eight relations below, and both endpoints must
    reduce to a clean natural-language term.  Surviving triples are reservoir
    sampled (Algorithm R, seeded) down to ``--conceptnet-target`` so the file
    does not inherit ConceptNet's heavily skewed relation/dataset ordering.

Usage (from the repository root)::

    python environment/data/loader.py                      # everything
    python environment/data/loader.py --arc-only
    python environment/data/loader.py --conceptnet-only --conceptnet-target 20000
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ``environment/`` is not a package root, so make ``import config`` work no
# matter which directory this script was launched from (mirrors the
# ``scripts/load_data.py`` convention of the earlier tasks).
ENVIRONMENT_DIR = Path(__file__).resolve().parents[1]
if str(ENVIRONMENT_DIR) not in sys.path:
    sys.path.insert(0, str(ENVIRONMENT_DIR))

import config  # noqa: E402


# ---------------------------------------------------------------------------
# ARC
# ---------------------------------------------------------------------------

#: Hub repo ids, tried in order (datasets >= 5 needs ``namespace/name``).
ARC_REPO_CANDIDATES: Tuple[str, ...] = ("allenai/ai2_arc", "ai2_arc")
ARC_CONFIG = "ARC-Challenge"

#: ARC splits are named ``train`` / ``validation`` / ``test``; we persist the
#: short names used everywhere else in this repository.
ARC_SPLIT_ALIASES: Dict[str, str] = {
    "train": "train",
    "val": "validation",
    "test": "test",
}

NUM_CHOICES = 4

# ---------------------------------------------------------------------------
# ConceptNet
# ---------------------------------------------------------------------------

CONCEPTNET_URL = (
    "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/"
    "conceptnet-assertions-5.7.0.csv.gz"
)

#: The eight relations TASKS_V3.md asks for.
CONCEPTNET_RELATIONS: Tuple[str, ...] = (
    "/r/Causes",
    "/r/IsA",
    "/r/HasProperty",
    "/r/PartOf",
    "/r/UsedFor",
    "/r/CapableOf",
    "/r/CausesDesire",
    "/r/HasPrerequisite",
)

CONCEPTNET_TARGET = 100_000
CONCEPTNET_CACHE_NAME = "conceptnet-assertions-5.7.0.csv.gz"

EN_PREFIX = "/c/en/"

DEFAULT_SEED = int(config.TRAIN["seed"])

#: A concept term is kept only if it looks like language, not a URI fragment.
TERM_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 '\-.]*$")
MAX_TERM_LEN = 40

#: ``IsA`` -> ``["Is", "A"]``, ``HasProperty`` -> ``["Has", "Property"]``.
_CAMEL_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")

USER_AGENT = "gmn-v3-data-loader/1.0"
DOWNLOAD_CHUNK = 4 * 1024 * 1024


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def save_json(records: Any, path: Path) -> None:
    """Write ``records`` as UTF-8 JSON, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    size_mb = path.stat().st_size / (1024 * 1024)
    count = len(records) if isinstance(records, list) else "-"
    print(f"  saved n={count} -> {path} ({size_mb:.1f} MB)")


def print_label_distribution(records: Sequence[Dict[str, Any]], name: str) -> None:
    """Print the 4-way label balance of an ARC split."""
    counts: Counter = Counter(int(record["label"]) for record in records)
    total = len(records)
    print(f"\n  {name} label distribution (n={total}):")
    for label in range(NUM_CHOICES):
        count = counts.get(label, 0)
        pct = 100.0 * count / total if total else 0.0
        print(f"    label {label}: {count:>5d}  ({pct:5.1f}%)  {'#' * int(round(pct / 2))}")



# ---------------------------------------------------------------------------
# ARC
# ---------------------------------------------------------------------------
def load_arc_split(split_name: str, repo_candidates: Sequence[str] = ARC_REPO_CANDIDATES):
    """Load one ARC-Challenge split from the first repo id that resolves."""
    last_error: Optional[Exception] = None
    for repo_id in repo_candidates:
        try:
            from datasets import load_dataset

            split = load_dataset(repo_id, ARC_CONFIG, split=split_name)
            print(f"  using dataset repo: {repo_id} (config={ARC_CONFIG}, split={split_name})")
            return split
        except Exception as exc:  # noqa: BLE001 - fall through to the next candidate
            last_error = exc
            print(f"  could not load {repo_id!r} ({type(exc).__name__}), trying next ...")
    raise RuntimeError(
        f"unable to load ARC split {split_name!r} from {tuple(repo_candidates)}"
    ) from last_error


def resolve_answer_index(answer_key: Any, choice_labels: Sequence[Any]) -> Optional[int]:
    """Map an ARC ``answerKey`` onto a 0-based choice index.

    Two encodings occur in ARC-Challenge: letters (``"A"``-``"D"``) and the
    legacy NYSED digits (``"1"``-``"4"``).  Locating the key inside
    ``choices["label"]`` resolves both, and the letter/digit fallbacks cover
    rows whose label list is missing or inconsistent.
    """
    if answer_key is None:
        return None

    # Primary path: the answer key is one of the choice labels, in order.
    as_text = str(answer_key).strip()
    for index, label in enumerate(choice_labels):
        if str(label).strip() == as_text:
            return index

    # Fallbacks for the two documented encodings.
    if as_text.isdigit():
        index = int(as_text) - 1  # "1" -> 0
        return index if index >= 0 else None
    if len(as_text) == 1 and as_text.upper() in "ABCDEFG":
        return ord(as_text.upper()) - ord("A")
    return None


def build_arc_records(split) -> Tuple[List[Dict[str, Any]], Counter]:
    """Convert a raw ARC-Challenge split into the fixed V3 record schema.

    Returns ``(records, dropped)`` where ``dropped`` counts rows removed by
    reason, so the caller can report exactly what was discarded.
    """
    records: List[Dict[str, Any]] = []
    dropped: Counter = Counter()

    for row in split:
        texts = [str(text) for text in row["choices"]["text"]]
        labels = list(row["choices"]["label"])

        if len(texts) != NUM_CHOICES:
            dropped[f"not_{NUM_CHOICES}_choices"] += 1
            continue

        index = resolve_answer_index(row["answerKey"], labels)
        if index is None or not 0 <= index < NUM_CHOICES:
            dropped["unresolved_answer_key"] += 1
            continue

        question = str(row["question"]).strip()
        choices = [text.strip() for text in texts]
        if not question or any(not choice for choice in choices):
            dropped["empty_text"] += 1
            continue

        records.append({"question": question, "choices": choices, "label": int(index)})

    return records, dropped


def prepare_arc() -> Dict[str, List[Dict[str, Any]]]:
    """Download/convert all three ARC-Challenge splits and persist them."""
    print("ARC-Challenge (allenai/ai2_arc) ...")
    prepared: Dict[str, List[Dict[str, Any]]] = {}

    for short_name, arc_split in ARC_SPLIT_ALIASES.items():
        split = load_arc_split(arc_split)
        records, dropped = build_arc_records(split)

        total_dropped = sum(dropped.values())
        detail = ", ".join(f"{reason}={count}" for reason, count in sorted(dropped.items()))
        print(
            f"  {short_name:<5s}: kept {len(records):>5d} / {len(split):>5d}"
            f"  (dropped {total_dropped}{': ' + detail if detail else ''})"
        )

        save_json(records, config.path(f"arc_{short_name}"))
        print_label_distribution(records, short_name)
        prepared[short_name] = records

    return prepared



# ---------------------------------------------------------------------------
# ConceptNet
# ---------------------------------------------------------------------------
def conceptnet_cache_path() -> Path:
    """Local path of the downloaded gzip (kept next to the filtered triples)."""
    return config.path("conceptnet").parent / CONCEPTNET_CACHE_NAME


def _remote_size(url: str) -> Optional[int]:
    """``Content-Length`` of ``url`` via HEAD, or ``None`` if unavailable."""
    try:
        request = urllib.request.Request(
            url, method="HEAD", headers={"User-Agent": USER_AGENT}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:  # noqa: BLE001 - size probing is best-effort
        return None


def download_conceptnet(
    url: str = CONCEPTNET_URL,
    dest: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """Stream the ConceptNet gzip to ``dest``, resuming a partial download.

    S3 serves ``Range`` requests, so an interrupted run continues where it
    stopped instead of restarting 475 MB.
    """
    dest = Path(dest) if dest is not None else conceptnet_cache_path()
    dest.parent.mkdir(parents=True, exist_ok=True)

    remote_size = _remote_size(url)
    local_size = dest.stat().st_size if dest.exists() else 0

    if force:
        local_size = 0
    elif remote_size is not None and local_size == remote_size:
        print(f"  cached raw file is complete: {dest} ({local_size / 1e6:.1f} MB)")
        return dest
    elif remote_size is not None and local_size > remote_size:
        print("  cached raw file is larger than the remote file, restarting ...")
        local_size = 0

    headers = {"User-Agent": USER_AGENT}
    if local_size:
        headers["Range"] = f"bytes={local_size}-"
        progress = f" / {remote_size / 1e6:.1f} MB" if remote_size else ""
        print(f"  resuming download at {local_size / 1e6:.1f} MB{progress}")
    else:
        print(f"  downloading {url}")

    request = urllib.request.Request(url, headers=headers)
    started = time.perf_counter()

    with urllib.request.urlopen(request, timeout=180) as response:
        # A server that ignores Range answers 200; then the body is the whole
        # file and the partial bytes must be discarded.
        if local_size and getattr(response, "status", 200) == 200:
            print("  server ignored Range, restarting from byte 0 ...")
            local_size = 0

        mode = "ab" if local_size else "wb"
        written = local_size
        total = remote_size or (int(response.headers.get("Content-Length", 0)) + local_size)
        next_report = written + 50 * 1000 * 1000

        with dest.open(mode) as handle:
            while True:
                chunk = response.read(DOWNLOAD_CHUNK)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
                if written >= next_report:
                    elapsed = time.perf_counter() - started
                    rate = (written - local_size) / max(elapsed, 1e-9) / 1e6
                    pct = 100.0 * written / total if total else 0.0
                    print(
                        f"    {written / 1e6:7.1f} MB / {total / 1e6:.1f} MB"
                        f"  ({pct:5.1f}%)  {rate:.2f} MB/s",
                        flush=True,
                    )
                    next_report += 50 * 1000 * 1000

    elapsed = time.perf_counter() - started
    print(
        f"  download finished: {written / 1e6:.1f} MB in {elapsed:.1f}s"
        f" ({written / max(elapsed, 1e-9) / 1e6:.2f} MB/s) -> {dest}"
    )
    return dest



def concept_term(uri: str) -> Optional[str]:
    """``/c/en/fire/n/01`` -> ``"fire"``; ``None`` if not a clean English term."""
    if not uri.startswith(EN_PREFIX):
        return None
    term = uri[len(EN_PREFIX):].split("/", 1)[0]
    if not term:
        return None
    text = term.replace("_", " ").strip()
    if not text or len(text) > MAX_TERM_LEN or not TERM_RE.match(text):
        return None
    return text


def relation_text(relation: str) -> Optional[str]:
    """``/r/HasProperty`` -> ``"has property"``; ``None`` if not one of the eight."""
    if relation not in CONCEPTNET_RELATIONS:
        return None
    name = relation[len("/r/"):]
    words = _CAMEL_RE.findall(name)
    return " ".join(word.lower() for word in words) if words else None


def iter_conceptnet_triples(csv_path: Path) -> Iterable[Tuple[str, str, str]]:
    """Yield ``(subject, relation, object)`` triples from the assertions gzip.

    ConceptNet assertion lines are tab separated::

        /a/[/r/IsA/,/c/en/dog/n/1,/c/en/animal/n/1] /r/IsA /c/en/dog/n/1 /c/en/animal/n/1 {...}

    The assertion-URI and JSON-metadata columns are ignored; only the relation
    and the two endpoint URIs are needed.
    """
    kept = 0
    seen_lines = 0
    with gzip.open(csv_path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            seen_lines += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                continue

            relation = relation_text(fields[1])
            if relation is None:
                continue

            subject = concept_term(fields[2])
            obj = concept_term(fields[3])
            if subject is None or obj is None:
                continue

            kept += 1
            yield subject, relation, obj
    print(f"  scanned {seen_lines:,} assertion lines, kept {kept:,} English triples")


def reservoir_sample(
    triples: Iterable[Tuple[str, str, str]],
    target: int,
    seed: int = DEFAULT_SEED,
) -> List[Tuple[str, str, str]]:
    """Uniformly sample ``target`` items from a stream in one pass (Algorithm R).

    ConceptNet's file order is grouped by dataset/relation.  Taking the first N
    triples would bake that skew into the training signal, so a seeded reservoir
    is used instead: memory stays O(target) and the result is deterministic.
    """
    rng = random.Random(seed)
    reservoir: List[Tuple[str, str, str]] = []
    for index, triple in enumerate(triples):
        if index < target:
            reservoir.append(triple)
        else:
            slot = rng.randint(0, index)
            if slot < target:
                reservoir[slot] = triple
    return reservoir


def dedupe_triples(triples: Sequence[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    """Drop repeated ``(subject, relation, object)`` triples, preserving order."""
    seen = set()
    unique: List[Tuple[str, str, str]] = []
    for triple in triples:
        if triple in seen:
            continue
        seen.add(triple)
        unique.append(triple)
    return unique


def prepare_conceptnet(
    target: int = CONCEPTNET_TARGET,
    seed: int = DEFAULT_SEED,
    force_download: bool = False,
    csv_path: Optional[Path] = None,
) -> List[Dict[str, str]]:
    """Download, filter, sample and persist the ConceptNet triples."""
    print("ConceptNet 5.7 assertions ...")
    csv_path = Path(csv_path) if csv_path is not None else download_conceptnet(force=force_download)

    print(
        "  filtering: English (/c/en/), relations = "
        + ", ".join(relation[len("/r/"):] for relation in CONCEPTNET_RELATIONS)
    )
    sampled = reservoir_sample(iter_conceptnet_triples(csv_path), target, seed)
    sampled = dedupe_triples(sampled)
    # Stable order keeps the artifact diffable and independent of stream order.
    sampled.sort()

    records = [
        {"subject": subject, "relation": relation, "object": obj}
        for subject, relation, obj in sampled
    ]
    save_json(records, config.path("conceptnet"))

    relation_counts: Counter = Counter(record["relation"] for record in records)
    print(f"\n  relation breakdown (n={len(records)}):")
    for relation, count in relation_counts.most_common():
        print(f"    {relation:<20s} {count:>7d}  ({100.0 * count / len(records):5.1f}%)")

    return records


# ---------------------------------------------------------------------------
# Reporting / entry point
# ---------------------------------------------------------------------------
def docstring_first_line() -> str:
    """First line of this module's docstring, used as the CLI description."""
    return (__doc__ or "").strip().splitlines()[0]


def print_arc_example(record: Dict[str, Any], title: str = "sample ARC example") -> None:
    """Pretty-print one ARC record so the schema can be eyeballed."""
    print(f"\n  [{title}]")
    print(f"    question : {record['question']}")
    for index, choice in enumerate(record["choices"]):
        marker = " <-- correct" if index == record["label"] else ""
        print(f"    choice[{index}] : {choice}{marker}")
    print(f"    label    : {record['label']}")


def print_triples(triples: Sequence[Dict[str, str]], count: int = 5) -> None:
    """Show a few triples in both stored and TASK 19 'sentence' form."""
    print("\n  [sample ConceptNet triples]")
    for record in triples[:count]:
        sentence = f"{record['subject']} {record['relation']} {record['object']}"
        print(
            f"    {record['subject']} | {record['relation']} | {record['object']}"
            f'   -> "{sentence}"'
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=docstring_first_line())
    parser.add_argument("--arc-only", action="store_true", help="skip ConceptNet")
    parser.add_argument("--conceptnet-only", action="store_true", help="skip ARC")
    parser.add_argument(
        "--conceptnet-target",
        type=int,
        default=CONCEPTNET_TARGET,
        help=f"triples to keep (default {CONCEPTNET_TARGET})",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="re-download the ConceptNet gzip even if cached",
    )
    parser.add_argument(
        "--conceptnet-csv",
        type=Path,
        default=None,
        help="use an existing conceptnet-assertions-*.csv.gz instead of downloading",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="reservoir sampling seed")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.arc_only and args.conceptnet_only:
        raise SystemExit("--arc-only and --conceptnet-only are mutually exclusive")

    started = time.perf_counter()
    config.ensure_dirs()

    print("=" * 74)
    print("TASK 18 - V3 environment data preparation")
    print("=" * 74)
    print(f"repo root: {config.PROJECT_ROOT}\n")

    arc_splits: Dict[str, List[Dict[str, Any]]] = {}
    triples: List[Dict[str, str]] = []

    if not args.conceptnet_only:
        arc_splits = prepare_arc()
        print()

    if not args.arc_only:
        triples = prepare_conceptnet(
            target=args.conceptnet_target,
            seed=args.seed,
            force_download=args.force_download,
            csv_path=args.conceptnet_csv,
        )
        print()

    print("=" * 74)
    print("SUMMARY")
    print("=" * 74)
    if arc_splits:
        print("ARC-Challenge counts:")
        for name in ARC_SPLIT_ALIASES:
            print(f"  {name:<5s}: {len(arc_splits[name]):>5d} examples")
        print("ARC label balance:")
        for name in ARC_SPLIT_ALIASES:
            records = arc_splits[name]
            counts = Counter(record["label"] for record in records)
            shares = "  ".join(
                f"{label}:{100.0 * counts.get(label, 0) / len(records):5.1f}%"
                for label in range(NUM_CHOICES)
            )
            print(f"  {name:<5s}: {shares}")
        print_arc_example(arc_splits["train"][0])
    if triples:
        print(f"\nConceptNet triples: {len(triples)}")
        print_triples(triples)

    print(f"\nDone in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

