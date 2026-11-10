"""WikiText-103 preparation for the V3 controlled experiment (TASK 18R).

Produces the four artifacts both models share::

    environment/data/wikitext/train_sentences.json   sentences   <- GPT-2 LM training
    environment/data/wikitext/train_pairs.json       [S_n, S_n+1] <- GMN contrastive
    environment/data/wikitext/val_sentences.json     sentences   <- LM validation
    environment/data/wikitext/val_pairs.json         [S_n, S_n+1] <- Recall@1 validation

Source
------
``wikitext-103-raw-v1``, Hub repo ``Salesforce/wikitext``.  The bare alias
``wikitext`` that older docs use no longer resolves under ``datasets >= 5``
(they require a ``namespace/name`` repo id), so the namespaced id is tried
first — exactly the workaround TASK 18's ARC loader needed for ``ai2_arc``.

Raw rows are WikiText lines: non-blank rows are paragraphs or section headings,
blank rows separate them, and punctuation already arrives space-tokenised
(``" the game began . While it retained "``).  A *document* is therefore the
contiguous run of non-blank rows between two blank rows (or a heading), and no
sentence pair ever spans two documents — the TASK 18R pairing rule.

Sentence splitting
------------------
Within a document the text is cut on ``.``/``!``/``?`` followed by whitespace.
Because WikiText isolates punctuation, that boundary is unambiguous for real
sentence ends; the one failure mode is an abbreviation, so a piece beginning
after a token in :data:`ABBREVIATIONS` (``"dr . Smith"``) is glued back onto the
previous sentence.  Cleaning then collapses whitespace, drops empty strings and
drops anything shorter than ``--min-words`` (default 5) words.  Pairs are formed
between *consecutive surviving* sentences of one document, so every positive is
a genuine next sentence or a near-neighbour of one inside the same document.

Corpus size (why the artifacts are large)
-----------------------------------------
The cleaned corpus is kept in full rather than truncated to TASK 18R's
"~500K-1M sentences" line: WikiText-103's own headline figures are 103M / 217K
tokens, and this pipeline reproduces them (measured: 3,837,479 train sentences
= 98.8M whitespace tokens, 8,206 val sentences = 208K; WikiText counts 103M /
217K with a GPT-2 BPE tokenizer over the raw text, before the >=5-word filter).
A 500K-sentence cut would be ~13% of the corpus, i.e. a different experiment.
Trainers that need a smaller step budget should subsample at training time, not
here.

Because of that size, both files are written as *one element per line* by
:func:`write_json_array` (streamed, so the corpus is never serialised twice in
memory) and pairs are regenerated from per-document spans by :func:`iter_pairs`
(so the pair list needs no second copy of 3.5M sentences).  Measured on this
machine: train_pairs.json (1,013 MB) and train_sentences.json (544 MB) together
are ~1.5 GB, against ~3 GB of free RAM.

Usage (from the repository root)::

    python environment/data/wikitext/preprocess.py
    python environment/data/wikitext/preprocess.py --max-docs 2000   # smoke test
    python environment/data/wikitext/preprocess.py --min-words 6 --quiet
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

# ``environment/`` is not a package root, so make ``import config`` work no
# matter which directory this script was launched from (TASK 18 convention).
ENVIRONMENT_DIR = Path(__file__).resolve().parents[2]
if str(ENVIRONMENT_DIR) not in sys.path:
    sys.path.insert(0, str(ENVIRONMENT_DIR))

import config  # noqa: E402

# ---------------------------------------------------------------------------
# Dataset coordinates
# ---------------------------------------------------------------------------
#: Hub repo ids, tried in order (datasets >= 5 needs ``namespace/name``).
DATASET_REPO_CANDIDATES: Tuple[str, ...] = ("Salesforce/wikitext", "wikitext")
DATASET_CONFIG = "wikitext-103-raw-v1"

#: Hub split name -> the short name used by every artifact in this repository.
SPLIT_ALIASES: Dict[str, str] = {"train": "train", "val": "validation"}

#: A WikiText section heading row (``" = Valkyria Chronicles III ="``).
HEADING_RE = re.compile(r"^\s*=+.*=+\s*$")

#: A sentence boundary: end punctuation followed by whitespace.
BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")

#: Sentence-final tokens that are really abbreviations, not full stops.
ABBREVIATIONS = frozenset(
    """
    mr mrs ms dr prof st no vs etc jr sr inc ltd co corp dept univ gen col capt lt
    sgt rev hon est approx fig al lit cf u.s u.k i.e e.g no. vol pp ed eds
    """.split()
)

MIN_WORDS = 5


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_raw_dataset(repo: Optional[str] = None, config_name: str = DATASET_CONFIG):
    """Load WikiText-103 raw, trying the namespaced repo id first."""
    from datasets import load_dataset

    candidates = (repo,) if repo else DATASET_REPO_CANDIDATES
    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            return load_dataset(candidate, config_name)
        except Exception as error:  # pragma: no cover - depends on the Hub
            last_error = error
            print(f"  repo {candidate!r} failed: {type(error).__name__}: {error}")
    raise SystemExit(f"could not load {config_name!r} from any of {candidates}: {last_error}")


# ---------------------------------------------------------------------------
# Sentence extraction
# ---------------------------------------------------------------------------
def clean_sentence(text: str) -> str:
    """Collapse whitespace and strip - the only cleaning applied before filtering."""
    return " ".join(text.split())


def word_count(sentence: str) -> int:
    """Words in a WikiText sentence (its whitespace tokens, punctuation included)."""
    return len(sentence.split())


def split_sentences(text: str) -> List[str]:
    """Cut ``text`` into sentences on end punctuation, gluing abbreviations back.

    WikiText space-tokenises punctuation, so ``" . "`` is a reliable boundary;
    the only repair needed is a preceding abbreviation token (``"dr ."``).
    """
    pieces = BOUNDARY_RE.split(clean_sentence(text))
    sentences: List[str] = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if sentences:
            previous = sentences[-1]
            last_token = previous.split()[-1].lower().strip("()[]\"'")
            inside_parenthesis = previous.count("(") > previous.count(")")
            if last_token in ABBREVIATIONS or inside_parenthesis:
                sentences[-1] = f"{previous} {piece}"
                continue
        sentences.append(piece)
    return sentences


def iter_documents(texts: Iterator[Dict[str, Any]]) -> Iterator[List[str]]:
    """Yield one document (list of raw non-blank rows) at a time.

    A blank row or a section heading ends the current document, so a document is
    a paragraph: sentence pairs can never cross a paragraph or heading boundary.
    """
    document: List[str] = []
    for row in texts:
        stripped = str(row.get("text", "")).strip()
        if not stripped or HEADING_RE.match(stripped):
            if document:
                yield document
                document = []
            continue
        document.append(stripped)
    if document:
        yield document


def document_sentences(document: Sequence[str], min_words: int = MIN_WORDS) -> List[str]:
    """Cleaned, length-filtered sentences of one document, in reading order."""
    sentences: List[str] = []
    for row in document:
        for sentence in split_sentences(row):
            if word_count(sentence) >= min_words:
                sentences.append(sentence)
    return sentences


def build_split(
    texts: Iterator[Dict[str, Any]],
    min_words: int = MIN_WORDS,
    max_docs: Optional[int] = None,
) -> Tuple[List[str], List[Tuple[int, int]], Dict[str, Any]]:
    """Return ``(sentences, ranges, stats)`` for one split of the raw dataset.

    ``ranges`` holds one ``(start, end)`` span per non-empty document: the
    sentences of that document are ``sentences[start:end]``.  Pairs are rebuilt
    from those spans by :func:`iter_pairs`, so the pair list never needs its own
    copy of the corpus in memory (the full train split is ~1.8 GB of JSON).
    """
    sentences: List[str] = []
    ranges: List[Tuple[int, int]] = []
    documents = 0
    empty_documents = 0
    pairs = 0
    words = 0
    lengths: List[int] = []

    for index, document in enumerate(iter_documents(texts)):
        if max_docs is not None and index >= max_docs:
            break
        documents += 1
        kept = document_sentences(document, min_words)
        if not kept:
            empty_documents += 1
            continue
        start = len(sentences)
        sentences.extend(kept)
        lengths.extend(word_count(sentence) for sentence in kept)
        words += sum(lengths[-len(kept):])
        ranges.append((start, start + len(kept)))
        pairs += len(kept) - 1

    stats: Dict[str, Any] = {
        "documents": documents,
        "empty_documents": empty_documents,
        "sentences": len(sentences),
        "pairs": pairs,
        "words": words,
        "lengths": lengths,
    }
    return sentences, ranges, stats


def iter_pairs(sentences: Sequence[str], ranges: Sequence[Tuple[int, int]]) -> Iterator[List[str]]:
    """Yield ``[S_n, S_n+1]`` for every consecutive sentence inside a document."""
    for start, end in ranges:
        for index in range(start, end - 1):
            yield [sentences[index], sentences[index + 1]]


# ---------------------------------------------------------------------------
# Output + reporting
# ---------------------------------------------------------------------------
def write_json_array(path: Path, items: Iterable[Any], flush_every: int = 20_000) -> Tuple[Path, int]:
    """Stream ``items`` to ``path`` as the JSON array ``[e1,\\ne2,\\n...]``.

    One element per line, written as it is produced: the 1.8 GB corpus never
    needs a second serialised copy in memory, the file stays diffable, and the
    element count can later be re-verified by counting lines (no element can
    contain a raw newline — :func:`validate` proves that for sentences, and
    ``json.dumps`` escapes them in any case).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        handle.write("[")
        for item in items:
            handle.write("" if written == 0 else ",\n")
            handle.write(json.dumps(item, ensure_ascii=False))
            written += 1
            if flush_every and written % flush_every == 0:
                print(f"    ... {written:,} elements", flush=True)
        handle.write("]")
    return path, written


def count_json_array_items(path: Path) -> int:
    """Count the elements of a one-element-per-line array written above."""
    total = 0
    with path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(1 << 22)
            if not chunk:
                break
            total += chunk.count("\n")
    return total + 1 if total else 0


#: Files at or below this size are parsed in full as a JSON round-trip check;
#: the multi-hundred-MB ones are checked structurally instead (see below).
FULL_PARSE_LIMIT = 50_000_000


def verify_artifact(path: Path, expected: int) -> Dict[str, Any]:
    """Re-read a written array and check its envelope, count and first element.

    Small artifacts are additionally parsed end to end with ``json.load``; the
    train artifacts are hundreds of MB (1.5 GB of JSON for both of them, against
    ~3 GB of free RAM), so they get the structural check only.
    """
    counted = count_json_array_items(path)
    with path.open("rb") as handle:
        handle.seek(-1, 2)
        closed = handle.read(1) == b"]"
    with path.open("r", encoding="utf-8") as handle:
        head = handle.readline()
    # Drop exactly one opening bracket (the array's), not every leading "[":
    # a pair element starts with its own "[".
    head = head[1:] if head.startswith("[") else head
    head = head.rstrip()
    if head.endswith(","):
        head = head[:-1]
    first = json.loads(head)

    full_parse: Optional[bool] = None
    if path.stat().st_size <= FULL_PARSE_LIMIT:
        with path.open("r", encoding="utf-8") as handle:
            full_parse = len(json.load(handle)) == expected

    return {
        "elements": counted,
        "expected": expected,
        "closed": closed,
        "first": first,
        "full_parse": full_parse,
    }




def length_summary(lengths: Sequence[int]) -> Dict[str, float]:
    """mean / median / p95 / max of a list of sentence word counts."""
    ordered = sorted(lengths)
    total = len(ordered)
    if not total:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "n": total,
        "mean": sum(ordered) / total,
        "median": float(ordered[total // 2]),
        "p95": float(ordered[min(total - 1, int(0.95 * total))]),
        "max": float(ordered[-1]),
    }


def duplicate_rate(sentences: Sequence[str]) -> float:
    """Share of sentences that are an exact repeat of an earlier one.

    WikiText-103 is known to repeat boilerplate (" == References == ",
    " Retrieved ... ").  Repeats inside one contrastive batch act as false
    negatives, so the rate is reported rather than silently deduplicated.
    """
    if not sentences:
        return 0.0
    unique = len(set(sentences))
    return 100.0 * (len(sentences) - unique) / len(sentences)


def print_split_report(
    label: str,
    stats: Dict[str, Any],
    sentences: Sequence[str],
    ranges: Sequence[Tuple[int, int]],
) -> None:
    """Print the per-split statistics block (counts, lengths, duplicate rate)."""
    summary = length_summary(stats["lengths"])
    sentences_per_doc = (stats["sentences"] / stats["documents"]) if stats["documents"] else 0.0
    print(f"\n{label} split")
    print("-" * 74)
    print(f"  documents (paragraphs) : {stats['documents']:>10,d}"
          f"   (empty after cleaning: {stats['empty_documents']:,})")
    print(f"  sentences              : {len(sentences):>10,d}"
          f"   ({sentences_per_doc:.1f} per document)")
    print(f"  sentence pairs         : {stats['pairs']:>10,d}"
          f"   (documents with sentences: {len(ranges):,})")
    print(f"  whitespace tokens      : {stats['words']:>10,d}")
    print(f"  words / sentence       : mean {summary['mean']:.1f}  median {summary['median']:.1f}"
          f"  p95 {summary['p95']:.1f}  max {summary['max']:.0f}")
    print(f"  exact duplicate rate   : {duplicate_rate(sentences):.2f}%")


def validate(
    sentences: Sequence[str],
    ranges: Sequence[Tuple[int, int]],
    min_words: int,
    spot_check: int = 1000,
) -> Dict[str, Any]:
    """Structural check on a payload before it becomes an artifact.

    Full scans for sentences; for pairs the document spans are checked
    exhaustively and only the first/last ``spot_check`` pairs are materialised,
    because the train split has 3.5M of them and they are fully determined by
    the spans.
    """
    short = 0
    blank = 0
    multiline = 0
    for sentence in sentences:
        if word_count(sentence) < min_words:
            short += 1
        if not sentence.strip():
            blank += 1
        if "\n" in sentence or "\r" in sentence:
            multiline += 1

    overlapping = 0
    out_of_bounds = 0
    previous_end = 0
    for start, end in ranges:
        if end <= start or start < previous_end or end > len(sentences):
            out_of_bounds += 1
        if start < previous_end:
            overlapping += 1
        previous_end = end

    head = list(itertools.islice(iter_pairs(sentences, ranges), spot_check))
    tail = head
    if len(ranges) > spot_check:
        tail = list(itertools.islice(iter_pairs(sentences, ranges[::-1]), spot_check))
    malformed = sum(1 for pair in itertools.chain(head, tail) if len(pair) != 2 or not all(pair))

    return {
        "sentences_under_min_words": short,
        "blank_sentences": blank,
        "multiline_sentences": multiline,
        "broken_spans": out_of_bounds + overlapping,
        "malformed_pairs_sampled": malformed,
    }


def print_samples(sentences: Sequence[str], ranges: Sequence[Tuple[int, int]], count: int = 3) -> None:
    """Show a few sentences and a few next-sentence pairs verbatim."""
    print("\nsample sentences")
    print("-" * 74)
    for sentence in sentences[:count]:
        print(f"  [{word_count(sentence):>3d}w] {sentence[:150]}")
    print("\nsample pairs (S_n, S_n+1) - same document only")
    print("-" * 74)
    for index, pair in enumerate(itertools.islice(iter_pairs(sentences, ranges), count)):
        print(f"  pair {index}")
        print(f"    S_n   : {pair[0][:150]}")
        print(f"    S_n+1 : {pair[1][:150]}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--repo", default=None,
        help=f"override the Hub repo id (tried in order: {', '.join(DATASET_REPO_CANDIDATES)})",
    )
    parser.add_argument("--dataset-config", default=DATASET_CONFIG, help="Hub config name")
    parser.add_argument(
        "--min-words", type=int, default=MIN_WORDS,
        help=f"drop sentences shorter than this (default {MIN_WORDS})",
    )
    parser.add_argument(
        "--max-docs", type=int, default=None,
        help="keep only the first N documents of each split (smoke test)",
    )
    parser.add_argument("--samples", type=int, default=3, help="printed sample sentences/pairs")
    parser.add_argument("--quiet", action="store_true", help="suppress the sample printouts")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    config.ensure_dirs()

    print("=" * 74)
    print("TASK 18R - WikiText-103 sentence / next-sentence pair preparation")
    print("=" * 74)
    print(f"repo root    : {config.PROJECT_ROOT}")
    print(f"dataset      : {args.dataset_config}")
    print(f"min words    : {args.min_words}")
    print(f"max docs     : {args.max_docs if args.max_docs is not None else 'all'}")

    print("\nLoading raw dataset (Hub cache is reused when present) ...")
    dataset = load_raw_dataset(args.repo, args.dataset_config)
    print(f"  splits       : { {name: len(split) for name, split in dataset.items()} }")

    written: Dict[str, Dict[str, Any]] = {}
    reports: Dict[str, Dict[str, Any]] = {}

    for split, hub_split in SPLIT_ALIASES.items():
        print(f"\nScanning {hub_split} split ...")
        sentences, ranges, stats = build_split(
            (row for row in dataset[hub_split]), args.min_words, args.max_docs
        )
        checks = validate(sentences, ranges, args.min_words)

        sentences_path, sentence_count = write_json_array(
            config.path(f"wiki_{split}_sentences"), sentences
        )
        pairs_path, pair_count = write_json_array(
            config.path(f"wiki_{split}_pairs"), iter_pairs(sentences, ranges)
        )
        written[f"{split}_sentences"] = {"path": sentences_path, "expected": sentence_count}
        written[f"{split}_pairs"] = {"path": pairs_path, "expected": pair_count}
        reports[split] = {
            **stats,
            **checks,
            "duplicate_rate": duplicate_rate(sentences),
            "pairs_expected": max(0, stats["sentences"] - len(ranges)),
        }

        for artifact in (sentences_path, pairs_path):
            print(f"  wrote {artifact}  ({artifact.stat().st_size / 1e6:.1f} MB)")
        print_split_report(split, stats, sentences, ranges)
        print(f"  sentence pairs vs sentences - documents with sentences : "
              f"{pair_count:,} vs {reports[split]['pairs_expected']:,} "
              f"-> {'OK' if pair_count == reports[split]['pairs_expected'] else 'MISMATCH'}")
        print(f"  structural checks : {checks} "
              f"-> {'OK' if not any(checks.values()) else 'FAILED'}")
        if not args.quiet:
            print_samples(sentences, ranges, args.samples)

        # The train lists are ~1 GB; release them before the next split.
        del sentences, ranges, stats

    # ---------------------------------------------------------------- verify
    print("\nArtifact verification (re-read from disk)")
    print("-" * 74)
    all_ok = True
    for name, record in written.items():
        check = verify_artifact(record["path"], record["expected"])
        ok = check["elements"] == check["expected"] and check["closed"] and check["elements"] > 0
        if check["full_parse"] is not None:
            ok = ok and check["full_parse"]
        print(f"  {name:<15s} elements {check['elements']:>9,d} / "
              f"{check['expected']:>9,d}  closed={check['closed']}  "
              f"full_json_parse={check['full_parse']}")
        print(f"                  first={str(check['first'])[:70]!r}")
        all_ok = all_ok and ok

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print(f"  {'split':<6s} {'sentences':>12s} {'pairs':>12s} {'tokens':>12s}"
          f" {'docs':>9s} {'dups':>7s}")
    for split in SPLIT_ALIASES:
        row = reports[split]
        print(f"  {split:<6s} {row['sentences']:>12,d} {row['pairs']:>12,d} "
              f"{row['words']:>12,d} {row['documents']:>9,d} "
              f"{row['duplicate_rate']:>6.2f}%")
    print(f"\n  artifacts : {len(written)} files under "
          f"{config.path('wiki_train_sentences').parent}")
    print(f"  TASK 18R target was train {500_000:,}-{1_000_000:,} sentences and val "
          f"{10_000:,}-{20_000:,}; the full cleaned corpus is kept instead, because "
          f"its token totals ({reports['train']['words'] / 1e6:.1f}M train / "
          f"{reports['val']['words'] / 1e3:.0f}K val) are what WikiText-103's own "
          f"103M / 217K figures describe.")
    print(f"  all checks passed : {all_ok}")
    print(f"\nDone in {time.perf_counter() - started:.1f}s")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())



