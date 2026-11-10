"""STEP 1 of the directional encoder: structural relation labels for WikiText pairs.

Why this exists
---------------
Four experiments now point the same way: the geometry that decides a prediction is
the CONTINUATION DIRECTION, not surface similarity. TASK 27/29 showed memory helps
only when the retrieved neighbours are topically relevant (+0.0039 in-corpus vs
-0.0049 out-of-corpus) and TASK 30's gate could not fix that (best threshold still
-0.0003 vs no memory). So the encoder has to be *trained* so that the direction
``P_actual - P_current`` encodes HOW the text continues - elaboration, a new topic, or
a conclusion - instead of how similar the two sentences happen to look.

Labels (and how the task's three rules are made disjoint)
---------------------------------------------------------
The task's rules overlap when applied literally: a paragraph's last sentence is also
"same paragraph, not first sentence", and the first sentence of a new paragraph is
only ever the *second* element of a pair. They are therefore applied as the disjoint
priority scheme they describe:

    new_topic    the pair crosses a paragraph boundary: ``S_n`` is the last kept
                 sentence of its paragraph and ``S_n+1`` is the first kept sentence of
                 the NEXT paragraph. A section heading between two paragraphs breaks
                 the comparison (``preprocess.iter_documents`` treats a heading as an
                 article boundary), so "new_topic" never means "different article".
    conclusion   inside one paragraph and ``S_n+1`` is that paragraph's LAST kept
                 sentence - the continuation terminates the paragraph.
    elaboration  everything else inside one paragraph (``S_n+1`` continues it).

Per paragraph of ``L`` sentences this yields ``L-2`` elaboration pairs and one
conclusion pair, plus one new_topic pair per paragraph boundary. WikiText paragraphs
are short (measured on the scanned prefix: 240 kept sentences over 55 sentence-bearing
rows = 4.4 sentences per paragraph, giving 143 elaboration / 40 new_topic /
42 conclusion candidates, i.e. ~63/18/19%), so the natural mix leans to elaboration but
is not extreme. The quotas still matter for STEP 2: the triplet loss samples a
same-relation positive and a DIFFERENT-relation negative from each batch of 32, and a
batch that happens to contain only one relation has no usable negative for those
anchors. The extractor therefore fills per-class QUOTAS (``--per-class``, default
16,667 -> ~50,000 pairs) and prints both the natural counts of the scanned prefix and
the emitted counts.

Source
------
``Salesforce/wikitext`` ``wikitext-103-raw-v1`` (already in the local Hub cache), read
through ``preprocess.py``'s helpers so the sentence splitting, whitespace cleaning and
``>= 5``-word filter are identical to the corpus the rest of the project uses.
WikiText rows ARE paragraphs, separated by blank rows, which is what makes the
paragraph structure above recoverable - ``train_sentences.json`` flattened it away.

Artifact
--------
    environment/data/wikitext/directional_pairs.json    {"s1", "s2", "relation"} per line

Usage (from the repository root)::

    python environment/data/wikitext/label_directional_pairs.py
    python environment/data/wikitext/label_directional_pairs.py --per-class 100 --max-rows 20000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

# ``environment/`` is not a package root, so make ``import config`` work from anywhere
# (TASK 18 convention; the data scripts live one level deeper).
ENVIRONMENT_DIR = Path(__file__).resolve().parents[2]
if str(ENVIRONMENT_DIR) not in sys.path:
    sys.path.insert(0, str(ENVIRONMENT_DIR))

import config  # noqa: E402
from data.wikitext.preprocess import (  # noqa: E402
    HEADING_RE,
    MIN_WORDS,
    load_raw_dataset,
    split_sentences,
    word_count,
    write_json_array,
)

#: The three structural relations, in the order they are reported.
RELATIONS: Tuple[str, str, str] = ("elaboration", "new_topic", "conclusion")
DEFAULT_PER_CLASS = 16667  # ~50,000 pairs in total
OUTPUT_PATH = config.path("wiki_directional_pairs")
SAMPLE_CHARS = 100


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label WikiText sentence pairs with paragraph-structure relations."
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--per-class", type=int, default=DEFAULT_PER_CLASS)
    parser.add_argument("--min-words", type=int, default=MIN_WORDS)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="stop after this many raw rows (0 = until the quotas are full)",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def paragraph_sentences(text: str, min_words: int) -> List[str]:
    """Cleaned, length-filtered sentences of one raw WikiText row (one paragraph)."""
    return [sentence for sentence in split_sentences(text) if word_count(sentence) >= min_words]


def iter_labeled_pairs(
    rows: Sequence[str],
    min_words: int = MIN_WORDS,
    per_class: Optional[int] = DEFAULT_PER_CLASS,
    max_rows: int = 0,
    census: Optional[Counter] = None,
    paragraphs: Optional[Counter] = None,
) -> Iterator[Dict[str, str]]:
    """Yield ``{"s1", "s2", "relation"}`` dicts, filling per-class quotas.

    ``census`` accumulates the NATURAL counts of every candidate pair seen (before the
    quota filter) so the caller can report how skewed the corpus really is; ``paragraphs``
    counts rows scanned, sentence-bearing rows, and article boundaries (headings).
    Quotas are filled in the order the corpus is read, so the result is deterministic.
    """
    counts: Counter = Counter()
    previous_last_sentence: Optional[str] = None
    for index, raw in enumerate(rows):
        if max_rows and index >= max_rows:
            break
        if per_class is not None and all(counts[rel] >= per_class for rel in RELATIONS):
            break
        text = str(raw).strip()
        if not text:
            continue
        if HEADING_RE.match(text):  # a heading starts a new article: no cross pair
            previous_last_sentence = None
            if paragraphs is not None:
                paragraphs["headings"] += 1
            continue
        sentences = paragraph_sentences(text, min_words)
        if not sentences:
            previous_last_sentence = None
            continue
        if paragraphs is not None:
            paragraphs["rows"] += 1
            paragraphs["kept_rows"] += 1
            paragraphs["sentences"] += len(sentences)

        # within-paragraph pairs: the last one closes the paragraph
        for position in range(len(sentences) - 1):
            relation = "conclusion" if position == len(sentences) - 2 else "elaboration"
            if census is not None:
                census[relation] += 1
            if per_class is not None and counts[relation] >= per_class:
                continue
            counts[relation] += 1
            yield {"s1": sentences[position], "s2": sentences[position + 1], "relation": relation}

        # across the paragraph boundary: the first sentence of the new paragraph
        if previous_last_sentence is not None:
            if census is not None:
                census["new_topic"] += 1
            if per_class is None or counts["new_topic"] < per_class:
                counts["new_topic"] += 1
                yield {
                    "s1": previous_last_sentence,
                    "s2": sentences[0],
                    "relation": "new_topic",
                }
        previous_last_sentence = sentences[-1]



def clip(text: str, limit: int = SAMPLE_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def verify_output(path: Path, expected: int) -> Dict[str, Any]:
    """Re-read the artifact: element count + the first element parses as a labeled pair."""
    elements = 0
    first: Optional[Dict[str, Any]] = None
    closed = False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text.startswith("["):
                text = text[1:]
            if text.endswith(","):
                text = text[:-1]
            text = text.strip()
            if not text:
                continue
            if text.endswith("]"):
                try:
                    item = json.loads(text)
                except json.JSONDecodeError:
                    item = json.loads(text[:-1])
                elements += 1
                closed = True
            else:
                item = json.loads(text)
                elements += 1
            if first is None:
                first = item
    ok = (
        elements == expected
        and closed
        and isinstance(first, dict)
        and set(first) == {"s1", "s2", "relation"}
        and first["relation"] in RELATIONS
    )
    return {"elements": elements, "expected": expected, "closed": closed, "first": first, "ok": ok}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()

    print("=" * 90)
    print("STEP 1 - structural relation labels over WikiText paragraph boundaries")
    print("=" * 90)
    print(
        f"  source   : wikitext-103-raw-v1 (train split) via preprocess.load_raw_dataset\n"
        f"  relations: {', '.join(RELATIONS)}  (per-class quota {args.per_class})\n"
        f"  output   : {args.output}"
    )

    dataset = load_raw_dataset()
    rows = dataset["train"]["text"]
    print(f"\n  loaded {len(rows):,} raw rows (each row is one paragraph)")

    census: Counter = Counter()
    paragraphs: Counter = Counter()
    pairs = list(
        iter_labeled_pairs(
            rows,
            min_words=args.min_words,
            per_class=args.per_class,
            max_rows=args.max_rows,
            census=census,
            paragraphs=paragraphs,
        )
    )
    counts = Counter(pair["relation"] for pair in pairs)
    print(
        f"  scanned  : {paragraphs['rows']:,} sentence-bearing rows, "
        f"{paragraphs['headings']:,} headings (article boundaries), "
        f"{paragraphs['sentences']:,} kept sentences"
    )
    print(
        f"  natural  : {census['elaboration']:,} elaboration / {census['new_topic']:,} "
        f"new_topic / {census['conclusion']:,} conclusion candidate pairs "
        f"(before quotas - this is how skewed the corpus is)"
    )
    print("\n  emitted pairs per relation")
    for relation in RELATIONS:
        share = 100 * counts[relation] / max(1, len(pairs))
        print(f"    {relation:<12} {counts[relation]:>8,}  ({share:5.1f}%)")
    print(f"    {'total':<12} {len(pairs):>8,}")

    path, written = write_json_array(args.output, pairs)
    print(f"\n  wrote {path}  ({path.stat().st_size / 1e6:.1f} MB, {written:,} elements)")
    check = verify_output(path, written)
    print(
        f"  verify   : elements {check['elements']:,}/{check['expected']:,} "
        f"closed={check['closed']} first={check['first']} -> "
        f"{'OK' if check['ok'] else 'FAILED'}"
    )

    if not args.quiet:
        print("\n  samples")
        for relation in RELATIONS:
            example = next(pair for pair in pairs if pair["relation"] == relation)
            print(f"    [{relation}]")
            print(f"      s1: {clip(example['s1'])}")
            print(f"      s2: {clip(example['s2'])}")

    print(f"\nDone in {time.perf_counter() - started:.1f}s")
    return 0 if check["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

