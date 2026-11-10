"""Train the V3 BPE tokenizer from scratch and report its statistics (TASK 19).

Corpus (all of it, nothing else — no pretrained vocabulary is involved):

    ARC-Challenge questions  (train + val + test)   2,577 strings
    ARC-Challenge choices    (train + val + test)  10,308 strings
    ConceptNet triples as sentences                99,187 strings
                                                   --------
                                                  112,072 strings

ConceptNet triples become ``"subject relation object"`` sentences, e.g.
``"fire causes heat"`` / ``"dog is a animal"``.

Usage (from the repository root)::

    python environment/tokenizer/train_tokenizer.py
    python environment/tokenizer/train_tokenizer.py --vocab-size 4000
    python environment/tokenizer/train_tokenizer.py --round-trip 300 --quiet
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ``environment/`` and ``environment/tokenizer/`` are not package roots, so make
# ``import config`` / ``from bpe import ...`` work from any cwd (TASK 18 rule).
ENVIRONMENT_DIR = Path(__file__).resolve().parents[1]
TOKENIZER_DIR = Path(__file__).resolve().parent
for _directory in (ENVIRONMENT_DIR, TOKENIZER_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import config  # noqa: E402
from bpe import (  # noqa: E402
    DEFAULT_MAX_LENGTH,
    DEFAULT_VOCAB_SIZE,
    MIN_PAIR_FREQUENCY,
    SPECIAL_TOKENS,
    BPETokenizer,
    display_token,
    normalize_text,
)

ARC_SPLITS: Tuple[str, ...] = ("train", "val", "test")

#: Science terms required by TASK 19's report.
SAMPLE_TERMS: Tuple[str, ...] = (
    "photosynthesis",
    "evaporation",
    "gravitational",
    "ecosystem",
    "electricity",
)

#: Extra samples named in TASKS_V3.md (a lone word and a punctuation-heavy phrase).
EXTRA_SAMPLE_TERMS: Tuple[str, ...] = ("causes", "The trophy didn't fit")

ROUND_TRIP_SAMPLES = 100


def configure_stdout() -> None:
    """Force UTF-8 output so ``Ġ`` prints even on a cp1252 Windows console."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover - non-TextIO streams
            pass


def show(token: str) -> str:
    """Render a token for display, making the word-boundary symbol visible."""
    return display_token(token)


def load_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_corpus() -> Tuple[List[str], List[str], List[str], Dict[str, int]]:
    """Return ``(all_texts, questions, choices, per_source_counts)``."""
    texts: List[str] = []
    questions: List[str] = []
    choices: List[str] = []
    counts: Dict[str, int] = {}

    for split in ARC_SPLITS:
        records = load_records(config.path(f"arc_{split}"))
        split_questions = [str(record["question"]) for record in records]
        split_choices = [str(choice) for record in records for choice in record["choices"]]
        questions.extend(split_questions)
        choices.extend(split_choices)
        texts.extend(split_questions)
        texts.extend(split_choices)
        counts[f"arc_{split}_questions"] = len(split_questions)
        counts[f"arc_{split}_choices"] = len(split_choices)

    triples = load_records(config.path("conceptnet"))
    sentences = [
        f"{record['subject']} {record['relation']} {record['object']}" for record in triples
    ]
    texts.extend(sentences)
    counts["conceptnet_sentences"] = len(sentences)

    counts["total_strings"] = len(texts)
    return texts, questions, choices, counts


def print_corpus(counts: Dict[str, int]) -> None:
    print("Corpus")
    print("-" * 74)
    for split in ARC_SPLITS:
        print(
            f"  ARC {split:<5s} questions {counts[f'arc_{split}_questions']:>6,d}"
            f"   choices {counts[f'arc_{split}_choices']:>6,d}"
        )
    print(f"  ConceptNet sentences            {counts['conceptnet_sentences']:>6,d}")
    print(f"  ---------------------------------------------")
    print(f"  TOTAL strings                   {counts['total_strings']:>6,d}")


def print_vocab_layout(tokenizer: BPETokenizer, characters: int) -> None:
    print("\nVocabulary layout")
    print("-" * 74)
    print(f"  ids 0-4                 specials : "
          f"{', '.join(f'{i}={t}' for i, t in enumerate(SPECIAL_TOKENS))}")
    print(f"  ids 5-{4 + characters:<5d}            {characters} single characters")
    print(f"  ids {5 + characters}-{tokenizer.vocab_size - 1:<6d}          "
          f"{len(tokenizer.merges)} merged tokens (merge order)")
    print("  first 12 ids            " + "  ".join(
        f"{index}:{show(token)!r}" for index, token in tokenizer.vocabulary_rows(12)
    ))


def print_top_merges(tokenizer: BPETokenizer, count: int = 10) -> None:
    print(f"\n{count} most frequent merge rules")
    print("-" * 74)
    print(f"  {'rank':>5s}  {'merged token':<24s} {'rule':<20s} {'count':>9s}")
    for row in tokenizer.top_merges(count):
        pair = row["pair"]
        rule = f"{show(pair[0])!r} + {show(pair[1])!r}"
        print(f"  {row['rank']:>5d}  {show(row['token'])!r:<24s} {rule:<20s} {row['count']:>9,d}")


def print_merge_distribution(tokenizer: BPETokenizer) -> None:
    """How meaningful the tail of the merge table is, for the natural-convergence rule."""
    counts = tokenizer.merge_counts
    if not counts:
        print("\n(merge counts unavailable — reporting only exists right after train())")
        return
    ordered = sorted(counts)
    total = len(ordered)

    def below(threshold: int) -> int:
        return sum(1 for count in ordered if count < threshold)

    print("\nMerge-count distribution")
    print("-" * 74)
    print(f"  highest merge count      : {ordered[-1]:,}  (rank 0)")
    print(f"  median merge count       : {ordered[total // 2]:,}")
    print(f"  lowest merge count       : {ordered[0]:,}  (final merge, rank {total - 1})")
    print(f"  merges with count < 10   : {below(10):,} / {total:,}"
          f"  ({100.0 * below(10) / total:.1f}%)")
    print(f"  merges with count < 5    : {below(5):,} / {total:,}"
          f"  ({100.0 * below(5) / total:.1f}%)")
    print(f"  merges with count < 2    : {below(2):,} / {total:,}  "
          f"(min_pair_frequency={MIN_PAIR_FREQUENCY})")


def print_tokenizations(tokenizer: BPETokenizer, terms: Sequence[str], title: str) -> None:
    print(f"\n{title}")
    print("-" * 74)
    for text in terms:
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.encode_unpadded(text)
        rendered = " ".join(show(token) for token in tokens)
        print(f"  {text!r}")
        print(f"      normalized  : {normalize_text(text)!r}")
        print(f"      {len(tokens):>3d} tokens    : {rendered}")
        print(f"      ids         : {ids}")


def length_stats(tokenizer: BPETokenizer, texts: Sequence[str]) -> Dict[str, float]:
    """Token-length summary for a list of strings."""
    lengths = sorted(len(tokenizer.encode_unpadded(text)) for text in texts)
    total = len(lengths)
    return {
        "n": total,
        "mean": sum(lengths) / total if total else 0.0,
        "median": lengths[total // 2] if total else 0.0,
        "p95": lengths[min(total - 1, int(0.95 * total))] if total else 0.0,
        "max": lengths[-1] if total else 0.0,
    }


def print_length_stats(
    tokenizer: BPETokenizer,
    questions: Sequence[str],
    choices: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    stats = {
        "question": length_stats(tokenizer, questions),
        "choice": length_stats(tokenizer, choices),
    }
    print("\nToken counts (whole splits, unpadded)")
    print("-" * 74)
    print(f"  {'split':<10s} {'n':>7s} {'mean':>8s} {'median':>8s} {'p95':>7s} {'max':>6s}")
    for label, row in stats.items():
        print(
            f"  {label:<10s} {row['n']:>7d} {row['mean']:>8.1f} "
            f"{row['median']:>8.1f} {row['p95']:>7.1f} {row['max']:>6.1f}"
        )
    return stats


def round_trip(tokenizer: BPETokenizer, texts: Sequence[str]) -> Tuple[int, List[Tuple[str, str, str]]]:
    """Compare ``decode(encode_unpadded(text))`` with ``normalize_text(text)``."""
    successes = 0
    failures: List[Tuple[str, str, str]] = []
    for text in texts:
        expected = normalize_text(text)
        got = tokenizer.decode(tokenizer.encode_unpadded(text))
        if got == expected:
            successes += 1
        else:
            failures.append((text, expected, got))
    return successes, failures


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE,
        help=f"target vocabulary size (default {DEFAULT_VOCAB_SIZE})",
    )
    parser.add_argument(
        "--min-pair-frequency", type=int, default=MIN_PAIR_FREQUENCY,
        help=f"stop when the best pair is rarer than this (default {MIN_PAIR_FREQUENCY})",
    )
    parser.add_argument(
        "--round-trip", type=int, default=ROUND_TRIP_SAMPLES,
        help=f"val examples used for the round-trip check (default {ROUND_TRIP_SAMPLES})",
    )
    parser.add_argument("--seed", type=int, default=int(config.TRAIN["seed"]))
    parser.add_argument("--quiet", action="store_true", help="suppress per-merge progress")
    parser.add_argument("--output", type=Path, default=None, help="override the save path")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_stdout()
    args = parse_args(argv)
    started = time.perf_counter()

    print("=" * 74)
    print("TASK 19 — BPE tokenizer from scratch")
    print("=" * 74)

    texts, questions, choices, counts = collect_corpus()
    print_corpus(counts)

    print("\nNormalisation (applied before every encode)")
    print("-" * 74)
    print(f"  raw        : {questions[0][:90]!r}")
    print(f"  normalized : {normalize_text(questions[0])[:90]!r}")

    print("\nTraining BPE ...")
    tokenizer = BPETokenizer()
    stats = tokenizer.train(
        texts,
        vocab_size=args.vocab_size,
        min_pair_frequency=args.min_pair_frequency,
        verbose=not args.quiet,
        log_every=1000,
    )
    elapsed = time.perf_counter() - started

    print("\nTraining result")
    print("-" * 74)
    print(f"  target vocab size    : {stats['target_vocab_size']:,}")
    print(f"  VOCAB SIZE ACHIEVED  : {stats['vocab_size']:,}")
    print(f"  merges learned       : {stats['merges']:,} "
          f"(merge ids start at {stats['merges_start_id']})")
    print(f"  stopped early        : {stats['stopped_early']} "
          f"(min pair frequency {stats['min_pair_frequency']})")
    print(f"  word types           : {stats['word_types']:,}")
    print(f"  word instances       : {stats['word_instances']:,}")
    print(f"  training time        : {elapsed:.1f}s")

    print_vocab_layout(tokenizer, stats["characters"])
    print_top_merges(tokenizer, 10)
    print_merge_distribution(tokenizer)
    print_tokenizations(tokenizer, SAMPLE_TERMS, "Sample tokenizations — science terms")
    print_tokenizations(tokenizer, EXTRA_SAMPLE_TERMS, "Sample tokenizations — TASKS_V3.md extras")
    lengths = print_length_stats(tokenizer, questions, choices)

    # ----------------------------------------------------------- round trip
    val_records = load_records(config.path("arc_val"))
    val_questions = [str(record["question"]) for record in val_records]
    val_choices = [str(choice) for record in val_records for choice in record["choices"]]

    rng = random.Random(args.seed)
    n_questions = min(args.round_trip, len(val_questions))
    n_choices = min(args.round_trip, len(val_choices))
    picked_questions = rng.sample(val_questions, n_questions)
    picked_choices = rng.sample(val_choices, n_choices)

    question_ok, question_bad = round_trip(tokenizer, picked_questions)
    choice_ok, choice_bad = round_trip(tokenizer, picked_choices)

    print("\nRound-trip on ARC val — decode(encode_unpadded(x)) == normalize_text(x)")
    print("-" * 74)
    print(f"  {n_questions} random val questions : {question_ok}/{n_questions} = "
          f"{100.0 * question_ok / n_questions:.1f}%")
    print(f"  {n_choices} random val choices   : {choice_ok}/{n_choices} = "
          f"{100.0 * choice_ok / n_choices:.1f}%")
    for text, expected, got in (question_bad + choice_bad)[:3]:
        print(f"    FAIL raw={text[:60]!r}")
        print(f"         expected={expected[:60]!r}")
        print(f"         got     ={got[:60]!r}")

    padded_ok = sum(
        1 for text in picked_questions
        if tokenizer.decode(tokenizer.encode(text)) == normalize_text(text)
    )
    truncated = sum(
        1 for text in picked_questions
        if len(tokenizer.encode_unpadded(text)) > DEFAULT_MAX_LENGTH
    )
    question_choice = [
        f"{record['question']} {record['choices'][record['label']]}" for record in val_records
    ]
    over_budget = sum(
        1 for text in question_choice
        if len(tokenizer.encode_unpadded(text)) > DEFAULT_MAX_LENGTH
    )
    print(f"\n  padded to max_seq_len={DEFAULT_MAX_LENGTH} : {padded_ok}/{n_questions} = "
          f"{100.0 * padded_ok / n_questions:.1f}%  (truncation is lossy by design)")
    print(f"  questions longer than {DEFAULT_MAX_LENGTH} tokens : "
          f"{truncated}/{n_questions}")
    print(f"  question + correct choice longer than {DEFAULT_MAX_LENGTH} tokens : "
          f"{over_budget}/{len(question_choice)} val records "
          f"({100.0 * over_budget / len(question_choice):.1f}%)")

    # ------------------------------------------------------------------ save
    output_path = tokenizer.save(args.output)
    print("\nSaved tokenizer")
    print("-" * 74)
    print(f"  path                 : {output_path}")
    print(f"  size                 : {output_path.stat().st_size / 1024:.0f} KB")
    print(f"  top-level JSON keys  : {sorted(tokenizer.to_dict())}")
    print(f"  vocab entries        : {len(tokenizer.vocab):,}")
    print(f"  merge rules          : {len(tokenizer.merges):,}")

    reloaded = BPETokenizer.load(args.output)
    same_vocab = reloaded.vocab == tokenizer.vocab
    same_merges = reloaded.merges == tokenizer.merges
    same_ids = all(
        reloaded.encode_unpadded(text) == tokenizer.encode_unpadded(text)
        for text in picked_questions + picked_choices
    )
    print(f"  reload check         : vocab={same_vocab}  merges={same_merges}  "
          f"encode identical={same_ids}")

    # --------------------------------------------------------------- summary
    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print(f"  corpus strings        : {counts['total_strings']:,}")
    print(f"  vocab size achieved   : {tokenizer.vocab_size:,} "
          f"(target {stats['target_vocab_size']:,}, natural convergence={stats['stopped_early']})")
    print(f"  merges learned        : {len(tokenizer.merges):,}")
    print(f"  avg tokens / question : {lengths['question']['mean']:.1f}")
    print(f"  avg tokens / choice   : {lengths['choice']['mean']:.1f}")
    print(f"  round-trip (val q)    : {100.0 * question_ok / n_questions:.1f}%")

    all_ok = (
        same_vocab and same_merges and same_ids
        and question_ok == n_questions and choice_ok == n_choices
    )
    print(f"  all checks passed     : {all_ok}")
    print(f"\nDone in {time.perf_counter() - started:.1f}s")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())



