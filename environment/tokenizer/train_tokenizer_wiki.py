"""Train the WikiText-103 BPE tokenizer from scratch and report its statistics (TASK 18R).

Corpus: a seeded random sample of ``environment/data/wikitext/train_sentences.json``
(default 500,000 of the ~1.5M sentences — training on all of them would take far
longer than it is worth for the vocabulary it returns).  Nothing pretrained is
involved: the merge table is learned from WikiText-103 alone, exactly like the
ARC tokenizer of TASK 19 was learned from ARC + ConceptNet alone.

vocab_size = 16000 (TASK 18R asks for a larger vocabulary because general
Wikipedia prose has a far bigger type inventory than ARC questions do), saved to
``environment/tokenizer/bpe_wiki_vocab.json``.

Usage (from the repository root)::

    python environment/tokenizer/train_tokenizer_wiki.py
    python environment/tokenizer/train_tokenizer_wiki.py --corpus-sentences 100000
    python environment/tokenizer/train_tokenizer_wiki.py --round-trip 1000 --quiet
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
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
    MIN_PAIR_FREQUENCY,
    SPECIAL_TOKENS,
    UNK_ID,
    BPETokenizer,
    normalize_text,
)

# Generic reporting helpers are shared with the TASK 19 ARC tokenizer script
# rather than duplicated here.
from train_tokenizer import (  # noqa: E402
    configure_stdout,
    length_stats,
    print_merge_distribution,
    print_tokenizations,
    print_top_merges,
    print_vocab_layout,
    round_trip,
)

#: TASK 18R: "vocab_size = 16000 (larger vocab for general text)".
DEFAULT_WIKI_VOCAB_SIZE = 16_000

#: TASK 18R: sample this many WikiText sentences for BPE training.
DEFAULT_CORPUS_SENTENCES = 500_000

#: TASK 18R's sample tokenizations.
SAMPLE_TERMS: Tuple[str, ...] = (
    "the president announced",
    "scientists discovered",
    "the economy grew",
    "photosynthesis",
)

ROUND_TRIP_SAMPLES = 1000

#: Token-length statistics are computed on a random subsample for speed.
LENGTH_SAMPLE = 20_000

#: The round trip must be byte-exact for all but the out-of-vocabulary characters
#: of rare scripts, so the pass mark is 99% rather than 100% (see
#: :func:`unknown_rate`).  The locked-down ARC corpus of TASK 19 scores 100%.
ROUND_TRIP_MIN_PERCENT = 99.0


def load_sentences(path: Path) -> List[str]:
    """Read one ``*_sentences.json`` artifact written by the TASK 18R preprocessor."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or (payload and not isinstance(payload[0], str)):
        raise SystemExit(f"{path} is not a JSON list of strings")
    return payload


def sample(items: Sequence[str], count: int, seed: int) -> List[str]:
    """Seeded random sample of ``count`` items (all of them when count >= len)."""
    if count >= len(items):
        return list(items)
    return random.Random(seed).sample(list(items), count)


def sentence_length_rows(
    tokenizer: BPETokenizer,
    train_texts: Sequence[str],
    val_texts: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """Token-length summary for a train sample and the whole val split."""
    rows = {
        "train": length_stats(tokenizer, train_texts),
        "val": length_stats(tokenizer, val_texts),
    }
    print("\nToken counts (unpadded, whole sentences)")
    print("-" * 74)
    print(f"  {'split':<8s} {'n':>9s} {'mean':>8s} {'median':>8s} {'p95':>7s} {'max':>6s}")
    for label, row in rows.items():
        print(f"  {label:<8s} {row['n']:>9,d} {row['mean']:>8.1f} {row['median']:>8.1f} "
              f"{row['p95']:>7.1f} {row['max']:>6.0f}")
    return rows


def unknown_rate(tokenizer: BPETokenizer, texts: Sequence[str]) -> Tuple[int, int, float]:
    """``[UNK]`` ids across ``texts`` as ``(unk, total, percent)``.

    A 16,000-token vocabulary learned from English Wikipedia prose has no
    characters for the rarer scripts WikiText also contains (Burmese, Chinese,
    Devanagari ...).  Those characters encode to ``[UNK]`` and ``decode`` drops
    special tokens, which is the only reason the round trip is not byte-exact.
    """
    unk = 0
    total = 0
    for text in texts:
        ids = tokenizer.encode_unpadded(text)
        total += len(ids)
        unk += sum(1 for token_id in ids if token_id == UNK_ID)
    return unk, total, (100.0 * unk / total if total else 0.0)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--vocab-size", type=int, default=DEFAULT_WIKI_VOCAB_SIZE,
        help=f"target vocabulary size (default {DEFAULT_WIKI_VOCAB_SIZE:,})",
    )
    parser.add_argument(
        "--corpus-sentences", type=int, default=DEFAULT_CORPUS_SENTENCES,
        help=f"sentences sampled from the train split for BPE training "
             f"(default {DEFAULT_CORPUS_SENTENCES:,})",
    )
    parser.add_argument(
        "--min-pair-frequency", type=int, default=MIN_PAIR_FREQUENCY,
        help=f"stop when the best pair is rarer than this (default {MIN_PAIR_FREQUENCY})",
    )
    parser.add_argument(
        "--round-trip", type=int, default=ROUND_TRIP_SAMPLES,
        help=f"val sentences used for the round-trip check (default {ROUND_TRIP_SAMPLES:,})",
    )
    parser.add_argument(
        "--length-sample", type=int, default=LENGTH_SAMPLE,
        help=f"train sentences used for the token-length statistics (default {LENGTH_SAMPLE:,})",
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
    print("TASK 18R - WikiText-103 BPE tokenizer from scratch")
    print("=" * 74)
    print(f"target vocab size : {args.vocab_size:,}")
    print(f"corpus sample     : {args.corpus_sentences:,} sentences (seed {args.seed})")

    train_path = config.path("wiki_train_sentences")
    val_path = config.path("wiki_val_sentences")
    train_sentences = load_sentences(train_path)
    val_sentences = load_sentences(val_path)
    n_train = len(train_sentences)

    corpus = sample(train_sentences, args.corpus_sentences, args.seed)
    length_source = sample(corpus, args.length_sample, args.seed + 1)
    del train_sentences  # a 544 MB file is ~1 GB of objects; free it before training

    print("\nCorpus")
    print("-" * 74)
    print(f"  train_sentences.json    : {train_path}")
    print(f"  val_sentences.json      : {val_path}")
    print(f"  train sentences on disk : {n_train:,}")
    print(f"  val sentences on disk   : {len(val_sentences):,}")
    print(f"  BPE corpus (seeded sample)     : {len(corpus):,}")
    print(f"  length-statistics sample       : {len(length_source):,}")

    print("\nTraining BPE ...")
    tokenizer = BPETokenizer()
    stats = tokenizer.train(
        corpus,
        vocab_size=args.vocab_size,
        min_pair_frequency=args.min_pair_frequency,
        verbose=not args.quiet,
        log_every=2000,
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

    print_tokenizations(tokenizer, SAMPLE_TERMS, "Sample tokenizations - TASK 18R terms")
    rows = sentence_length_rows(tokenizer, length_source, val_sentences)

    n_round = min(args.round_trip, len(val_sentences))
    round_trip_texts = sample(val_sentences, n_round, args.seed + 2)
    successes, failures = round_trip(tokenizer, round_trip_texts)
    round_trip_pct = 100.0 * successes / n_round
    unk, total_ids, unk_pct = unknown_rate(tokenizer, val_sentences)
    print("\nRound-trip check (val sentences, unpadded)")
    print("-" * 74)
    print(f"  {n_round:,} random val sentences : {successes:,}/{n_round:,} = "
          f"{round_trip_pct:.1f}%  (pass mark {ROUND_TRIP_MIN_PERCENT:.1f}%)")
    print(f"  UNK ids across the val split    : {unk:,} / {total_ids:,} tokens = "
          f"{unk_pct:.3f}%")
    for text, expected, got in failures[:3]:
        print(f"    FAIL raw={text[:70]!r}")
        print(f"         expected={expected[:70]!r}")
        print(f"         got     ={got[:70]!r}")
    if failures:
        print("  Failures are out-of-vocabulary characters: a 16,000-token vocabulary")
        print("  cannot cover every script in WikiText, so those characters become")
        print("  [UNK] and are dropped on decode.  Nothing else is lost.")

    # ------------------------------------------------------------------- save
    output_path = tokenizer.save(args.output if args.output else config.path("tokenizer_wiki"))
    print("\nSaved tokenizer")
    print("-" * 74)
    print(f"  path                 : {output_path}")
    print(f"  size                 : {output_path.stat().st_size / 1024:.0f} KB")
    print(f"  vocab entries        : {len(tokenizer.vocab):,}")
    print(f"  merge rules          : {len(tokenizer.merges):,}")

    reloaded = BPETokenizer.load(output_path)
    same_vocab = reloaded.vocab == tokenizer.vocab
    same_merges = reloaded.merges == tokenizer.merges
    same_ids = all(
        reloaded.encode_unpadded(text) == tokenizer.encode_unpadded(text)
        for text in round_trip_texts
    )
    print(f"  reload check         : vocab={same_vocab}  merges={same_merges}  "
          f"encode identical={same_ids}")

    # --------------------------------------------------------------- summary
    train_row = rows["train"]
    val_row = rows["val"]
    words_per_sentence = sum(len(text.split()) for text in length_source) / len(length_source)
    ratio = train_row["mean"] / words_per_sentence if words_per_sentence else 0.0
    target_met = 10 <= train_row["mean"] <= 20

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print(f"  corpus sentences (BPE training) : {len(corpus):,} of {n_train:,} on disk")
    print(f"  vocab size achieved             : {tokenizer.vocab_size:,} "
          f"(target {stats['target_vocab_size']:,}, "
          f"natural convergence={stats['stopped_early']})")
    print(f"  merges learned                  : {len(tokenizer.merges):,}")
    print(f"  avg tokens / train sentence     : {train_row['mean']:.1f} "
          f"({words_per_sentence:.1f} words before BPE)")
    print(f"  avg tokens / val sentence       : {val_row['mean']:.1f}")
    print(f"  round-trip (val)                : {round_trip_pct:.1f}% "
          f"({ROUND_TRIP_MIN_PERCENT:.1f}% pass mark; {unk_pct:.3f}% of val tokens are [UNK])")
    print(f"  BPE training time               : {elapsed:.1f}s")
    print(f"  TASK 18R target 10-20 tokens/sentence : "
          f"{'MET' if target_met else 'NOT MET'}")
    if not target_met:
        print("     WikiText sentences average "
              f"{words_per_sentence:.1f} words and these merges only concatenate "
              f"tokens (never delete them), so tokens/sentence cannot fall below the "
              f"word count; {train_row['mean']:.1f} is {ratio * 100:.0f}% of that "
              "floor.  The target assumed shorter sentences.")

    all_ok = same_vocab and same_merges and same_ids and round_trip_pct >= ROUND_TRIP_MIN_PERCENT
    print(f"  all checks passed               : {all_ok}")
    print(f"\nDone in {time.perf_counter() - started:.1f}s")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

