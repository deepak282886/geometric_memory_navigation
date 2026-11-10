"""Byte-pair-encoding tokenizer built from scratch for the V3 experiment (TASK 19).

No pretrained tokenizer, no borrowed vocabulary: the merge table is learned from
the ARC-Challenge + ConceptNet corpus prepared in TASK 18 and nothing else.

Preprocessing (identical at train, encode and round-trip time — note 5)
----------------------------------------------------------------------
1. lowercase everything
2. put a space around every punctuation character, so punctuation becomes its
   own token — nothing is deleted
3. collapse repeated whitespace and strip the ends

``"The trophy didn't fit."`` -> ``"the trophy didn ' t fit ."``

Word boundaries and vocabulary layout
-------------------------------------
A merge must never fuse the end of one word with the start of the next, and the
decoder must be able to put the spaces back.  Both are solved the way GPT-2
solves them: every word is prefixed with the reserved symbol ``Ġ``
(``WORD_PREFIX``, U+0120).  ``"the cat"`` starts as
``("Ġ","t","h","e") ("Ġ","c","a","t")`` and merges only happen *inside* a word.
Decoding is then ``"".join(tokens).replace(WORD_PREFIX, " ").strip()``.

Vocabulary layout::

    id 0..4       [PAD] [UNK] [BOS] [EOS] [MASK]   <- always first, reserved
    id 5..5+C-1   the C single characters seen in the corpus (incl. Ġ)
    id ...        merged tokens, in merge order

So merges start at ``5 + C`` and can never occupy the reserved ids 0-4.

Public API
----------
``BPETokenizer.train(texts, vocab_size)`` learns ``vocab`` + ``merges``;
``encode`` / ``encode_unpadded`` / ``tokenize`` / ``decode`` use them;
``save`` / ``load`` persist exactly ``{"vocab": {...}, "merges": [[a, b], ...]}``.
"""

from __future__ import annotations

import heapq
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# ``environment/`` is not a package root, so make ``import config`` work no
# matter which directory this module was imported from (TASK 18 convention).
ENVIRONMENT_DIR = Path(__file__).resolve().parents[1]
if str(ENVIRONMENT_DIR) not in sys.path:
    sys.path.insert(0, str(ENVIRONMENT_DIR))

import config  # noqa: E402

# ---------------------------------------------------------------------------
# Special tokens — MUST be the first five ids, in this order.
# ---------------------------------------------------------------------------
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
MASK_TOKEN = "[MASK]"

SPECIAL_TOKENS: Tuple[str, ...] = (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN)

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
MASK_ID = 4

#: Word-boundary prefix (GPT-2's ``Ġ``).  Part of the initial character
#: vocabulary; normalisation never produces it.
WORD_PREFIX = "\u0120"

#: Merge counting stops once the best remaining pair is rarer than this.
MIN_PAIR_FREQUENCY = 2

DEFAULT_VOCAB_SIZE = int(config.MODEL["vocab_size"])
DEFAULT_MAX_LENGTH = int(config.MODEL["max_seq_len"])

#: ``"a.b"`` -> ``"a . b"``: anything that is not a word char or whitespace is
#: punctuation and gets isolated.  ``\w`` is Unicode-aware, so ``á``/``ñ`` stay
#: inside their word.
PUNCTUATION_RE = re.compile(r"([^\w\s])")
WHITESPACE_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Normalisation helpers (module level: train, encode and tests all share them)
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Lowercase, isolate punctuation with spaces, collapse whitespace, strip."""
    text = str(text).lower()
    text = PUNCTUATION_RE.sub(r" \1 ", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def split_words(normalized_text: str) -> List[str]:
    """Split an already-normalised string into word units (punctuation included)."""
    return normalized_text.split(" ") if normalized_text else []


def word_symbols(word: str) -> List[str]:
    """``"cat"`` -> ``["Ġ", "c", "a", "t"]`` — the initial BPE symbol sequence."""
    return [WORD_PREFIX, *word]


def pair_counter(symbols: Sequence[str]) -> "Counter":
    """Weighted-count every adjacent symbol pair in ``symbols`` (multiplicity kept)."""
    counter: Counter = Counter()
    for index in range(len(symbols) - 1):
        counter[(symbols[index], symbols[index + 1])] += 1
    return counter


def merge_pair(symbols: Sequence[str], pair: Tuple[str, str], merged: str) -> List[str]:
    """Replace every non-overlapping ``pair`` occurrence with ``merged``.

    Scanning left to right means ``("a","a")`` in ``"a a a"`` consumes indices
    0-1 and leaves index 2, matching the counting rule used during training.
    """
    first, second = pair
    out: List[str] = []
    index = 0
    length = len(symbols)
    while index < length:
        if index + 1 < length and symbols[index] == first and symbols[index + 1] == second:
            out.append(merged)
            index += 2
        else:
            out.append(symbols[index])
            index += 1
    return out


def display_token(token: str) -> str:
    """Render ``token`` with the word-boundary symbol as ``|``.

    ``Ġ`` is invisible/mojibake-prone in many consoles and log pipelines, so
    anything meant for humans goes through here when it might contain it.
    """
    return token.replace(WORD_PREFIX, "|")


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
class BPETokenizer:
    """Character-initialised BPE with a learned merge table.

    Attributes
    ----------
    vocab : dict[str, int]
        token -> id.  Ids 0-4 are the special tokens, then the single characters,
        then merged tokens in merge order.
    merges : list[list[str, str]]
        Ordered merge rules; ``merges[0]`` was learned first.
    merge_counts : list[int]
        Frequency each merge had when it was learned.  For reporting only — it is
        **not** part of the saved format, so it is empty after :meth:`load`.
    """

    def __init__(self) -> None:
        self.vocab: Dict[str, int] = {}
        self.merges: List[List[str]] = []
        self.merge_counts: List[int] = []
        self._id_to_token: Dict[int, str] = {}
        self._merge_ranks: Dict[Tuple[str, str], int] = {}

    # ---------------------------------------------------------------- dunder
    def __len__(self) -> int:
        return len(self.vocab)

    def __repr__(self) -> str:
        return (
            f"BPETokenizer(vocab_size={len(self.vocab)}, "
            f"merges={len(self.merges)}, trained={bool(self.merges)})"
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def is_trained(self) -> bool:
        return bool(self.vocab)

    # ------------------------------------------------------------------- ids
    def token_id(self, token: str) -> int:
        """Id of ``token``, falling back to ``[UNK]`` when absent."""
        return self.vocab.get(token, UNK_ID)

    def id_token(self, token_id: int) -> str:
        """Token for ``token_id`` (``[UNK]`` for out-of-range ids)."""
        return self._id_to_token.get(int(token_id), UNK_TOKEN)

    def merge_rank(self, pair: Tuple[str, str]) -> Optional[int]:
        """Position of ``pair`` in the merge table; ``None`` if never merged."""
        return self._merge_ranks.get((pair[0], pair[1]))

    # --------------------------------------------------------------- summary
    def top_merges(self, count: int = 10) -> List[Dict[str, Any]]:
        """The ``count`` highest-frequency merge rules, most frequent first."""
        rows = [
            {"pair": list(pair), "token": pair[0] + pair[1], "count": freq, "rank": rank}
            for rank, (pair, freq) in enumerate(zip(self.merges, self.merge_counts))
        ]
        rows.sort(key=lambda row: (-row["count"], row["rank"]))
        return rows[:count]


    # ------------------------------------------------------------- training
    def train(
        self,
        texts: Iterable[str],
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        min_pair_frequency: int = MIN_PAIR_FREQUENCY,
        verbose: bool = True,
        log_every: int = 1000,
    ) -> Dict[str, Any]:
        """Learn ``vocab`` and ``merges`` from ``texts``.

        Pairs are counted over *unique* words weighted by frequency, and only the
        words that actually contain the winning pair are re-counted after each
        merge — a full corpus re-scan per merge would be quadratic.

        Stops early at *natural convergence*: when the most frequent remaining
        pair occurs fewer than ``min_pair_frequency`` times there is no useful
        merge left, so the vocabulary ends up smaller than ``vocab_size``.

        Returns a stats dict.
        """
        if vocab_size < len(SPECIAL_TOKENS):
            raise ValueError(f"vocab_size must be >= {len(SPECIAL_TOKENS)}")

        word_freqs = self._count_words(texts)
        words = list(word_freqs)
        freqs = [word_freqs[word] for word in words]
        symbols: List[List[str]] = [word_symbols(word) for word in words]

        # ids 0-4: specials.  Next: every single character seen in the corpus.
        self.vocab = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        characters = sorted({symbol for sequence in symbols for symbol in sequence})
        for character in characters:
            self.vocab[character] = len(self.vocab)

        pair_counts, pair_words = self._initial_pair_counts(symbols, freqs)
        heap: List[Tuple[int, Tuple[str, str]]] = [
            (-count, pair) for pair, count in pair_counts.items()
        ]
        heapq.heapify(heap)

        target_merges = max(0, vocab_size - len(self.vocab))
        stopped_early = False

        while len(self.merges) < target_merges:
            best = self._pop_best_pair(heap, pair_counts)
            if best is None or pair_counts[best] < min_pair_frequency:
                stopped_early = True
                break

            merged_token = best[0] + best[1]
            if merged_token in self.vocab:
                # Defensive: a different pair already produced this string.
                # Drop the pair and carry on rather than corrupt the id space.
                pair_counts.pop(best, None)
                pair_words.pop(best, None)
                continue

            self.vocab[merged_token] = len(self.vocab)
            self.merges.append([best[0], best[1]])
            self.merge_counts.append(int(pair_counts[best]))

            # Snapshot the word list first: pair_words[best] shrinks as we go.
            for word_index in sorted(pair_words.get(best, ())):
                old_symbols = symbols[word_index]
                new_symbols = merge_pair(old_symbols, best, merged_token)
                symbols[word_index] = new_symbols
                self._apply_merge(
                    word_index,
                    pair_counter(old_symbols),
                    pair_counter(new_symbols),
                    freqs[word_index],
                    pair_counts,
                    pair_words,
                    heap,
                )

            pair_counts.pop(best, None)
            pair_words.pop(best, None)

            done = len(self.merges)
            if verbose and log_every and done % log_every == 0:
                print(
                    f"    merge {done:>5d}/{target_merges}  vocab={len(self.vocab):>5d}  "
                    f"new={display_token(merged_token)!r}  count={self.merge_counts[-1]:,}",
                    flush=True,
                )

        self._rebuild_index()
        return {
            "word_types": len(words),
            "word_instances": int(sum(freqs)),
            "characters": len(characters),
            "target_vocab_size": int(vocab_size),
            "vocab_size": len(self.vocab),
            "merges": len(self.merges),
            "merges_start_id": len(SPECIAL_TOKENS) + len(characters),
            "stopped_early": stopped_early,
            "min_pair_frequency": int(min_pair_frequency),
        }


    # ------------------------------------------------------- training helpers
    @staticmethod
    def _count_words(texts: Iterable[str]) -> Counter:
        """Normalise ``texts`` and count unique words (space-split)."""
        word_freqs: Counter = Counter()
        for text in texts:
            for word in split_words(normalize_text(text)):
                word_freqs[word] += 1
        return word_freqs

    @staticmethod
    def _initial_pair_counts(
        symbols: Sequence[Sequence[str]], freqs: Sequence[int]
    ) -> Tuple[Counter, Dict[Tuple[str, str], Set[int]]]:
        """Weighted pair counts plus, per pair, the word indices containing it."""
        pair_counts: Counter = Counter()
        pair_words: Dict[Tuple[str, str], Set[int]] = {}
        for word_index, sequence in enumerate(symbols):
            frequency = freqs[word_index]
            for pair, multiplicity in pair_counter(sequence).items():
                pair_counts[pair] += multiplicity * frequency
                pair_words.setdefault(pair, set()).add(word_index)
        return pair_counts, pair_words

    @staticmethod
    def _pop_best_pair(
        heap: List[Tuple[int, Tuple[str, str]]],
        pair_counts: Counter,
    ) -> Optional[Tuple[str, str]]:
        """Pop the most frequent live pair (stale heap entries are skipped).

        Heap keys are ``(-count, pair)`` so ties resolve to the lexicographically
        smaller pair, which makes training fully deterministic.
        """
        while heap:
            negative_count, pair = heapq.heappop(heap)
            live = pair_counts.get(pair)
            if live is not None and live == -negative_count:
                return pair
        return None

    @staticmethod
    def _apply_merge(
        word_index: int,
        old_pairs: Counter,
        new_pairs: Counter,
        frequency: int,
        pair_counts: Counter,
        pair_words: Dict[Tuple[str, str], Set[int]],
        heap: List[Tuple[int, Tuple[str, str]]],
    ) -> None:
        """Incrementally repair the pair statistics after one word changed.

        ``old_pairs`` / ``new_pairs`` are the pair *multisets of that single
        word*, so the weighted delta is ``(new - old) * frequency``.
        """
        for pair in set(old_pairs) | set(new_pairs):
            delta = (new_pairs.get(pair, 0) - old_pairs.get(pair, 0)) * frequency
            if delta == 0:
                continue
            new_count = pair_counts.get(pair, 0) + delta
            if new_count > 0:
                pair_counts[pair] = new_count
                if new_pairs.get(pair, 0) > 0:
                    pair_words.setdefault(pair, set()).add(word_index)
                elif pair in pair_words:
                    pair_words[pair].discard(word_index)
                heapq.heappush(heap, (-new_count, pair))
            else:
                pair_counts.pop(pair, None)
                pair_words.pop(pair, None)

    def _rebuild_index(self) -> None:
        """Refresh the id->token map and the merge ranks from vocab/merges."""
        self._id_to_token = {index: token for token, index in self.vocab.items()}
        self._merge_ranks = {(pair[0], pair[1]): rank for rank, pair in enumerate(self.merges)}


    # --------------------------------------------------------------- encoding
    def _apply_merges(self, symbols: List[str]) -> List[str]:
        """Repeatedly apply the lowest-ranked applicable merge to one word.

        Merges only concatenate symbols, so the concatenation of the result is
        independent of the order in which they are applied — which is exactly
        why :meth:`decode` can simply join the tokens back together.
        """
        while len(symbols) > 1:
            best_rank: Optional[int] = None
            best_index = -1
            for index in range(len(symbols) - 1):
                rank = self._merge_ranks.get((symbols[index], symbols[index + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_index = index
            if best_index < 0:
                break
            symbols = (
                symbols[:best_index]
                + [symbols[best_index] + symbols[best_index + 1]]
                + symbols[best_index + 2:]
            )
        return symbols

    def tokenize(self, text: str) -> List[str]:
        """``text`` -> list of BPE token *strings* (no ids, no padding)."""
        tokens: List[str] = []
        for word in split_words(normalize_text(text)):
            tokens.extend(self._apply_merges(word_symbols(word)))
        return tokens

    def encode_unpadded(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """``text`` -> token ids, never padded.

        ``max_length=None`` keeps the full sequence, which is what the
        round-trip test and the trainers need to see true lengths.
        """
        ids = [self.token_id(token) for token in self.tokenize(text)]
        if max_length is not None:
            ids = ids[: max(0, int(max_length))]
        return ids

    def encode(
        self,
        text: str,
        max_length: Optional[int] = DEFAULT_MAX_LENGTH,
        add_special_tokens: bool = False,
    ) -> List[int]:
        """``text`` -> exactly ``max_length`` ids: truncate if longer, pad if shorter.

        ``max_length`` defaults to ``config.MODEL["max_seq_len"]`` with ``[PAD]``
        filling — 64 since TASK 19R switched the environment to the WikiText
        sentence-level contract.  Pass ``max_length=None`` for the lossless,
        unpadded sequence.
        """
        ids = self.encode_unpadded(text, max_length=None)
        if add_special_tokens:
            ids = [BOS_ID, *ids, EOS_ID]
        if max_length is None:
            return ids
        max_length = max(0, int(max_length))
        if len(ids) >= max_length:
            return ids[:max_length]
        return ids + [PAD_ID] * (max_length - len(ids))

    def encode_batch(
        self,
        texts: Sequence[str],
        max_length: Optional[int] = DEFAULT_MAX_LENGTH,
        add_special_tokens: bool = False,
    ) -> List[List[int]]:
        """Apply :meth:`encode` to every string in ``texts``."""
        return [self.encode(text, max_length, add_special_tokens) for text in texts]


    # --------------------------------------------------------------- decoding
    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Invert :meth:`encode`: join tokens and turn ``Ġ`` back into spaces."""
        pieces: List[str] = []
        for token_id in ids:
            token = self._id_to_token.get(int(token_id))
            if token is None:
                continue
            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue
            pieces.append(token)
        return "".join(pieces).replace(WORD_PREFIX, " ").strip()

    # ------------------------------------------------------------------- io
    def to_dict(self) -> Dict[str, Any]:
        """Exactly the two documented keys, JSON-serialisable as-is."""
        return {"vocab": dict(self.vocab), "merges": [list(pair) for pair in self.merges]}

    def save(self, path: Optional[Path] = None) -> Path:
        """Write ``{"vocab": {...}, "merges": [[a, b], ...]}`` to ``path``."""
        path = Path(path) if path is not None else config.path("tokenizer")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "BPETokenizer":
        """Rebuild a tokenizer from the JSON written by :meth:`save`."""
        path = Path(path) if path is not None else config.path("tokenizer")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if set(payload) != {"vocab", "merges"}:
            raise ValueError(f"unexpected tokenizer keys in {path}: {sorted(payload)}")

        tokenizer = cls()
        tokenizer.vocab = {str(token): int(index) for token, index in payload["vocab"].items()}
        tokenizer.merges = [[str(pair[0]), str(pair[1])] for pair in payload["merges"]]
        tokenizer.merge_counts = []
        tokenizer._rebuild_index()

        missing = [token for token in SPECIAL_TOKENS if token not in tokenizer.vocab]
        if missing:
            raise ValueError(f"tokenizer file {path} is missing special tokens {missing}")
        for token, expected in zip(SPECIAL_TOKENS, range(len(SPECIAL_TOKENS))):
            if tokenizer.vocab[token] != expected:
                raise ValueError(
                    f"special token {token!r} has id {tokenizer.vocab[token]}, expected {expected}"
                )
        return tokenizer

    def vocabulary_rows(self, limit: int = 20) -> List[Tuple[int, str]]:
        """``[(id, token)]`` for the first ``limit`` ids — a vocab-layout readout."""
        return [(index, self._id_to_token[index]) for index in sorted(self._id_to_token)][:limit]


if __name__ == "__main__":
    raise SystemExit(
        "Training lives in environment/tokenizer/train_tokenizer.py (TASK 19)."
    )
