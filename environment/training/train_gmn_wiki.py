"""Train the GMN style encoder on WikiText-103 (TASK 21R).

Objective: **next-sentence contrastive prediction** — SimCSE / CLIP style
in-batch-negative NT-Xent over ``(S_n, S_n+1)`` pairs.  No token prediction
anywhere: this is the contrastive arm of the controlled comparison, and its
GPT-2 counterpart (TASK 20R) used the same corpus, the same config dict, the
same optimiser and the same schedule.

    anchor_i   = encode(S_n_i)      (256,) unit vector
    positive_i = encode(S_n+1_i)    (256,) unit vector
    S          = A @ P.T                      (B, B) cosine similarities
    loss       = 0.5 * (CE(S/T, arange(B)) + CE(S.T/T, arange(B)))     symmetric
                 with the hardest negative of each row penalised by +0.5

Three objective settings were changed **on request** after the first trajectory
showed in-batch alignment pinned at 0.94-0.95 with uniformity only -0.2 (TASK
21R note 2 specified plain ``CE(S/T)`` at T=0.07):

* ``T = 0.07 -> 0.05`` (``--temperature``): sharper softmax, harder negatives;
* ``hard-negative penalty = 0.5`` (``--hard-negative-penalty``): a constant is
  added to each row's hardest-negative logit, so the gradient cannot ignore the
  negatives it already finds confusable.  Non-standard, off at 0.0;
* symmetric loss (``--no-symmetric`` disables): every positive also predicts its
  anchor, doubling the gradient signal.

``--temperature 0.07 --hard-negative-penalty 0 --no-symmetric`` restores the
original TASK 21R objective exactly.  All three settings are recorded in the
checkpoint and the history file.

Data (TASK 21R notes 1-2)
-------------------------
``train_pairs.json`` (1.0 GB, 3,570,028 pairs) is streamed **line by line**, one
pair per line, and sentences are tokenized on the fly — the file is never
materialised.  To keep in-batch negatives honest the stream is shuffled inside a
rolling buffer of :data:`SHUFFLE_BUFFER` pairs before batching: pairs that are
adjacent *in the file* are also adjacent in the text (pair ``i``'s positive is
pair ``i+1``'s anchor), so a purely sequential pass would put near-duplicate
sentences in the same batch as false negatives.  A 65,536-pair buffer makes the
chance that a pair's file-neighbour shares its batch ``63 / 65535 = 0.10%``.

Padding (TASK 21R note 3)
-------------------------
Each sentence is ``[BOS] tokens [EOS]`` capped at ``seq_len - 1 = 63`` tokens so
that the encoder's own ``[CLS]`` prefix lands exactly on ``seq_len``.  Batches are
padded to the *batch* maximum and ``key_padding_mask`` is passed to every
attention block; TASK 19R verified that this reproduces a trimmed batch exactly
(min cosine 1.000000, max abs diff 1e-7) whereas omitting it drops the cosine to
0.647.  Dynamic padding is therefore not an approximation, and it roughly halves
the tokens pushed through the trunk.

Collapse watchdog (TASK 21R note 4)
-----------------------------------
Alignment (mean positive cosine) and uniformity (``log E[exp(-2 d^2)]`` over
in-batch pairs, SimCSE's definition) are measured every log interval.  Collapse
is ``alignment > 0.99 AND uniformity > -1.0``: the run stops, reports, and names
the restart command with ``--temperature 0.05``.  Healthy ranges: alignment
0.2-0.8, uniformity -3.0 to -1.5.

Evaluation (TASK 21R note 5)
----------------------------
Per epoch, Recall@1 over 1,000 seeded val pairs: each anchor is scored against
its true next sentence plus 99 random other positives (random baseline 1%,
target > 20%).  The stricter "true positive is the row maximum among all 999
others" figure and MRR are reported as well.

Usage (from the repository root)::

    python environment/training/train_gmn_wiki.py --smoke
    python environment/training/train_gmn_wiki.py --epochs 2
    python environment/training/train_gmn_wiki.py --temperature 0.05   # after collapse
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ``environment/``, ``environment/models/``, ``environment/tokenizer/`` and
# ``environment/training/`` are not package roots, so expose them on sys.path
# (TASK 18/19/20 convention).
TRAINING_DIR = Path(__file__).resolve().parent
ENVIRONMENT_DIR = TRAINING_DIR.parent
MODELS_DIR = ENVIRONMENT_DIR / "models"
TOKENIZER_DIR = ENVIRONMENT_DIR / "tokenizer"
for _directory in (ENVIRONMENT_DIR, MODELS_DIR, TOKENIZER_DIR, TRAINING_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import config  # noqa: E402
from bpe import BOS_ID, EOS_ID, PAD_ID, BPETokenizer  # noqa: E402
from gmn_style import LATENT_DIM, GMNStyle  # noqa: E402

# The LR schedule is imported from the TASK 21 ARC trainer so both arms share one
# implementation of "linear warmup then cosine decay" (as train_gpt2_wiki.py does).
from train_gpt2 import build_scheduler  # noqa: E402

#: NT-Xent temperature.  TASK 21R note 2 specified 0.07; reduced to **0.05 on
#: request** after the first trajectory showed in-batch alignment sitting at
#: 0.88-0.93 with uniformity only -0.3, i.e. the representations were barely
#: separating.  A lower temperature sharpens the softmax, so the negative term
#: pulls harder and the embeddings spread out more.  0.07 is still available via
#: ``--temperature 0.07`` for comparison, and 0.05 is also the value TASK 21R
#: note 4 names as the post-collapse restart setting.
DEFAULT_TEMPERATURE = 0.05

#: FIX 2 (on request): constant added to each row's hardest-negative logit.
#: 0.0 disables hard-negative mining and recovers plain NT-Xent.
DEFAULT_HARD_NEGATIVE_PENALTY = 0.5

#: FIX 3 (on request): also score ``CE(S.T / T)``, i.e. let every positive predict
#: its anchor.  Doubles the gradient signal and improves uniformity (CLIP style).
DEFAULT_SYMMETRIC = True

#: Log cadence: step | loss | alignment | uniformity | lr | elapsed (TASK 21R note 6).
LOG_EVERY = 500

#: Collapse watchdog thresholds and the healthy band to report against.
COLLAPSE_ALIGNMENT = 0.99
COLLAPSE_UNIFORMITY = -1.0
HEALTHY_ALIGNMENT = (0.2, 0.8)
HEALTHY_UNIFORMITY = (-3.0, -1.5)

#: Pairs buffered before shuffling (see the module docstring for the rationale).
SHUFFLE_BUFFER = 65_536

#: Recall@1 protocol: 1,000 pairs, 99 random negatives each, seeded.
EVAL_PAIRS = 1000
EVAL_NEGATIVES = 99

#: TASK 21R note 7: keep only this many ``*_epoch{N}.pt`` files on disk.
KEEP_EPOCH_CHECKPOINTS = 2

#: Recall@1 gates from TASK 21R note 8.
RECALL_CONTINUE = 0.05
RECALL_WORRYING = 0.02


def configure_stdout() -> None:
    """Force UTF-8 output so tokens such as ``Ġ`` survive a cp1252 console."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover - non-TextIO
            pass


# ---------------------------------------------------------------------------
# Streaming pair pipeline (TASK 21R notes 1-3)
# ---------------------------------------------------------------------------
def iter_pairs(path: Path, limit: Optional[int] = None) -> Iterator[Tuple[str, str]]:
    """Stream ``[S_n, S_n+1]`` pairs out of a one-element-per-line JSON array.

    Each line carries one pair plus the array's own separators, so
    ``JSONDecoder.raw_decode`` parses the leading value and ignores the rest —
    ``train_pairs.json`` (1.0 GB) is never materialised.
    """
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        count = 0
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if count == 0 and line.startswith("["):
                line = line[1:]
            pair, _ = decoder.raw_decode(line)
            yield str(pair[0]), str(pair[1])
            count += 1
            if limit is not None and count >= limit:
                break


def count_pairs(path: Path) -> int:
    """Number of pairs in a ``*_pairs.json`` file, streamed (no tokenizing)."""
    return sum(1 for _ in iter_pairs(path))


def sentence_ids(tokenizer: BPETokenizer, sentence: str) -> List[int]:
    """``[BOS] tokens [EOS]`` for one sentence, uncapped."""
    return [BOS_ID, *tokenizer.encode_unpadded(sentence, max_length=None), EOS_ID]


def cap_ids(ids: Sequence[int], max_raw: int) -> Tuple[np.ndarray, bool]:
    """Cap to ``max_raw`` ids, keeping ``[EOS]``: ``(array, was_truncated)``.

    ``max_raw = seq_len - 1`` leaves room for the ``[CLS]`` the encoder prepends,
    so nothing the encoder receives is truncated.  When a sentence is too long its
    middle is dropped and ``[EOS]`` is kept, so the end marker survives.
    """
    if len(ids) > max_raw:
        return np.asarray(ids[: max_raw - 1] + [EOS_ID], dtype=np.uint16), True
    return np.asarray(ids, dtype=np.uint16), False


def tokenize_sentence(tokenizer: BPETokenizer, sentence: str, max_raw: int) -> np.ndarray:
    """Capped ``[BOS] tokens [EOS]`` as a compact uint16 array (evaluation helper)."""
    return cap_ids(sentence_ids(tokenizer, sentence), max_raw)[0]


def stack_sentences(sentences: Sequence[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Pad a batch to its own maximum and return ``(ids, key_padding_mask, width)``.

    ``key_padding_mask`` is ``True`` at ``[PAD]`` positions, which is what the
    trunk's ``masked_fill`` expects.  Padding to the batch maximum rather than to
    ``seq_len`` is exact, not an approximation: TASK 19R measured masked-vs-trimmed
    cosine 1.000000 with a max absolute difference of 1e-7.
    """
    width = max(int(sentence.size) for sentence in sentences)
    ids = np.full((len(sentences), width), PAD_ID, dtype=np.int64)
    for row, sentence in enumerate(sentences):
        ids[row, : sentence.size] = sentence
    mask = ids == PAD_ID
    return torch.from_numpy(ids), torch.from_numpy(mask), width


class ShuffledPairStream:
    """Rolling shuffle buffer over the streamed pairs.

    Consuming a buffer sequentially and re-shuffling it only when it runs dry
    keeps mixing good (a pair is drawn into a batch at a random point of a
    ~1,000-step window) while re-shuffling 65,536 items once per ~1,023 steps
    instead of once per step.
    """

    def __init__(
        self,
        path: Path,
        tokenizer: BPETokenizer,
        max_raw: int,
        capacity: int = SHUFFLE_BUFFER,
        seed: int = 0,
        limit: Optional[int] = None,
    ) -> None:
        self.path = Path(path)
        self.capacity = int(capacity)
        self.pairs_read = 0
        self.sentences = 0
        self.truncated = 0
        self.exhausted = False
        self._tokenizer = tokenizer
        self._max_raw = int(max_raw)
        self._stream = iter_pairs(self.path, limit)
        self._rng = np.random.default_rng(seed)
        self._anchors: List[np.ndarray] = []
        self._positives: List[np.ndarray] = []
        self._cursor = 0
        self._fill()
        self._shuffle()

    def _fill(self) -> None:
        """Top the buffer up from the stream (stopping when the file ends)."""
        while len(self._anchors) < self.capacity and not self.exhausted:
            try:
                anchor, positive = next(self._stream)
            except StopIteration:
                self.exhausted = True
                break
            for sentence, target in ((anchor, self._anchors), (positive, self._positives)):
                ids, truncated = cap_ids(
                    sentence_ids(self._tokenizer, sentence), self._max_raw
                )
                target.append(ids)
                self.sentences += 1
                self.truncated += int(truncated)
            self.pairs_read += 1

    @property
    def truncated_percent(self) -> float:
        """Share of sentences that hit the ``max_raw`` cap (TASK 18R flagged 3%)."""
        return 100.0 * self.truncated / self.sentences if self.sentences else 0.0

    def _shuffle(self) -> None:
        order = self._rng.permutation(len(self._anchors))
        self._anchors = [self._anchors[index] for index in order]
        self._positives = [self._positives[index] for index in order]
        self._cursor = 0

    def __len__(self) -> int:
        return len(self._anchors) - self._cursor

    def next_batch(self, batch_size: int) -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
        """Next ``batch_size`` pairs, or ``None`` when the epoch is finished."""
        if self._cursor + batch_size > len(self._anchors):
            remaining = len(self._anchors) - self._cursor
            if self.exhausted and remaining == 0:
                return None
            self._anchors = self._anchors[self._cursor:]
            self._positives = self._positives[self._cursor:]
            self._fill()
            self._shuffle()
            if len(self._anchors) < batch_size:
                return None
        batch = list(zip(
            self._anchors[self._cursor:self._cursor + batch_size],
            self._positives[self._cursor:self._cursor + batch_size],
        ))
        self._cursor += batch_size
        return batch


# ---------------------------------------------------------------------------
# Contrastive objective (TASK 21R note 2) and geometry diagnostics (note 4)
# ---------------------------------------------------------------------------
def uniformity_of(embeddings: torch.Tensor) -> float:
    """SimCSE uniformity: ``log E[exp(-2 * ||x_i - x_j||^2)]`` over ``i != j``.

    Unit vectors satisfy ``||x_i - x_j||^2 = 2 - 2 cos``, so this follows from the
    Gram matrix.  More negative = representations spread out more evenly.
    """
    count = int(embeddings.shape[0])
    gram = embeddings @ embeddings.transpose(0, 1)
    squared = (2.0 - 2.0 * gram).clamp_min(0.0)
    off_diagonal = ~torch.eye(count, dtype=torch.bool)
    return float(torch.log(torch.exp(-2.0 * squared[off_diagonal]).mean()))


def contrastive_forward(
    model: GMNStyle,
    anchors: Sequence[np.ndarray],
    positives: Sequence[np.ndarray],
    temperature: float,
    hard_negative_penalty: float = 0.0,
    symmetric: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """One contrastive step: in-batch negatives over the ``(B, B)`` cosine matrix.

    ``loss = 0.5 * (CE(S/T) + CE(S.T/T))`` when ``symmetric`` (CLIP style: every
    sentence predicts its partner *and* is predicted by it), otherwise just
    ``CE(S/T)``.

    ``hard_negative_penalty`` adds a constant to each row's hardest negative logit
    before the softmax — the highest-cosine non-positive in the batch — so the
    gradient cannot ignore the cases it finds confusable.  This is a
    **non-standard** addition on top of NT-Xent, applied on request; 0.0 disables
    it and recovers plain NT-Xent.

    Returns ``(loss, embeddings, diagnostics)`` where ``embeddings`` is the
    ``(2B, 256)`` unit matrix holding the anchors followed by the positives.
    """
    sentences = list(anchors) + list(positives)
    ids, mask, width = stack_sentences(sentences)
    embeddings = model.encode(ids, key_padding_mask=mask)
    batch = len(anchors)
    cosine = embeddings[:batch] @ embeddings[batch:].transpose(0, 1)   # unit vectors
    labels = torch.arange(batch, dtype=torch.long)
    diagonal = torch.eye(batch, dtype=torch.bool)

    def logits_and_hardest(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``matrix / T`` plus, per row, the index and cosine of the hardest negative."""
        scaled = matrix / float(temperature)
        if not hard_negative_penalty:
            return scaled, torch.empty(0, dtype=torch.long)
        hidden = scaled.masked_fill(diagonal, float("-inf"))
        hardest = hidden.argmax(dim=1)
        penalty = torch.zeros_like(scaled)
        penalty[torch.arange(batch), hardest] = float(hard_negative_penalty)
        return scaled + penalty, hardest

    anchor_logits, anchor_hardest = logits_and_hardest(cosine)
    loss_anchor = F.cross_entropy(anchor_logits, labels)

    loss_positive = torch.zeros((), dtype=loss_anchor.dtype)
    positive_hardest = torch.empty(0, dtype=torch.long)
    if symmetric:
        positive_logits, positive_hardest = logits_and_hardest(cosine.transpose(0, 1))
        loss_positive = F.cross_entropy(positive_logits, labels)

    loss = 0.5 * (loss_anchor + loss_positive) if symmetric else loss_anchor

    with torch.no_grad():
        off_diagonal = cosine.masked_fill(diagonal, float("-inf"))
        hardest_cosine = float(off_diagonal.max(dim=1).values.mean()) if batch > 1 else 0.0
    diagnostics = {
        "batch": batch,
        "width": width,
        "padded_positions": int(mask.sum()),
        "positions": int(ids.numel()),
        "loss_anchor": float(loss_anchor.detach()),
        "loss_positive": float(loss_positive.detach()) if symmetric else None,
        "hardest_negative_cosine": hardest_cosine,
        "hard_negative_penalty": float(hard_negative_penalty),
        "symmetric": bool(symmetric),
        "hard_negative_indices": int(anchor_hardest.numel() + positive_hardest.numel()),
    }
    return loss, embeddings, diagnostics


def geometry_of(embeddings: torch.Tensor, batch: int) -> Dict[str, float]:
    """Alignment (mean positive cosine) and uniformity, detached from the graph."""
    with torch.no_grad():
        cosine = embeddings[:batch] @ embeddings[batch:].transpose(0, 1)
        return {
            "alignment": float(cosine.diagonal().mean()),
            "uniformity": uniformity_of(embeddings),
        }


def collapse_warning(alignment: float, uniformity: float) -> Optional[str]:
    """The TASK 21R note 4 collapse rule, or ``None`` while the run is healthy."""
    if alignment > COLLAPSE_ALIGNMENT and uniformity > COLLAPSE_UNIFORMITY:
        return (
            f"COLLAPSE DETECTED: alignment {alignment:.4f} > {COLLAPSE_ALIGNMENT} "
            f"AND uniformity {uniformity:.4f} > {COLLAPSE_UNIFORMITY} — every "
            "sentence is mapping to nearly the same vector, so the contrastive loss "
            "carries no information.  Stopping; restart with --temperature 0.05 "
            "(TASK 21R note 4)."
        )
    return None


def health_note(alignment: float, uniformity: float) -> str:
    """A short verdict against the healthy bands TASK 21R quotes."""
    alignment_ok = HEALTHY_ALIGNMENT[0] <= alignment <= HEALTHY_ALIGNMENT[1]
    uniformity_ok = HEALTHY_UNIFORMITY[0] <= uniformity <= HEALTHY_UNIFORMITY[1]
    verdict = "healthy" if (alignment_ok and uniformity_ok) else "OUTSIDE BAND"
    return (f"{verdict} (targets: alignment {HEALTHY_ALIGNMENT[0]}-{HEALTHY_ALIGNMENT[1]}, "
            f"uniformity {HEALTHY_UNIFORMITY[0]}-{HEALTHY_UNIFORMITY[1]})")


# ---------------------------------------------------------------------------
# Checkpoints and results (TASK 21R note 7)
# ---------------------------------------------------------------------------
def epoch_checkpoint_path(epoch: int) -> Path:
    """``environment/checkpoints/gmn_wiki_epoch{N}.pt``."""
    return config.path("gmn_wiki_ckpt").with_name(f"gmn_wiki_epoch{epoch}.pt")


def prune_epoch_checkpoints(keep: int = KEEP_EPOCH_CHECKPOINTS) -> List[str]:
    """Delete all but the newest ``keep`` ``gmn_wiki_epoch*.pt`` files."""
    best = config.path("gmn_wiki_ckpt")
    candidates = sorted(
        best.parent.glob("gmn_wiki_epoch*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    removed: List[str] = []
    for stale in candidates[keep:]:
        removed.append(stale.name)
        stale.unlink()
    return removed


def save_checkpoint(path: Path, payload: Dict[str, Any]) -> Path:
    """Write a checkpoint, creating the directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def write_result(result: Dict[str, Any]) -> Path:
    """Write the full result record to ``environment/results/gmn_wiki_history.json``."""
    path = config.path("gmn_wiki_history")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return path


def write_progress_snapshot(
    history: List[Dict[str, Any]],
    mid_epoch: List[Dict[str, Any]],
    best: Dict[str, Any],
    stop_reason: str,
) -> Path:
    """Refresh the history file after each epoch so a crash keeps the curves."""
    path = config.path("gmn_wiki_history")
    existing: Dict[str, Any] = {}
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
        except json.JSONDecodeError:  # pragma: no cover - only on a torn write
            existing = {}
    existing.update({
        "script": "train_gmn_wiki.py",
        "task": "TASK 21R",
        "history": history,
        "mid_epoch": mid_epoch,
        "best": dict(best),
        "epochs_completed": len(history),
        "stop_reason": stop_reason,
        "in_progress": True,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2)
    return path


# ---------------------------------------------------------------------------
# Evaluation: Recall@1 (TASK 21R note 5)
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_sentences(
    model: GMNStyle,
    sentences: Sequence[np.ndarray],
    batch_size: int = 64,
) -> torch.Tensor:
    """Encode tokenized sentences in batches -> ``(N, 256)`` unit vectors."""
    was_training = model.training
    model.eval()
    chunks = []
    for start in range(0, len(sentences), batch_size):
        window = sentences[start:start + batch_size]
        ids, mask, _ = stack_sentences(window)
        chunks.append(model.encode(ids, key_padding_mask=mask))
    model.train(was_training)
    return torch.cat(chunks, dim=0)


def load_val_pairs(path: Path, limit: Optional[int] = None) -> List[Tuple[str, str]]:
    """Val pairs are small (2 MB), so unlike the train file they are held in RAM."""
    return list(iter_pairs(path, limit))


@torch.no_grad()
def evaluate_recall(
    model: GMNStyle,
    pairs: Sequence[Tuple[str, str]],
    tokenizer: BPETokenizer,
    max_raw: int,
    n_pairs: int = EVAL_PAIRS,
    n_negatives: int = EVAL_NEGATIVES,
    seed: int = 0,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """Recall@1 by the TASK 21R protocol, plus stricter figures and val geometry.

    Each of ``n_pairs`` seeded anchors is scored against its own next sentence and
    ``n_negatives`` random other positives; Recall@1 is the share whose true
    positive outscores all sampled negatives.  ``recall_at_1_full_pool`` instead
    requires the true positive to beat *every* other candidate (a harder question),
    and ``mrr`` ranks over the sampled candidates.  Alignment and uniformity over
    the same fixed sample are reported too, so the geometry numbers are comparable
    from epoch to epoch (the per-step figures come from a random training batch).
    """
    pool_size = min(int(n_pairs), len(pairs))
    rng = np.random.default_rng(seed)
    if len(pairs) > pool_size:
        chosen = rng.choice(len(pairs), pool_size, replace=False)
        sample = [pairs[int(index)] for index in chosen]
    else:
        sample = list(pairs)

    anchors = [tokenize_sentence(tokenizer, anchor, max_raw) for anchor, _ in sample]
    positives = [tokenize_sentence(tokenizer, positive, max_raw) for _, positive in sample]
    started = time.perf_counter()
    anchor_vec = encode_sentences(model, anchors, batch_size)
    positive_vec = encode_sentences(model, positives, batch_size)
    cosine = anchor_vec @ positive_vec.transpose(0, 1)
    true_scores = cosine.diagonal()

    recall_sampled = 0
    ranks: List[int] = []
    for index in range(pool_size):
        candidates = np.delete(np.arange(pool_size), index)
        negatives = rng.choice(candidates, int(n_negatives), replace=False)
        beaten_by = int((cosine[index, negatives] >= true_scores[index]).sum())
        recall_sampled += int(beaten_by == 0)
        ranks.append(beaten_by + 1)

    ranks_full = (cosine > true_scores.unsqueeze(1)).sum(dim=1)
    together = torch.cat([anchor_vec, positive_vec], dim=0)
    return {
        "pairs_evaluated": pool_size,
        "negatives_per_anchor": int(n_negatives),
        "random_baseline_recall_at_1": round(1.0 / (1.0 + int(n_negatives)), 4),
        "recall_at_1": round(recall_sampled / pool_size, 4),
        "recall_at_1_full_pool": round(float((ranks_full == 0).float().mean()), 4),
        "recall_at_5_full_pool": round(float((ranks_full < 5).float().mean()), 4),
        "mrr_sampled": round(float(np.mean(1.0 / np.asarray(ranks))), 4),
        "val_alignment": round(float(cosine.diagonal().mean()), 4),
        "val_uniformity": round(uniformity_of(together), 4),
        "seconds": round(time.perf_counter() - started, 1),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the whole TASK 21R training and return the result record."""
    config.ensure_dirs()
    torch.manual_seed(int(args.seed))
    threads = int(args.threads) if args.threads else (os.cpu_count() or 1)
    torch.set_num_threads(threads)
    try:  # CPU-only nicety: denormals are pure overhead here
        torch.set_flush_denormal(True)
    except RuntimeError:  # pragma: no cover - platform dependent
        pass

    model_config = config.model_config(args.model)
    seq_len = int(model_config["max_seq_len"])
    max_raw = seq_len - 1
    tokenizer = BPETokenizer.load(config.path("tokenizer_wiki"))
    train_pairs_path = config.path("wiki_train_pairs")
    val_pairs_path = config.path("wiki_val_pairs")

    print("=" * 74)
    print("TASK 21R — GMN style pretraining on WikiText-103")
    print("=" * 74)
    print(f"  model config : {args.model}  {model_config}")
    print(f"  latent dim   : {LATENT_DIM} (encoder output width, always)")
    print(f"  tokenizer    : {config.path('tokenizer_wiki')}  "
          f"(vocab {tokenizer.vocab_size:,})")
    print(f"  objective    : NT-Xent next-sentence contrastive, in-batch negatives, "
          f"T={args.temperature}")
    print(f"  batch        : {args.batch_size} pairs -> {2 * args.batch_size} encodes, "
          f"<= {seq_len} tokens each (dynamic padding, key_padding_mask)")
    print(f"  optimiser    : AdamW lr {args.lr} wd {args.weight_decay} | warmup "
          f"{args.warmup_steps} | cosine decay | clip {args.grad_clip}")
    print(f"  torch        : {torch.__version__}  threads {threads}")

    print("\n  counting train pairs (streamed, no tokenizing) ...", flush=True)
    count_started = time.perf_counter()
    n_train_pairs = (int(args.limit_pairs) if args.limit_pairs is not None
                     else count_pairs(train_pairs_path))
    val_pairs = load_val_pairs(val_pairs_path)
    print(f"  train pairs  : {n_train_pairs:,}  "
          f"({time.perf_counter() - count_started:.0f}s to count)")
    print(f"  val pairs    : {len(val_pairs):,}")

    steps_per_epoch = math.ceil(n_train_pairs / args.batch_size)
    total_steps = steps_per_epoch * int(args.epochs)
    if args.max_steps:
        total_steps = min(total_steps, int(args.max_steps))
    print(f"  schedule     : {steps_per_epoch:,} steps/epoch, {total_steps:,} total, "
          f"{args.epochs} epoch(s)")

    model = GMNStyle(config_dict=model_config)
    print(f"  parameters   : {model.count_parameters():,} "
          f"(shared trunk + {LATENT_DIM}-d projection head)")
    print(f"  logging      : every {args.log_every} steps"
          + (f", Recall@1 every {args.recall_every} steps" if args.recall_every else "")
          + f", checkpoints in {config.path('gmn_wiki_ckpt').parent}")
    print(flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = build_scheduler(optimizer, total_steps, args.warmup_steps)

    history: List[Dict[str, Any]] = []
    mid_epoch: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {"epoch": 0, "recall_at_1": -1.0}
    collapse: Dict[str, Any] = {}
    early_gate: Dict[str, Any] = {}
    stop_reason = f"completed all {args.epochs} requested epoch(s)"
    global_step = 0
    started = time.perf_counter()

    for epoch in range(1, int(args.epochs) + 1):
        stream = ShuffledPairStream(
            train_pairs_path, tokenizer, max_raw,
            capacity=int(args.shuffle_buffer), seed=int(args.seed) + epoch,
            limit=args.limit_pairs,
        )
        epoch_started = time.perf_counter()
        window_started = epoch_started
        epoch_loss_sum = 0.0
        window_loss_sum = 0.0
        window_steps = 0
        window_positions = 0
        steps_this_epoch = 0
        alignment_sum = 0.0
        uniformity_sum = 0.0
        geometry_samples = 0
        last_geometry = {"alignment": float("nan"), "uniformity": float("nan")}
        print(f"\n  epoch {epoch}: streaming pairs (shuffle buffer "
              f"{args.shuffle_buffer:,}, seed {int(args.seed) + epoch})", flush=True)

        while True:
            batch = stream.next_batch(int(args.batch_size))
            if batch is None:
                break
            anchors = [pair[0] for pair in batch]
            positives = [pair[1] for pair in batch]

            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss, embeddings, diag = contrastive_forward(
                model, anchors, positives, args.temperature,
                hard_negative_penalty=args.hard_negative_penalty,
                symmetric=args.symmetric,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            global_step += 1
            steps_this_epoch += 1
            step_loss = float(loss.detach())
            epoch_loss_sum += step_loss
            window_loss_sum += step_loss
            window_steps += 1
            window_positions += diag["positions"] - diag["padded_positions"]

            if global_step % args.log_every == 0:
                last_geometry = geometry_of(embeddings, diag["batch"])
                alignment_sum += last_geometry["alignment"]
                uniformity_sum += last_geometry["uniformity"]
                geometry_samples += 1
                now = time.perf_counter()
                mean = window_loss_sum / max(1, window_steps)
                rate = window_positions / max(1e-9, now - window_started)
                eta = ((now - epoch_started) * (steps_per_epoch - steps_this_epoch)
                       / max(1, steps_this_epoch))
                print(f"  step {global_step:>7,} | loss {mean:7.4f} | alignment "
                      f"{last_geometry['alignment']:6.4f} | uniformity "
                      f"{last_geometry['uniformity']:7.4f} | lr "
                      f"{optimizer.param_groups[0]['lr']:.3e} | {(now - started) / 60:7.1f} min | "
                      f"{rate / 1000:5.1f}K tok/s | epoch eta {eta / 3600:5.2f} h", flush=True)
                mid_epoch.append({
                    "kind": "geometry",
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": round(mean, 4),
                    "loss_anchor": round(diag["loss_anchor"], 4),
                    "loss_positive": (round(diag["loss_positive"], 4)
                                      if diag["loss_positive"] is not None else None),
                    "hardest_negative_cosine": round(diag["hardest_negative_cosine"], 4),
                    "alignment": round(last_geometry["alignment"], 4),
                    "uniformity": round(last_geometry["uniformity"], 4),
                })
                window_loss_sum = 0.0
                window_steps = 0
                window_positions = 0
                window_started = now

                warning = collapse_warning(
                    last_geometry["alignment"], last_geometry["uniformity"]
                )
                if warning:
                    collapse = {
                        "detected": True,
                        "epoch": epoch,
                        "global_step": global_step,
                        "alignment": round(last_geometry["alignment"], 4),
                        "uniformity": round(last_geometry["uniformity"], 4),
                        "temperature": args.temperature,
                        "suggested_temperature": 0.05,
                    }
                    stop_reason = f"COLLAPSE at step {global_step} (epoch {epoch})"
                    print(f"  {warning}", flush=True)
                    break

            if (args.recall_every and global_step % int(args.recall_every) == 0
                    and steps_this_epoch < steps_per_epoch):
                mid = evaluate_recall(
                    model, val_pairs, tokenizer, max_raw, seed=int(args.seed),
                    n_pairs=int(args.eval_pairs), batch_size=int(args.eval_batch_size),
                )
                mid_epoch.append({"kind": "recall", "epoch": epoch,
                                  "global_step": global_step, **mid})
                print(f"  step {global_step:>7,} | mid-epoch Recall@1 "
                      f"{mid['recall_at_1']:.2%} (full-pool {mid['recall_at_1_full_pool']:.2%}, "
                      f"MRR {mid['mrr_sampled']:.4f}; val alignment {mid['val_alignment']:.4f}, "
                      f"val uniformity {mid['val_uniformity']:.4f}; {mid['seconds']:.0f}s)",
                      flush=True)

            if args.max_steps and global_step >= int(args.max_steps):
                stop_reason = f"stopped at --max-steps {args.max_steps}"
                break

        if collapse.get("detected"):
            break

        epoch_loss = epoch_loss_sum / max(1, steps_this_epoch)
        epoch_seconds = time.perf_counter() - epoch_started
        recall_result = evaluate_recall(
            model, val_pairs, tokenizer, max_raw, seed=int(args.seed),
            n_pairs=int(args.eval_pairs), batch_size=int(args.eval_batch_size),
        )
        if geometry_samples:
            mean_alignment = alignment_sum / geometry_samples
            mean_uniformity = uniformity_sum / geometry_samples
            geometry_source = f"mean of {geometry_samples} per-step sample(s)"
        else:
            # A short run can finish before the first log interval, which would
            # otherwise report a spurious 0.0 for both metrics.  Fall back to the
            # fixed val sample, which is measured every epoch regardless.
            mean_alignment = recall_result["val_alignment"]
            mean_uniformity = recall_result["val_uniformity"]
            geometry_source = "fixed val sample (no log interval completed)"
        print(f"  epoch {epoch:>2d} | train_loss {epoch_loss:7.4f} | alignment "
              f"{mean_alignment:6.4f} | uniformity {mean_uniformity:7.4f} | Recall@1 "
              f"{recall_result['recall_at_1']:.2%} (full-pool "
              f"{recall_result['recall_at_1_full_pool']:.2%}) | {epoch_seconds / 60:6.1f} min",
              flush=True)
        print(f"    per-step geometry: {health_note(mean_alignment, mean_uniformity)}",
              flush=True)
        print(f"    val geometry: alignment {recall_result['val_alignment']:.4f}, uniformity "
              f"{recall_result['val_uniformity']:.4f} | MRR {recall_result['mrr_sampled']:.4f} | "
              f"random baseline {recall_result['random_baseline_recall_at_1']:.2%} | "
              f"{recall_result['seconds']:.0f}s for {recall_result['pairs_evaluated']:,} pairs",
              flush=True)
        print(f"    stream: {stream.pairs_read:,} pairs, {stream.sentences:,} sentences, "
              f"{stream.truncated:,} truncated to {max_raw} ids "
              f"({stream.truncated_percent:.2f}%)", flush=True)

        history.append({
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": round(epoch_loss, 4),
            "alignment": round(mean_alignment, 4),
            "uniformity": round(mean_uniformity, 4),
            "recall_at_1": recall_result["recall_at_1"],
            "recall_at_1_full_pool": recall_result["recall_at_1_full_pool"],
            "recall_at_5_full_pool": recall_result["recall_at_5_full_pool"],
            "mrr_sampled": recall_result["mrr_sampled"],
            "val_alignment": recall_result["val_alignment"],
            "val_uniformity": recall_result["val_uniformity"],
            "health": health_note(mean_alignment, mean_uniformity),
            "lr": optimizer.param_groups[0]["lr"],
            "steps": steps_this_epoch,
            "pairs": stream.pairs_read,
            "sentences": stream.sentences,
            "truncated_sentences": stream.truncated,
            "truncated_percent": round(stream.truncated_percent, 3),
            "epoch_seconds": round(epoch_seconds, 1),
            "elapsed_s": round(time.perf_counter() - started, 1),
            "eval": recall_result,
        })

        payload = {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": epoch_loss,
            "alignment": mean_alignment,
            "uniformity": mean_uniformity,
            "recall_at_1": recall_result["recall_at_1"],
            "recall_at_1_full_pool": recall_result["recall_at_1_full_pool"],
            "model_config": model_config,
            "train_config": dict(config.TRAIN),
            "objective": "next-sentence contrastive (NT-Xent, in-batch negatives)",
            "temperature": args.temperature,
            "hard_negative_penalty": args.hard_negative_penalty,
            "symmetric_loss": bool(args.symmetric),
            "latent_dim": LATENT_DIM,
            "tokenizer": str(config.path("tokenizer_wiki")),
            "seq_len": seq_len,
            "args": vars(args),
        }
        saved_epoch = Path("(nothing: --no-save)")
        removed: List[str] = []
        if not args.no_save:
            saved_epoch = save_checkpoint(epoch_checkpoint_path(epoch), payload)
            removed = prune_epoch_checkpoints()

        marker = ""
        if recall_result["recall_at_1"] > best.get("recall_at_1", -1.0):
            best = {
                "epoch": epoch,
                "recall_at_1": recall_result["recall_at_1"],
                "recall_at_1_full_pool": recall_result["recall_at_1_full_pool"],
                "alignment": round(mean_alignment, 4),
                "uniformity": round(mean_uniformity, 4),
            }
            if not args.no_save:
                save_checkpoint(config.path("gmn_wiki_ckpt"), payload)
                marker = "  <- best (also saved to gmn_wiki.pt)"
        print(f"    checkpoint {saved_epoch.name}  "
              f"(pruned: {', '.join(removed) if removed else 'nothing'}){marker}", flush=True)
        if not args.no_save:
            write_progress_snapshot(history, mid_epoch, best, stop_reason)

        run_truncated = bool(args.max_steps and global_step >= int(args.max_steps))
        if epoch == 1 and not run_truncated:
            recall = recall_result["recall_at_1"]
            early_gate = {
                "epoch": 1,
                "recall_at_1": recall,
                "random_baseline": recall_result["random_baseline_recall_at_1"],
                "continue_threshold": RECALL_CONTINUE,
                "worrying_threshold": RECALL_WORRYING,
                "alignment": round(mean_alignment, 4),
                "uniformity": round(mean_uniformity, 4),
            }
            if recall > RECALL_CONTINUE:
                early_gate["verdict"] = "continue to epoch 2"
                print(f"  gate (TASK 21R note 8): Recall@1 {recall:.2%} > "
                      f"{RECALL_CONTINUE:.0%} -> CONTINUE to epoch 2", flush=True)
            elif recall < RECALL_WORRYING:
                early_gate["verdict"] = "below 2% — barely above random"
                stop_reason = (
                    f"STOPPED after epoch 1: Recall@1 {recall:.2%} < {RECALL_WORRYING:.0%} "
                    f"(random baseline {recall_result['random_baseline_recall_at_1']:.2%}); "
                    f"geometry alignment {mean_alignment:.4f}, uniformity "
                    f"{mean_uniformity:.4f}"
                )
                print(f"  gate (TASK 21R note 8): {stop_reason}", flush=True)
                print(f"    collapse needs alignment > {COLLAPSE_ALIGNMENT} AND uniformity > "
                      f"{COLLAPSE_UNIFORMITY}; observed {mean_alignment:.4f} / "
                      f"{mean_uniformity:.4f} -> "
                      f"{health_note(mean_alignment, mean_uniformity)}", flush=True)
                break
            else:
                early_gate["verdict"] = "between 2% and 5% — continuing with a warning"
                print(f"  gate (TASK 21R note 8): Recall@1 {recall:.2%} is above random but "
                      f"below {RECALL_CONTINUE:.0%} -> continuing with a warning", flush=True)

        if run_truncated:
            break

    total_seconds = time.perf_counter() - started
    result: Dict[str, Any] = {
        "script": "train_gmn_wiki.py",
        "task": "TASK 21R",
        "objective": "next-sentence contrastive prediction (NT-Xent, in-batch negatives)",
        "model": args.model,
        "model_config": model_config,
        "params": model.count_parameters(),
        "latent_dim": LATENT_DIM,
        "temperature": args.temperature,
        "hard_negative_penalty": args.hard_negative_penalty,
        "symmetric_loss": bool(args.symmetric),
        "seq_len": seq_len,
        "max_raw_tokens": max_raw,
        "batch_size": int(args.batch_size),
        "pair_batch": int(args.batch_size),
        "encodes_per_step": 2 * int(args.batch_size),
        "shuffle_buffer": int(args.shuffle_buffer),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_steps": int(args.warmup_steps),
        "grad_clip": args.grad_clip,
        "seed": int(args.seed),
        "threads": threads,
        "torch": torch.__version__,
        "epochs_requested": int(args.epochs),
        "epochs_completed": len(history),
        "steps_per_epoch": steps_per_epoch,
        "total_steps_scheduled": total_steps,
        "global_step": global_step,
        "train_pairs": n_train_pairs,
        "val_pairs": len(val_pairs),
        "recall_protocol": {
            "pairs": EVAL_PAIRS,
            "negatives_per_anchor": EVAL_NEGATIVES,
            "random_baseline": round(1.0 / (1.0 + EVAL_NEGATIVES), 4),
        },
        "history": history,
        "mid_epoch": mid_epoch,
        "best": {
            "epoch": best.get("epoch", 0),
            "recall_at_1": best.get("recall_at_1"),
            "recall_at_1_full_pool": best.get("recall_at_1_full_pool"),
            "alignment": best.get("alignment"),
            "uniformity": best.get("uniformity"),
        },
        "collapse": collapse or {"detected": False},
        "early_gate": early_gate,
        "stopped_early": stop_reason != f"completed all {args.epochs} requested epoch(s)",
        "stop_reason": stop_reason,
        "total_seconds": round(total_seconds, 1),
        "total_minutes": round(total_seconds / 60, 1),
        "seconds_per_step": round(total_seconds / max(1, global_step), 4),
        "checkpoints": {
            "best": str(config.path("gmn_wiki_ckpt")),
            "epochs": sorted(
                path.name
                for path in config.path("gmn_wiki_ckpt").parent.glob("gmn_wiki_epoch*.pt")
            ),
        },
        "in_progress": False,
    }
    if not args.no_save:
        write_result(result)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--model", choices=sorted(config.MODEL_CONFIGS), default="small",
                        help="model config to train (DECISION 2: small)")
    parser.add_argument("--epochs", type=int, default=2,
                        help="passes over all 3.57M pairs (TASK 21R note 8: 2)")
    parser.add_argument("--batch-size", type=int, default=int(config.TRAIN["batch_size"]),
                        help="pairs per batch; every other pair is an in-batch negative")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help="NT-Xent temperature; drop to 0.05 after a collapse")
    parser.add_argument("--hard-negative-penalty", type=float,
                        default=DEFAULT_HARD_NEGATIVE_PENALTY,
                        help="constant added to each row's hardest-negative logit "
                             "(0 disables hard-negative mining)")
    parser.add_argument("--no-symmetric", dest="symmetric", action="store_false",
                        default=DEFAULT_SYMMETRIC,
                        help="use CE(S/T) only instead of the symmetric "
                             "0.5*(CE(S/T) + CE(S.T/T))")
    parser.add_argument("--lr", type=float, default=float(config.TRAIN["lr"]))
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=int(config.TRAIN["warmup_steps"]))
    parser.add_argument("--grad-clip", type=float, default=float(config.TRAIN["grad_clip"]))
    parser.add_argument("--seed", type=int, default=int(config.TRAIN["seed"]))
    parser.add_argument("--log-every", type=int, default=LOG_EVERY,
                        help=f"log cadence in steps (default {LOG_EVERY})")
    parser.add_argument("--recall-every", type=int, default=0,
                        help="mid-epoch Recall@1 cadence in steps (0 = per epoch only)")
    parser.add_argument("--eval-pairs", type=int, default=EVAL_PAIRS,
                        help=f"pairs used by the Recall@1 evaluation (default {EVAL_PAIRS})")
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--shuffle-buffer", type=int, default=SHUFFLE_BUFFER,
                        help=f"pairs held for shuffling (default {SHUFFLE_BUFFER:,})")
    parser.add_argument("--max-steps", type=int, default=None, help="stop after N steps")
    parser.add_argument("--limit-pairs", type=int, default=None,
                        help="smoke mode: only the first N pairs of the file")
    parser.add_argument("--threads", type=int, default=None, help="torch CPU thread count")
    parser.add_argument("--no-save", action="store_true",
                        help="do not write checkpoints or the history file")
    parser.add_argument("--smoke", action="store_true",
                        help="pipeline check: 20k pairs, 20 steps, small eval, nothing saved")
    args = parser.parse_args(argv)

    if args.smoke:
        args.limit_pairs = args.limit_pairs or 20_000
        args.max_steps = args.max_steps or 20
        args.epochs = 1
        args.recall_every = args.recall_every or 5
        args.eval_pairs = min(int(args.eval_pairs), 200)
        args.shuffle_buffer = min(int(args.shuffle_buffer), 4_000)
        args.no_save = True
    return args


def print_summary(result: Dict[str, Any]) -> None:
    """The DONE block's numbers, straight from the result record."""
    best_recall = result["best"]["recall_at_1"] or 0.0
    best_full = result["best"]["recall_at_1_full_pool"] or 0.0

    print("\n" + "=" * 74)
    print("SUMMARY — TASK 21R (GMN style, next-sentence contrastive)")
    print("=" * 74)
    print(f"  model                : {result['model']} "
          f"{result['model_config']['n_layers']} x {result['model_config']['hidden_dim']}, "
          f"ffn {result['model_config']['ffn_dim']}  ({result['params']:,} params)")
    print(f"  objective            : NT-Xent in-batch negatives, T={result['temperature']}, "
          f"{result['pair_batch']} pairs/step ({result['encodes_per_step']} encodes)")
    print(f"  loss                 : "
          f"{'symmetric 0.5*(CE(S/T)+CE(S.T/T))' if result.get('symmetric_loss') else 'CE(S/T)'}"
          f" + hardest-negative penalty {result.get('hard_negative_penalty', 0.0)}")
    print(f"  data                 : {result['train_pairs']:,} train pairs, "
          f"{result['val_pairs']:,} val pairs, shuffle buffer {result['shuffle_buffer']:,}")
    print(f"  schedule             : {result['steps_per_epoch']:,} steps/epoch x "
          f"{result['epochs_requested']} epoch(s), lr {result['lr']} -> cosine decay")
    print(f"  {'epoch':>5s} {'train_loss':>10s} {'align':>7s} {'uniform':>8s} {'R@1':>7s} "
          f"{'R@1full':>8s} {'MRR':>6s} {'minutes':>8s}")
    for row in result["history"]:
        print(f"  {row['epoch']:>5d} {row['train_loss']:>10.4f} {row['alignment']:>7.4f} "
              f"{row['uniformity']:>8.4f} {row['recall_at_1']:>7.2%} "
              f"{row['recall_at_1_full_pool']:>8.2%} {row['mrr_sampled']:>6.4f} "
              f"{row['epoch_seconds'] / 60:>8.1f}")
    recall_points = [entry for entry in result["mid_epoch"] if entry.get("kind") == "recall"]
    if recall_points:
        first, last = recall_points[0], recall_points[-1]
        print(f"  mid-epoch Recall@1   : first {first['recall_at_1']:.2%} "
              f"(step {first['global_step']:,}) -> last {last['recall_at_1']:.2%} "
              f"(step {last['global_step']:,})")
    geometry_points = [entry for entry in result["mid_epoch"] if entry.get("kind") == "geometry"]
    if geometry_points:
        first, last = geometry_points[0], geometry_points[-1]
        print(f"  mid-epoch geometry   : alignment {first['alignment']:.4f} -> "
              f"{last['alignment']:.4f}, uniformity {first['uniformity']:.4f} -> "
              f"{last['uniformity']:.4f}")
    print(f"  collapse detected    : {result['collapse'].get('detected', False)}")
    if result["collapse"].get("detected"):
        print(f"    restart with       : --temperature "
              f"{result['collapse']['suggested_temperature']}")
    if result["early_gate"]:
        gate = result["early_gate"]
        print(f"  epoch-1 gate         : Recall@1 {gate['recall_at_1']:.2%} vs random "
              f"{gate['random_baseline']:.2%} -> {gate['verdict']}")
    print(f"  best Recall@1        : {best_recall:.2%} at epoch {result['best']['epoch']} "
          f"(full-pool {best_full:.2%}) -> {result['checkpoints']['best']}")
    print(f"  Recall@1 > 20%       : {best_recall > 0.20}")
    print(f"  Recall@1 > 5%        : {best_recall > RECALL_CONTINUE}")
    print(f"  total training time  : {result['total_minutes']:.1f} min "
          f"({result['seconds_per_step']:.3f} s/step, {result['threads']} threads)")
    print(f"  stop reason          : {result['stop_reason']}")
    print(f"  checkpoints          : {', '.join(result['checkpoints']['epochs']) or 'none'}")
    print(f"\n  history file         : {config.path('gmn_wiki_history')}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_stdout()
    args = parse_args(argv)
    result = train(args)
    print_summary(result)
    print(f"\nDone in {result['total_seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
