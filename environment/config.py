"""Central configuration for the V3 controlled experiment (TASK 18).

Everything either model needs lives in this one file, and **both models read
the exact same values**.  The V3 experiment is only meaningful if GPT-2 style
and GMN style see identical data with identical parameter budgets, so
architecture and training objective stay the only variables.

``MODEL`` / ``TRAIN`` / ``GMN`` / ``PATHS`` are the frozen contract from
``TASKS_V3.md`` and must not be edited per-model.  The helpers underneath exist
only so scripts can resolve the relative ``PATHS`` entries no matter which
directory they were launched from.

This module deliberately imports nothing but the standard library, so it can be
imported by data prep, tokenizer, training and eval code alike.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Union

# ---------------------------------------------------------------------------
# ALL hyperparameters in one place
# Both models use identical values
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Model configs (TASK 19R)
# ---------------------------------------------------------------------------
# Two sizes for the WikiText-103 experiment.  BOTH models are built from the
# same dict, so architecture (causal vs bidirectional) and the training
# objective stay the only variables.
#
#   MODEL_SMALL    fast pipeline verification (smoke runs of TASK 20R/21R)
#   MODEL_MEDIUM   the real experiment
#
# `vocab_size = 16000` and `max_seq_len = 64` are properties of the TASK 18R
# WikiText artifacts (bpe_wiki_vocab.json, sentence-level sequences), not free
# knobs: a config that disagrees with them cannot be trained on this data.
#
# Measured unique parameter counts (see environment/models/model_sizes.py):
#
#   config   GPT-2        GMN          block      vs ~22M target
#   SMALL    8,851,456    8,917,248     789,760    (+/- its own ~9M check)
#   MEDIUM  22,139,520   22,238,080   1,774,464    +0.6%  -> MET
#
# MEDIUM's fields were corrected by DECISION 1: TASK 19R originally specified
# 12 x 512 with ffn 2048 for "~22M", which measures 46,054,400 (the embedding
# alone is 16000 x 512 = 8.2M and the 12 blocks are 37.8M), so the target was
# unreachable with those fields.  9 x 384 with ffn 1536 measures 22,139,520 —
# within 0.6% of 22M while keeping vocab 16000 / seq_len 64 (the TASK 18R
# contract) and an 8-head, 4x-ffn shape.  Verified by instantiating both models.

MODEL_SMALL = dict(
    n_layers    = 6,
    n_heads     = 8,
    hidden_dim  = 256,
    ffn_dim     = 1024,
    vocab_size  = 16000,
    max_seq_len = 64,
    dropout     = 0.1,
)

MODEL_MEDIUM = dict(
    n_layers    = 9,
    n_heads     = 8,
    hidden_dim  = 384,
    ffn_dim     = 1536,
    vocab_size  = 16000,
    max_seq_len = 64,
    dropout     = 0.1,
)

#: Named lookup, so scripts can take ``--model small|medium``.
MODEL_CONFIGS: Dict[str, Dict[str, int]] = {
    "small": MODEL_SMALL,
    "medium": MODEL_MEDIUM,
}


def model_config(name: Union[str, Mapping[str, Any]] = "small") -> Dict[str, Any]:
    """Return a copy of a model config dict, by name or by dict.

    ``model_config("medium")`` / ``model_config(config.MODEL_MEDIUM)`` both work;
    passing a dict copies it, so a caller can tweak one field without mutating
    the shared config.
    """
    if isinstance(name, Mapping):
        return dict(name)
    key = str(name).strip().lower()
    if key not in MODEL_CONFIGS:
        raise KeyError(f"unknown model config {name!r}; known: {sorted(MODEL_CONFIGS)}")
    return dict(MODEL_CONFIGS[key])


#: Default size for anything that does not take a ``--model`` flag.  This is the
#: WikiText contract: ARC-era scripts (TASKs 18-24) are superseded, so `MODEL`
#: follows the experiment that is actually running.
MODEL = MODEL_SMALL


# TRAIN values below are the WikiText-103 pretraining schedule (TASK 18R): a
# much larger corpus than ARC needs a lower LR, a longer warmup and a larger
# batch (in-batch negatives for the GMN objective), and `seq_len` is
# sentence-level rather than document-level.  `seed` is kept because the data
# loader and the model tests consume it.
TRAIN = dict(
    batch_size   = 64,
    lr           = 3e-4,
    epochs       = 10,
    warmup_steps = 1000,
    grad_clip    = 1.0,
    seed         = 42,
    seq_len      = 64,
)

GMN = dict(
    temperature  = 0.07,
    lambda_nav   = 0.0,     # start with no navigation
    top_k        = 10,
)

# ---------------------------------------------------------------------------
# WikiText-103 artifacts (TASK 18R) used by the WikiText trainings (20R/21R)
# ---------------------------------------------------------------------------
WIKITEXT = dict(
    train_sentences = "environment/data/wikitext/train_sentences.json",
    train_pairs     = "environment/data/wikitext/train_pairs.json",
    val_sentences   = "environment/data/wikitext/val_sentences.json",
    val_pairs       = "environment/data/wikitext/val_pairs.json",
    # Token-stream caches built by env/training/train_gpt2_wiki.py (TASK 20R).
    # uint16 token ids, 2 bytes each, memory-mapped at train time.
    train_tokens    = "environment/data/wikitext/train_tokens.bin",
    val_tokens      = "environment/data/wikitext/val_tokens.bin",
    tokens_meta     = "environment/data/wikitext/tokens_meta.json",
    # --- directional encoder (structural relation labels over paragraph boundaries) ---
    directional_pairs = "environment/data/wikitext/directional_pairs.json",
)
PATHS = dict(
    arc_train    = "environment/data/arc/train.json",
    arc_val      = "environment/data/arc/val.json",
    arc_test     = "environment/data/arc/test.json",
    conceptnet   = "environment/data/conceptnet/triples.json",
    tokenizer    = "environment/tokenizer/bpe_vocab.json",
    gpt2_ckpt    = "environment/checkpoints/gpt2_style.pt",
    gmn_ckpt     = "environment/checkpoints/gmn_style.pt",
    results      = "environment/results/comparison.json",
    # --- WikiText-103 pretraining (TASK 18R onwards) ---
    tokenizer_wiki      = "environment/tokenizer/bpe_wiki_vocab.json",
    gpt2_wiki_ckpt      = "environment/checkpoints/gpt2_wiki.pt",
    gmn_wiki_ckpt       = "environment/checkpoints/gmn_wiki.pt",
    gpt2_wiki_history   = "environment/results/gpt2_wiki_history.json",
    gmn_wiki_history    = "environment/results/gmn_wiki_history.json",
    # --- bidirectional LM arm (LLM vs LLM, NTP objective) ---
    gmn_lm_ckpt         = "environment/checkpoints/gmn_lm_wiki.pt",
    gmn_lm_history      = "environment/results/gmn_lm_history.json",
    wiki_geometry_png   = "environment/results/wiki_geometry.png",
    geometry_ckpt       = "environment/results/geometry_verification.json",
    hellaswag_results   = "environment/results/hellaswag_results.json",
)

# Register the four WIKITEXT artifacts as ``wiki_<name>`` so every entry point
# can reach them through :func:`path` like any other artifact.
PATHS.update({f"wiki_{name}": value for name, value in WIKITEXT.items()})


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------
# ``PATHS`` entries are relative to the repository root (not to whatever cwd a
# script happens to be launched from), so every entry point goes through
# :func:`path` instead of touching ``PATHS`` directly.

#: Repository root — ``<repo>/environment/config.py`` -> ``<repo>``.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

#: Absolute path of the ``environment/`` package directory.
ENVIRONMENT_ROOT: Path = Path(__file__).resolve().parent


def path(name: str) -> Path:
    """Return the absolute :class:`Path` registered under ``PATHS[name]``.

    Absolute entries are returned untouched; relative entries are resolved
    against :data:`PROJECT_ROOT`.
    """
    if name not in PATHS:
        raise KeyError(f"unknown path {name!r}; known keys: {sorted(PATHS)}")
    candidate = Path(PATHS[name])
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def ensure_dirs() -> Dict[str, Path]:
    """Create the parent directory of every :data:`PATHS` entry.

    Safe to call repeatedly.  Returns ``{path key: parent directory}`` so
    callers can log where things went.
    """
    created: Dict[str, Path] = {}
    for name in PATHS:
        parent = path(name).parent
        parent.mkdir(parents=True, exist_ok=True)
        created[name] = parent
    return created


if __name__ == "__main__":  # tiny smoke test / self-documentation
    print(f"PROJECT_ROOT     : {PROJECT_ROOT}")
    print(f"ENVIRONMENT_ROOT : {ENVIRONMENT_ROOT}")
    print("\nResolved PATHS:")
    for key in PATHS:
        print(f"  {key:<12s} -> {path(key)}")
    print("\nMODEL :", MODEL)
    print("TRAIN :", TRAIN)
    print("GMN   :", GMN)
