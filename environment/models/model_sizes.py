"""Parameter counts for every model config, for both architectures (TASK 19R).

TASK 19R asks for the table

    Config    | GPT-2 params | GMN params
    SMALL     | XX.XM        | XX.XM
    MEDIUM    | XX.XM        | XX.XM

and for MODEL_MEDIUM to land near 22M.  This script prints that table from
*instantiated* models (the ground truth, including weight tying and biases),
cross-checks it against a closed-form formula, and reports the target.

Usage (from the repository root)::

    python environment/models/model_sizes.py
    python environment/models/model_sizes.py --config medium

Exits non-zero only if the analytic formula disagrees with the measured count or
the two architectures drift apart by more than 5% — a missed size *target* is
reported, not treated as a failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

MODELS_DIR = Path(__file__).resolve().parent
ENVIRONMENT_DIR = MODELS_DIR.parent
TOKENIZER_DIR = ENVIRONMENT_DIR / "tokenizer"
for _directory in (ENVIRONMENT_DIR, MODELS_DIR, TOKENIZER_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import config  # noqa: E402
from gmn_style import LATENT_DIM, GMNStyle  # noqa: E402
from gpt2_style import GPT2Style  # noqa: E402
from transformer import resolve_model_config  # noqa: E402

#: TASK 19R size targets, and how far off a measurement may be before it is
#: called out.  SMALL's target is the ~9M the task derives by hand; MEDIUM's is
#: the "~22M (matching GPT-2 small)" line, which DECISION 1 made reachable by
#: correcting the fields from 12 x 512 to 9 x 384.
TARGETS: Dict[str, int] = {"small": 9_000_000, "medium": 22_000_000}
TARGET_TOLERANCE = 0.10

#: Architectures must stay this close to each other (TASK 19R: same parameter
#: count, different objective only).
MAX_ARCHITECTURE_GAP = 5.0

#: Reference variants near the ~22M target, for the MEDIUM decision.  ``ffn`` is
#: 4x the width, matching MODEL_SMALL/MEDIUM's shape.  MODEL_MEDIUM itself is
#: 9 x 384 and appears in the main table; these are the other candidates.
MEDIUM_CANDIDATES: Tuple[Tuple[int, int], ...] = (
    (12, 320),
    (13, 320),
    (14, 320),
    (10, 384),
    (12, 384),
)


def block_parameters(hidden_dim: int, ffn_dim: int) -> int:
    """Unique parameters in one TransformerBlock.

    ``4 x (H^2 + H)`` for q/k/v/out projections with bias, the two FFN matrices
    with bias, and ``2 x 2H`` for the two LayerNorms.
    """
    attention = 4 * (hidden_dim * hidden_dim + hidden_dim)
    feed_forward = (hidden_dim * ffn_dim + ffn_dim) + (ffn_dim * hidden_dim + hidden_dim)
    layer_norms = 2 * 2 * hidden_dim
    return attention + feed_forward + layer_norms


def gpt2_parameters(config_dict: Optional[Mapping[str, Any]] = None) -> int:
    """Closed-form GPT-2 style parameter count (LM head tied to the embedding)."""
    cfg = resolve_model_config(config_dict)
    return (
        cfg["vocab_size"] * cfg["hidden_dim"]        # token embedding == LM head
        + cfg["max_seq_len"] * cfg["hidden_dim"]     # learned positional table
        + cfg["n_layers"] * block_parameters(cfg["hidden_dim"], cfg["ffn_dim"])
        + 2 * cfg["hidden_dim"]                      # final LayerNorm
    )


def gmn_parameters(
    config_dict: Optional[Mapping[str, Any]] = None,
    latent_dim: int = LATENT_DIM,
) -> int:
    """Closed-form GMN style parameter count (same trunk + 256-d projection)."""
    cfg = resolve_model_config(config_dict)
    projection = cfg["hidden_dim"] * latent_dim + latent_dim
    return gpt2_parameters(cfg) + projection


def measure(config_dict: Mapping[str, Any]) -> Dict[str, int]:
    """Instantiate both models and return their unique parameter counts."""
    gpt2 = GPT2Style(config_dict=config_dict)
    gmn = GMNStyle(config_dict=config_dict)
    measured = {"gpt2": gpt2.count_parameters(), "gmn": gmn.count_parameters()}
    del gpt2, gmn
    return measured


def target_verdict(label: str, params: int) -> Tuple[bool, str]:
    target = TARGETS[label]
    delta = 100.0 * (params - target) / target
    return abs(delta) <= 100 * TARGET_TOLERANCE, f"{delta:+.1f}%"


def report_main_table(labels: Sequence[str]) -> Tuple[bool, Dict[str, Dict[str, int]]]:
    """Print the TASK 19R table and cross-check the formula against it."""
    print(f"  {'config':<7s} {'layers':>6s} {'heads':>5s} {'hidden':>6s} {'ffn':>5s} "
          f"{'vocab':>6s} {'seq':>4s} {'GPT-2':>12s} {'GMN':>12s} {'diff':>11s}")
    measured_all: Dict[str, Dict[str, int]] = {}
    formula_ok = True
    gap_ok = True

    for label in labels:
        cfg = config.model_config(label)
        measured = measure(cfg)
        measured_all[label] = measured
        difference = measured["gmn"] - measured["gpt2"]
        gap = 100.0 * abs(difference) / measured["gpt2"]

        print(f"  {label.upper():<7s} {cfg['n_layers']:>6d} {cfg['n_heads']:>5d} "
              f"{cfg['hidden_dim']:>6d} {cfg['ffn_dim']:>5d} {cfg['vocab_size']:>6d} "
              f"{cfg['max_seq_len']:>4d} {measured['gpt2']:>12,} {measured['gmn']:>12,} "
              f"{difference:>+8,} ({gap:.2f}%)")

        formula_gpt2 = gpt2_parameters(cfg)
        formula_gmn = gmn_parameters(cfg)
        same = formula_gpt2 == measured["gpt2"] and formula_gmn == measured["gmn"]
        formula_ok = formula_ok and same
        gap_ok = gap_ok and gap < MAX_ARCHITECTURE_GAP
        print(f"          formula: GPT-2 {formula_gpt2:,} / GMN {formula_gmn:,} "
              f"-> {'matches measured' if same else 'MISMATCH'}")
    return formula_ok and gap_ok, measured_all


def report_targets(measured_all: Dict[str, Dict[str, int]]) -> None:
    """TASK 19R's size targets, judged on the measured counts."""
    print("\nTarget check")
    print("-" * 74)
    for label, measured in measured_all.items():
        params = measured["gpt2"]
        ok, delta_text = target_verdict(label, params)
        print(f"  {label.upper():<7s} target ~{TARGETS[label]:>11,} | "
              f"measured {params:>12,} (GPT-2) -> "
              f"{'MET' if ok else 'NOT MET'} ({delta_text})")

    print()
    for label, measured in measured_all.items():
        cfg = config.model_config(label)
        embedding = cfg["vocab_size"] * cfg["hidden_dim"]
        positions = cfg["max_seq_len"] * cfg["hidden_dim"]
        per_block = block_parameters(cfg["hidden_dim"], cfg["ffn_dim"])
        final_ln = 2 * cfg["hidden_dim"]
        blocks = cfg["n_layers"] * per_block
        print(f"  {label.upper()} decomposition ({cfg['n_layers']} x {cfg['hidden_dim']}, "
              f"ffn {cfg['ffn_dim']}, vocab {cfg['vocab_size']}, seq {cfg['max_seq_len']}):")
        print(f"    token embedding (tied LM head) : {embedding:>12,} "
              f"= {cfg['vocab_size']} x {cfg['hidden_dim']}")
        print(f"    positional embedding           : {positions:>12,} "
              f"= {cfg['max_seq_len']} x {cfg['hidden_dim']}")
        print(f"    transformer blocks             : {blocks:>12,} "
              f"= {cfg['n_layers']} x {per_block:,}")
        print(f"    final LayerNorm                : {final_ln:>12,}")
        print(f"    sum (GPT-2)                    : {measured['gpt2']:>12,}   "
              f"GMN = + {cfg['hidden_dim'] * LATENT_DIM + LATENT_DIM:,} "
              f"= {measured['gmn']:,}")
        if not target_verdict(label, measured["gpt2"])[0]:
            print(f"    -> ~{TARGETS[label]:,} is UNREACHABLE with these fields: the "
                  f"blocks alone are {blocks / 1e6:.1f}M.  See the reference "
                  "variants below.")

    print("\n  reference variants near 22M (analytic, not implemented; ffn = 4 x width):")
    for layers, hidden in MEDIUM_CANDIDATES:
        candidate = dict(config.model_config("medium"), n_layers=layers,
                         hidden_dim=hidden, ffn_dim=4 * hidden)
        params = gpt2_parameters(candidate)
        print(f"    {layers:>2d} layers x {hidden:>3d} (ffn {4 * hidden:>4d}) : "
              f"GPT-2 {params:>12,}  GMN {gmn_parameters(candidate):>12,}  "
              f"({100.0 * (params - TARGETS['medium']) / TARGETS['medium']:+.1f}%)")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--config", dest="configs", action="append", choices=sorted(config.MODEL_CONFIGS),
        help="config to report (repeatable; default: all)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    labels: List[str] = args.configs or sorted(config.MODEL_CONFIGS)

    print("=" * 74)
    print("TASK 19R - parameter counts per model config")
    print("=" * 74)
    print(f"latent dimension (GMN projection) : {LATENT_DIM}")
    print(f"pooling                           : [CLS] at position 0\n")

    ok, measured_all = report_main_table(labels)
    report_targets(measured_all)

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    for label, measured in measured_all.items():
        print(f"  {label.upper():<7s} GPT-2 {measured['gpt2']:>12,} ({measured['gpt2'] / 1e6:.2f}M)   "
              f"GMN {measured['gmn']:>12,} ({measured['gmn'] / 1e6:.2f}M)")
    print(f"\n  formula == measured, and architectures within "
          f"{MAX_ARCHITECTURE_GAP:.0f}% : {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
