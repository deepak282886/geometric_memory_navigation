"""STEP 0 result: which encoder makes the memory sweep monotone - and by how much?

TASK 27's finding was that the *geometry* was the bottleneck, not the memory: with
the original VAE the controlled sweep (fixed 1,024 examples, growing memory)
started at cos 0.5903 with no memory, dropped to 0.5837 at 256 memories (t -8.78)
and only rose to 0.5984 at 38K memories (t +10.23) - relevance mattered, size did
not, and small/irrelevant memory actively hurt.

This script reads the TASK 27 histories of two or more encoders and prints the
comparison the task asks for:

    Encoder     | Memory 0 | Memory 256 | Memory 38K | trend

Trend statistics per encoder (computed from the sweep's own numbers):

    non_increasing   loss falls (or ties) at every step of the sweep
    memoryless_wins  how many sweep sizes beat the ``C = 0`` reference, paired
    spearman_rho     rank correlation of memory size vs loss (negative = bigger
                     memory goes with a lower loss)
    first_to_last    loss/cos at the smallest vs the largest memory

Artifacts
---------
    experiments/predictor_encoder_comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_HISTORIES = [
    f"Original (vae.pt)={PROJECT_ROOT / 'experiments' / 'predictor_history.json'}",
    f"Continuation (vae_continuation.pt)="
    f"{PROJECT_ROOT / 'experiments' / 'predictor_history_continuation.json'}",
]
OUTPUT_PATH = PROJECT_ROOT / "experiments" / "predictor_encoder_comparison.json"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare TASK 27 memory sweeps across encoders."
    )
    parser.add_argument(
        "--histories",
        nargs="+",
        default=DEFAULT_HISTORIES,
        help="LABEL=path entries pointing at predictor_history*.json",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args(argv)


def parse_histories(entries: Sequence[str]) -> List[Tuple[str, Path]]:
    parsed: List[Tuple[str, Path]] = []
    for entry in entries:
        if "=" in entry:
            label, raw = entry.split("=", 1)
        else:
            raw, label = entry, Path(entry).stem
        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        parsed.append((label.strip(), path))
    return parsed


def spearman_rho(first: Sequence[float], second: Sequence[float]) -> float:
    """Rank correlation of two sequences (no scipy needed for a trend flag)."""
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.size < 2:
        return 0.0
    rank_a = np.argsort(np.argsort(a)).astype(np.float64)
    rank_b = np.argsort(np.argsort(b)).astype(np.float64)
    rank_a = rank_a - rank_a.mean()
    rank_b = rank_b - rank_b.mean()
    denominator = float(np.sqrt((rank_a**2).sum() * (rank_b**2).sum()))
    return float((rank_a * rank_b).sum() / denominator) if denominator > 0 else 0.0


def sweep_analysis(history: Dict[str, Any]) -> Dict[str, Any]:
    """Trend statistics for one encoder's controlled memory-size sweep."""
    sweep = history["with_memory"]["memory_size_sweep"]
    sizes = [int(entry["memory_size"]) for entry in sweep]
    losses = [float(entry["mean_loss"]) for entry in sweep]
    cosines = [float(entry["mean_cos"]) for entry in sweep]
    deltas = [
        (
            None
            if entry.get("paired_vs_memoryless") is None
            else float(entry["paired_vs_memoryless"]["mean_delta"])
        )
        for entry in sweep
    ]
    t_stats = [
        (
            None
            if entry.get("paired_vs_memoryless") is None
            else float(entry["paired_vs_memoryless"]["paired_t_statistic"])
        )
        for entry in sweep
    ]
    steps = np.diff(np.asarray(losses, dtype=np.float64))
    return {
        "sizes": sizes,
        "losses": losses,
        "cosines": cosines,
        "paired_delta_vs_memoryless": deltas,
        "paired_t_vs_memoryless": t_stats,
        "non_increasing": bool((steps <= 0).all()) if steps.size else False,
        "strictly_decreasing": bool((steps < 0).all()) if steps.size else False,
        "monotone_steps": int((steps <= 0).sum()),
        "num_steps": int(steps.size),
        "spearman_rho_size_vs_loss": spearman_rho(sizes, losses),
        "memoryless_wins": int(
            sum(1 for value in deltas if value is not None and value > 0)
        ),
        "memoryless_comparisons": int(sum(1 for value in deltas if value is not None)),
        "first_to_last_loss": (losses[-1] - losses[0]) if losses else 0.0,
        "first_to_last_cos": (cosines[-1] - cosines[0]) if cosines else 0.0,
    }


def value_at(sweep: Dict[str, Any], size: int, key: str) -> Optional[float]:
    """Sweep value for an exact memory size (None when that size was not run)."""
    if size not in sweep["sizes"]:
        return None
    return sweep[key][sweep["sizes"].index(size)]


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    specs = parse_histories(args.histories)
    for label, path in specs:
        if not path.exists():
            raise FileNotFoundError(f"{label}: {path} does not exist (run TASK 27 first)")

    print("=" * 100)
    print("STEP 0 RESULT - memory sweep by encoder (same 1,024 fixed examples, growing memory)")
    print("=" * 100)

    encoders: Dict[str, Any] = {}
    for label, path in specs:
        with path.open("r", encoding="utf-8") as handle:
            history = json.load(handle)
        encoders[label] = {
            "history": str(path),
            "encoder_checkpoint": history["config"]["vae_checkpoint"],
            "sweep": sweep_analysis(history),
            "passes": [
                {
                    "pass": record["pass"],
                    "train_loss": record["train"]["mean_loss"],
                    "eval_loss": record["eval"]["mean_loss"],
                    "eval_cos": record["eval"]["mean_cos"],
                    "top1_recall": record["eval"].get("top1_recall"),
                    "top1_recall_excl_self": record["eval"].get("top1_recall_excl_self"),
                    "memory_size": record["eval"]["memory_size"],
                }
                for record in history["with_memory"]["passes"]
            ],
            "zero_context_cos": history["comparison"]["zero_context_same_predictor"]["mean_cos"],
            "paired_vs_zero_context": history["comparison"]["paired_memory_vs_zero_context"],
            "no_memory_predictor_cos": (
                None
                if history["comparison"].get("no_memory_baseline_predictor") is None
                else history["comparison"]["no_memory_baseline_predictor"]["mean_cos"]
            ),
            "num_pairs": history["data"]["num_pairs"],
            "sweep_sample": history["comparison"]["memory_size_sweep"]["sample_size"],
        }

    largest = max(max(entry["sweep"]["sizes"]) for entry in encoders.values())

    print("\n  the table the task asks for: sweep cos, higher is better")
    print(
        f"  {'encoder':<48} | Memory 0 | Memory 256 | Memory (full) | trend"
    )
    print("  " + "-" * 112)
    for label, entry in encoders.items():
        sweep = entry["sweep"]
        base = value_at(sweep, 0, "cosines")
        small = value_at(sweep, 256, "cosines")
        # every arm is read at its OWN largest memory (arm-specific full memory)
        full_size = sweep["sizes"][-1] if sweep["sizes"] else 0
        full = value_at(sweep, full_size, "cosines")
        if sweep["non_increasing"]:
            trend = "monotone non-increasing"
        elif sweep["spearman_rho_size_vs_loss"] < -0.4:
            trend = f"improving trend (rho {sweep['spearman_rho_size_vs_loss']:+.2f})"
        else:
            trend = f"non-monotone (rho {sweep['spearman_rho_size_vs_loss']:+.2f})"
        small_text = "n/a" if small is None else f"{small:.4f}"
        if small is not None and base is not None:
            small_text += "(-)" if small < base else "(+)"
        full_text = "n/a" if full is None else f"{full:.4f} @ {full_size}"
        if full is not None and base is not None:
            full_text += "(+)" if full > base else "(-)"
        print(
            f"  {label:<48} | {'n/a' if base is None else f'{base:.4f}':>8} | "
            f"{small_text:>10} | {full_text:>13} | {trend}"
        )
    print("  " + "-" * 112)

    print("\n  sweep detail (memory size | loss | cos | paired delta vs C=0 | t)")
    for label, entry in encoders.items():
        sweep = entry["sweep"]
        print(f"    {label}")
        for index, size in enumerate(sweep["sizes"]):
            delta = sweep["paired_delta_vs_memoryless"][index]
            t_stat = sweep["paired_t_vs_memoryless"][index]
            tail = (
                "(reference)"
                if delta is None or t_stat is None
                else f"{delta:+.4f} | {t_stat:+.2f}"
            )
            print(
                f"      {size:>6} | {sweep['losses'][index]:.4f} | "
                f"{sweep['cosines'][index]:.4f} | {tail}"
            )


    print("\n  per-pass frozen evaluation (eval loss / eval cos) and end state")
    for label, entry in encoders.items():
        passes = entry["passes"]
        print(f"    {label}")
        print(
            "      "
            + "  ".join(
                f"p{record['pass']}:{record['eval_loss']:.4f}/{record['eval_cos']:.4f}"
                for record in passes
            )
        )
        print(
            f"      eval cos {passes[0]['eval_cos']:.4f} -> {passes[-1]['eval_cos']:.4f} "
            f"| memory {passes[-1]['memory_size']} "
            f"| top1 recall (excl. self) {passes[-1]['top1_recall_excl_self']} "
            f"| C=0 same-model cos {entry['zero_context_cos']:.4f} "
            f"| no-memory-model cos {entry['no_memory_predictor_cos']}"
        )

    print("\n  VERDICT")
    for label, entry in encoders.items():
        sweep = entry["sweep"]
        paired = entry["paired_vs_zero_context"]
        print(
            f"    {label}: sweep non-increasing = {sweep['non_increasing']} "
            f"({sweep['monotone_steps']}/{sweep['num_steps']} steps), "
            f"{sweep['memoryless_wins']}/{sweep['memoryless_comparisons']} sizes beat C=0, "
            f"first->last cos {sweep['first_to_last_cos']:+.4f}, "
            f"rho(size, loss) {sweep['spearman_rho_size_vs_loss']:+.2f}, "
            f"final paired delta vs C=0 {paired['mean_delta']:+.4f} "
            f"(t {paired['paired_t_statistic']:+.2f})"
        )
    monotone = [
        label for label, entry in encoders.items() if entry["sweep"]["non_increasing"]
    ]
    # Recommendation rule (arm selection only, no new numbers): prefer a monotone
    # sweep; otherwise the strongest size trend (most negative rho), tie-broken by
    # the paired delta against C = 0. Only arms measured on the same number of
    # pairs are comparable, so the held-out control arm is reported separately
    # instead of being ranked against the full-pair arms.
    main_pairs = max(entry["num_pairs"] for entry in encoders.values())
    comparable = [
        label for label, entry in encoders.items() if entry["num_pairs"] == main_pairs
    ]
    other = [label for label in encoders if label not in comparable]

    def rank_key(label: str) -> Tuple[int, float, float]:
        sweep = encoders[label]["sweep"]
        return (
            0 if sweep["non_increasing"] else 1,
            sweep["spearman_rho_size_vs_loss"],
            -encoders[label]["paired_vs_zero_context"]["paired_t_statistic"],
        )

    recommended = min(comparable, key=rank_key) if comparable else None
    print(
        f"    monotone encoders: {monotone if monotone else 'none'} "
        f"-> TASK 28 recommendation (comparable arms only, "
        f"{main_pairs}-pair runs): {recommended}"
    )
    if other:
        print(
            "    not ranked (different example set, "
            f"{encoders[other[0]]['num_pairs']} pairs): {other}"
        )
    if recommended is not None:
        best = encoders[recommended]
        print(
            f"      {recommended}: sweep level {best['sweep']['cosines'][0]:.4f} "
            f"(no memory) -> {best['sweep']['cosines'][-1]:.4f} (full memory), "
            f"rho {best['sweep']['spearman_rho_size_vs_loss']:+.2f}, "
            f"paired delta vs C=0 {best['paired_vs_zero_context']['mean_delta']:+.4f} "
            f"(t {best['paired_vs_zero_context']['paired_t_statistic']:+.2f})"
        )

    payload = {
        "task": "STEP 0 result - memory sweep comparison across encoders",
        "largest_memory_size": largest,
        "encoders": encoders,
        "verdict": {
            "monotone_encoders": monotone,
            "comparable_arms": comparable,
            "non_comparable_arms": other,
            "recommended_for_task_28": recommended,
            "rule": (
                "prefer a monotone sweep, else the most negative rho(size, loss), "
                "tie-broken by the paired t-statistic against C = 0; only arms on the "
                "same pair count are ranked"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nsaved {args.output}")
    print("=" * 100)
    return payload


if __name__ == "__main__":
    main()

