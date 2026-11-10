"""TASK 30: threshold sweep for the retrieval-confidence gate.

The gate (``src/sampler/jev.py::retrieval_confidence``) trusts the memory context only
when the mean cosine of the top-3 retrieved neighbours clears a threshold; otherwise
that query falls back to ``C = zeros``. TASK 29 measured why this is worth trying: the
memory helps in-corpus (+0.0039) and hurts out-of-corpus (-0.0049), and the harm is
detectable per query (corr(context alignment, delta) = +0.569).

Three arms are scored at every checkpoint by ``run_loop_core``:

    ARM A  gated memory      (the gate under test)
    ARM B  always memory     (TASK 29's behaviour)
    ARM C  never memory      (C = zeros; the baseline)

What this script adds on top of ``run_prediction_loop.py --gate-threshold``:

1. **Calibration.** The threshold's scale depends on how many neighbours are averaged.
   TASK 29's quoted in-corpus/out-of-corpus separation (0.7148 vs 0.6704) is a TOP-10
   mean; the gate averages the TOP-3, which sits higher for the same query, so the
   task's 0.5 is not on the same scale as its own rationale. The script therefore
   measures the actual distribution of the gate statistic for in-corpus (train) and
   out-of-corpus (eval) queries and prints the percentiles, so the sweep can be read
   against where the two distributions actually overlap. ``--auto-extend`` adds
   thresholds inside that overlap on top of the requested list.

2. **Oracle upper bounds**, on the same 500 eval pairs and the same final memory:
     * ``oracle_pick_best_per_query`` - per query, take whichever of (memory, no memory)
       is better. This is the best ANY gate can achieve, so it bounds the whole idea.
     * ``oracle_alignment_gate`` - use memory only when its context points at the true
       next position (cos(C, true tangent) >= 0). A gate that can see the answer; not
       deployable, but it says how much of the harm a better *statistic* could remove.

Artifacts
---------
    experiments/gated_loop_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for path in (str(PROJECT_ROOT), str(SCRIPTS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from diagnose_loop_relevance import tangent_direction  # noqa: E402
from run_prediction_loop import (  # noqa: E402
    AUTO_FIT_AFTER,
    DEFAULT_PREDICTOR,
    DEFAULT_VAE,
    EVAL_EVERY,
    EVAL_SIZE,
    EVAL_SOURCE_BASE,
    EVAL_START,
    GATE_TOP_N,
    MAX_SIZE,
    NUM_REGIONS,
    PREDICT_BATCH_SIZE,
    PROBE_REGIONS,
    SCORE_MODE,
    STEPS,
    TRAIN_PAIRS_PATH,
    TRAIN_VEC_CACHE,
    VAL_PAIRS_PATH,
    VAL_VEC_CACHE,
    contexts_for_arm_a,
    cosine_to_actual,
    encode_sentences_cached,
    evaluate_arms,
    fuse_context,
    load_pairs,
    load_vae,
    paired_stats,
    predict,
    run_loop_core,
    stream_pairs,
    unique_sentences,
)
from src.memory.region_store import RegionStore  # noqa: E402
from src.predictor.latent_predictor import LATENT_DIM, load_latent_predictor  # noqa: E402
from src.sampler.jev import retrieval_confidence  # noqa: E402

RESULTS_PATH = PROJECT_ROOT / "experiments" / "gated_loop_results.json"
#: the thresholds the task asks for
DEFAULT_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.65, 0.7)
CALIBRATION_SAMPLE = 500
SEED = 42


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TASK 30: sweep the retrieval-confidence gate threshold."
    )
    parser.add_argument("--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS))
    parser.add_argument(
        "--auto-extend",
        type=int,
        default=4,
        help="extra thresholds taken from the calibration overlap (0 disables)",
    )
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--eval-size", type=int, default=EVAL_SIZE)
    parser.add_argument("--eval-start", type=int, default=EVAL_START)
    parser.add_argument("--num-regions", type=int, default=NUM_REGIONS)
    parser.add_argument("--probe-regions", type=int, default=PROBE_REGIONS)
    parser.add_argument("--max-size", type=int, default=MAX_SIZE)
    parser.add_argument("--auto-fit-after", type=int, default=AUTO_FIT_AFTER)
    parser.add_argument("--gate-top-n", type=int, default=GATE_TOP_N)
    parser.add_argument("--vae", type=Path, default=DEFAULT_VAE)
    parser.add_argument("--predictor", type=Path, default=DEFAULT_PREDICTOR)
    parser.add_argument("--train-pairs", type=Path, default=TRAIN_PAIRS_PATH)
    parser.add_argument("--val-pairs", type=Path, default=VAL_PAIRS_PATH)
    parser.add_argument("--train-cache", type=Path, default=TRAIN_VEC_CACHE)
    parser.add_argument("--val-cache", type=Path, default=VAL_VEC_CACHE)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args(argv)


def build_memory(args: argparse.Namespace) -> RegionStore:
    """A fresh RegionStore with the loop's configuration (one per threshold pass)."""
    return RegionStore(
        num_regions=args.num_regions,
        latent_dim=LATENT_DIM,
        max_size=args.max_size,
        auto_fit_after=args.auto_fit_after,
        score_mode=SCORE_MODE,
        seed=args.seed,
    )


def fill_memory(
    memory: RegionStore,
    train_current: np.ndarray,
    train_actual: np.ndarray,
    outcome: float = 0.5,
) -> None:
    """Feed every experience in, in order (used by the calibration memory)."""
    for step in range(len(train_current)):
        memory.add(
            train_current[step],
            outcome,
            target_vector=train_actual[step],
            source_index=step,
        )


def gate_statistic(
    memory: RegionStore,
    current: np.ndarray,
    sources: np.ndarray,
    probe_regions: int,
    top_n: int,
) -> np.ndarray:
    """The gate's own statistic per query (mean top-n neighbour cosine)."""
    values = np.zeros(len(current), dtype=np.float64)
    for index in range(len(current)):
        records = memory.query(
            current[index], k=10, probe_regions=probe_regions, exclude_source=int(sources[index])
        )
        _, confidence = retrieval_confidence(current[index], records, threshold=0.0, top_n=top_n)
        values[index] = confidence
    return values


def percentiles(values: np.ndarray) -> Dict[str, float]:
    """Compact distribution summary used for the calibration table and the JSON."""
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
    }



def calibrate(
    args: argparse.Namespace,
    memory: RegionStore,
    predictor,
    train_current: np.ndarray,
    train_actual: np.ndarray,
    eval_current: np.ndarray,
    eval_actual: np.ndarray,
    eval_sources: np.ndarray,
    train_sources: np.ndarray,
) -> Dict[str, Any]:
    """Distribution of the gate statistic, and the oracle bounds on the eval pairs."""
    print("\ncalibration: where does the gate statistic actually sit?")
    eval_stat = gate_statistic(
        memory, eval_current, eval_sources, args.probe_regions, args.gate_top_n
    )
    train_stat = gate_statistic(
        memory, train_current, train_sources, args.probe_regions, args.gate_top_n
    )

    print("  group                 | mean   | median | p10    | p25    | p75    | p90")
    print("  " + "-" * 80)
    for label, values in (("in-corpus (train)", train_stat), ("out-of-corpus (eval)", eval_stat)):
        row = percentiles(values)
        print(
            f"  {label:<21} | {row['mean']:.4f} | {row['median']:.4f} | {row['p10']:.4f} | "
            f"{row['p25']:.4f} | {row['p75']:.4f} | {row['p90']:.4f}"
        )

    # oracle bounds: what could ANY gate achieve on these 500 pairs?
    contexts, _ = contexts_for_arm_a(
        memory, eval_current, eval_sources, args.probe_regions, None, args.gate_top_n
    )
    with_memory = cosine_to_actual(predict(predictor, eval_current, contexts), eval_actual)
    without = cosine_to_actual(
        predict(predictor, eval_current, np.zeros_like(contexts)), eval_actual
    )
    alignment = np.asarray(
        [
            float(
                contexts[index].astype(np.float64)
                @ tangent_direction(eval_current[index], eval_actual[index])
            )
            for index in range(len(eval_current))
        ]
    )
    best_per_query = np.maximum(with_memory, without)
    alignment_gate = np.where(alignment >= 0.0, with_memory, without)
    print("\n  oracle bounds on the eval pairs (mean cos)")
    print(f"    always memory (ARM B)              : {with_memory.mean():.4f}")
    print(f"    never memory  (ARM C)              : {without.mean():.4f}")
    print(
        f"    oracle: better of the two per query: {best_per_query.mean():.4f}  "
        f"(+{best_per_query.mean() - without.mean():.4f} vs ARM C)"
    )
    print(
        f"    oracle: alignment >= 0 gate        : {alignment_gate.mean():.4f}  "
        f"(+{alignment_gate.mean() - without.mean():.4f} vs ARM C)"
    )

    return {
        "eval_gate_statistic": percentiles(eval_stat),
        "train_gate_statistic": percentiles(train_stat),
        "oracle": {
            "always_memory": float(with_memory.mean()),
            "never_memory": float(without.mean()),
            "better_of_two_per_query": float(best_per_query.mean()),
            "alignment_gate": float(alignment_gate.mean()),
            "gain_vs_never_memory_better_of_two": float(best_per_query.mean() - without.mean()),
            "gain_vs_never_memory_alignment_gate": float(
                alignment_gate.mean() - without.mean()
            ),
            "share_queries_memory_helps": float((with_memory > without).mean()),
            "mean_alignment": float(alignment.mean()),
        },
    }


def extra_thresholds(
    thresholds: Sequence[float], calibration: Dict[str, Any], count: int
) -> List[float]:
    """Thresholds inside the in-corpus / out-of-corpus overlap, so the sweep bites.

    The informative range is where the two distributions overlap: above the train
    median the gate would reject most in-corpus queries (losing the upside); below the
    eval p75 it accepts almost everything (doing nothing). Candidates are drawn from
    that window and skipped when they are within 0.005 of a requested threshold.
    """
    if count <= 0:
        return []
    train_median = calibration["train_gate_statistic"]["median"]
    eval_p10 = calibration["eval_gate_statistic"]["p10"]
    eval_p75 = calibration["eval_gate_statistic"]["p75"]
    low, high = min(eval_p10, train_median), max(eval_p10, train_median)
    candidates = [low + (high - low) * fraction for fraction in (0.2, 0.4, 0.6, 0.8)]
    candidates += [eval_p75, train_median]
    extras: List[float] = []
    for candidate in candidates:
        rounded = round(float(candidate), 3)
        if any(abs(rounded - value) < 0.005 for value in list(thresholds) + extras):
            continue
        extras.append(rounded)
        if len(extras) >= count:
            break
    return extras



def run_threshold(
    args: argparse.Namespace,
    predictor,
    threshold: float,
    train_current: np.ndarray,
    train_actual: np.ndarray,
    eval_current: np.ndarray,
    eval_actual: np.ndarray,
    eval_sources: np.ndarray,
) -> Dict[str, Any]:
    """One full loop with the gate at ``threshold``; returns its summary rows.

    The memory is rebuilt from scratch for every threshold (the added outcome weights
    depend on the gated prediction quality, so the memory genuinely differs between
    passes) - which also means the ARM B / ARM C columns should come out nearly
    identical across passes. That is reported as a determinism check rather than
    assumed.
    """
    memory = build_memory(args)
    started = time.perf_counter()
    result = run_loop_core(
        predictor,
        memory,
        train_current,
        train_actual,
        eval_current,
        eval_actual,
        eval_sources,
        eval_every=args.eval_every,
        probe_regions=args.probe_regions,
        gate_threshold=threshold,
        gate_top_n=args.gate_top_n,
        collect_steps=False,
        verbose=False,
    )
    checkpoints = result["checkpoints"]
    active = [row for row in checkpoints if row["step"] > args.auto_fit_after]
    final = checkpoints[-1] if checkpoints else {}
    summary: Dict[str, Any] = {
        "threshold": float(threshold),
        "seconds": time.perf_counter() - started,
        "eval_cos_final": float(final.get("mean_cos_arm_a", 0.0)),
        "eval_cos_memory_active_mean": float(
            np.mean([row["mean_cos_arm_a"] for row in active])
        )
        if active
        else 0.0,
        "gate_rate_final": float(final.get("gate_rate", 0.0)),
        "gate_rate_memory_active": float(np.mean([row["gate_rate"] for row in active]))
        if active
        else 0.0,
        "arm_b_final": float(final.get("mean_cos_arm_b", 0.0)),
        "arm_c_final": float(final.get("mean_cos_arm_c", 0.0)),
        "vs_arm_c_final": float(final.get("delta_a_vs_c", 0.0)),
        "vs_arm_b_final": float(final.get("delta_a_vs_b", 0.0)),
        "memory_size_final": int(final.get("memory_size", 0)),
        "paired_t_vs_c": float(final.get("arm_a_gated", {}).get("paired_t_statistic", 0.0)),
        "paired_t_vs_b": float(final.get("arm_a_vs_b", {}).get("paired_t_statistic", 0.0)),
        "share_a_wins_vs_c": float(final.get("arm_a_gated", {}).get("share_arm_a_wins", 0.0)),
        "checkpoints": [
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in checkpoints
        ],
        # per-pair arrays of the last checkpoint: needed for the paired tests in main
        "_cos_a": final.get("_cos_a"),
        "_cos_b": final.get("_cos_b"),
        "_cos_c": final.get("_cos_c"),
        "_memory_summary": memory.summary(),
    }
    print(
        f"  threshold {threshold:.3f} | gate rate {100 * summary['gate_rate_memory_active']:>5.1f}% "
        f"| eval cos @10000 {summary['eval_cos_final']:.4f} "
        f"| memory-active mean {summary['eval_cos_memory_active_mean']:.4f} "
        f"| vs C {summary['vs_arm_c_final']:+.4f} | vs B {summary['vs_arm_b_final']:+.4f} "
        f"| t(C) {summary['paired_t_vs_c']:+.2f} | {summary['seconds']:.0f}s"
    )
    return summary



def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print("=" * 100)
    print("TASK 30 - retrieval-confidence gate: threshold sweep over three arms")
    print("=" * 100)

    train_pairs = stream_pairs(args.train_pairs, args.steps)
    val_pairs = load_pairs(args.val_pairs)
    eval_pairs = val_pairs[args.eval_start : args.eval_start + args.eval_size]
    print(
        f"\ndata: {len(train_pairs)} streamed train pairs | eval "
        f"val_pairs[{args.eval_start}:{args.eval_start + len(eval_pairs)}] "
        f"({len(eval_pairs)} pairs, never added to the memory)"
    )

    vae = load_vae(args.vae, device)
    train_sentences, train_index = unique_sentences(train_pairs)
    train_latents = encode_sentences_cached(vae, train_sentences, args.train_cache)
    train_current = np.ascontiguousarray(
        train_latents[[train_index[first] for first, _ in train_pairs]]
    )
    train_actual = np.ascontiguousarray(
        train_latents[[train_index[second] for _, second in train_pairs]]
    )
    val_sentences, val_index = unique_sentences(val_pairs)
    val_latents = encode_sentences_cached(vae, val_sentences, args.val_cache)
    eval_current = np.ascontiguousarray(val_latents[[val_index[a] for a, _ in eval_pairs]])
    eval_actual = np.ascontiguousarray(val_latents[[val_index[b] for _, b in eval_pairs]])
    eval_sources = np.arange(EVAL_SOURCE_BASE, EVAL_SOURCE_BASE + len(eval_pairs), dtype=np.int64)

    predictor = load_latent_predictor(args.predictor, device)
    predictor.eval()
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)
    print(
        f"predictor: {predictor.count_parameters():,} params, FROZEN | "
        f"RegionStore(num_regions={args.num_regions}, probe_regions={args.probe_regions}, "
        f"max_size={args.max_size}, score_mode={SCORE_MODE}, "
        f"auto_fit_after={args.auto_fit_after}) | gate = mean top-{args.gate_top_n} "
        f"neighbour cosine >= threshold"
    )

    # ---------------------------------------------------------------- calibration
    calibration_memory = build_memory(args)
    print(f"\nbuilding the calibration memory ({len(train_current)} adds) ...")
    started = time.perf_counter()
    fill_memory(calibration_memory, train_current, train_actual)
    print(
        f"  size {calibration_memory.size} | regions {calibration_memory.num_regions} "
        f"| fitted {calibration_memory.is_fitted} | {time.perf_counter() - started:.0f}s"
    )
    sample = np.sort(
        rng.choice(
            len(train_current),
            size=min(CALIBRATION_SAMPLE, len(train_current)),
            replace=False,
        )
    )
    calibration = calibrate(
        args,
        calibration_memory,
        predictor,
        train_current[sample],
        train_actual[sample],
        eval_current,
        eval_actual,
        eval_sources,
        sample,
    )

    thresholds = sorted(set(float(value) for value in args.thresholds))
    extras = extra_thresholds(thresholds, calibration, args.auto_extend)
    if extras:
        print(
            f"\n  auto-extended thresholds inside the overlap of the two distributions: "
            f"{extras}"
        )
    thresholds = sorted(set(thresholds + extras))

    print(f"\nsweep: {len(thresholds)} thresholds, one full 10,000-step loop each")
    rows: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for threshold in thresholds:
        rows.append(
            run_threshold(
                args,
                predictor,
                threshold,
                train_current,
                train_actual,
                eval_current,
                eval_actual,
                eval_sources,
            )
        )
    print(f"\n  sweep wall time: {time.perf_counter() - started:.0f}s")

    return _report(args, calibration, rows, eval_pairs)



def _report(
    args: argparse.Namespace,
    calibration: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    eval_pairs: Sequence[Tuple[str, str]],
) -> Dict[str, Any]:
    """Print the requested table and the verdict, then save the results JSON."""
    # pre-registered selection rule: highest mean eval cos over the MEMORY-ACTIVE
    # checkpoints (not the single final checkpoint), so the pick cannot be cherry-picked
    best = max(rows, key=lambda row: row["eval_cos_memory_active_mean"])
    baseline_b = float(np.mean([row["arm_b_final"] for row in rows]))
    baseline_c = float(np.mean([row["arm_c_final"] for row in rows]))
    spread_b = float(
        np.max([row["arm_b_final"] for row in rows]) - np.min([row["arm_b_final"] for row in rows])
    )
    spread_c = float(
        np.max([row["arm_c_final"] for row in rows]) - np.min([row["arm_c_final"] for row in rows])
    )

    print("\n" + "=" * 100)
    print("THRESHOLD SWEEP - ARM A (gated memory) at step 10,000 vs the two baselines")
    print("=" * 100)
    print(
        "  threshold | gate_rate | eval_cos | vs ARM_C | vs ARM_B | t(A-C) | t(A-B)"
        " | mem-active mean"
    )
    print("  " + "-" * 96)
    for row in rows:
        print(
            f"  {row['threshold']:>9.3f} | {100 * row['gate_rate_memory_active']:>8.1f}% | "
            f"{row['eval_cos_final']:>8.4f} | {row['vs_arm_c_final']:>+8.4f} | "
            f"{row['vs_arm_b_final']:>+8.4f} | {row['paired_t_vs_c']:>+6.2f} | "
            f"{row['paired_t_vs_b']:>+6.2f} | {row['eval_cos_memory_active_mean']:>15.4f}"
        )
    print("  " + "-" * 96)
    print(
        f"  references: ARM B (always memory) {baseline_b:.4f} | ARM C (no memory) "
        f"{baseline_c:.4f} | across-pass variation of those baselines: B {spread_b:.6f}, "
        f"C {spread_c:.6f}"
    )

    print("\nKEY QUESTION - is there a threshold where ARM A > ARM C?")
    winners = [row for row in rows if row["vs_arm_c_final"] > 0]
    oracle = calibration["oracle"]
    if winners:
        print(
            f"  YES at {len(winners)} threshold(s): "
            + ", ".join(
                f"{row['threshold']:.3f} ({row['vs_arm_c_final']:+.4f})" for row in winners
            )
        )
    else:
        print("  NO - no threshold in the sweep makes the gated arm beat no-memory.")
    print(
        f"  best by pre-registered rule (memory-active mean eval cos): threshold "
        f"{best['threshold']:.3f} -> {best['eval_cos_memory_active_mean']:.4f} "
        f"(vs C {best['vs_arm_c_final']:+.4f}, vs B {best['vs_arm_b_final']:+.4f}, "
        f"gate rate {100 * best['gate_rate_memory_active']:.1f}%)"
    )
    print(
        f"  oracle bounds on the eval pairs: better-of-two-per-query "
        f"{oracle['better_of_two_per_query']:.4f} "
        f"({oracle['gain_vs_never_memory_better_of_two']:+.4f} vs no memory); "
        f"alignment-gate {oracle['alignment_gate']:.4f} "
        f"({oracle['gain_vs_never_memory_alignment_gate']:+.4f}); memory helps "
        f"{100 * oracle['share_queries_memory_helps']:.1f}% of pairs, mean context "
        f"alignment {oracle['mean_alignment']:+.4f}"
    )

    print(
        f"\nPAIRED TESTS at the best threshold ({len(eval_pairs)} eval pairs, last checkpoint)"
    )
    final_a, final_b, final_c = best["_cos_a"], best["_cos_b"], best["_cos_c"]
    if final_a is not None:
        stats_ac = paired_stats(final_a, final_c)
        stats_ab = paired_stats(final_a, final_b)
        for label, stats in (
            ("ARM A (gated) vs ARM C (no memory)", stats_ac),
            ("ARM A (gated) vs ARM B (always memory)", stats_ab),
        ):
            print(
                f"  {label:<38} delta {stats['mean_delta']:+.4f} | A wins "
                f"{100 * stats['share_arm_a_wins']:.1f}% ({stats['arm_a_wins']}/"
                f"{stats['num_examples']}) | paired t {stats['paired_t_statistic']:+.2f}"
            )
        significant_ac = abs(stats_ac["paired_t_statistic"]) > 1.96
        significant_ab = abs(stats_ab["paired_t_statistic"]) > 1.96
    else:
        stats_ac = {}
        stats_ab = {}
        significant_ac = significant_ab = False
    return _write_results(
        args, calibration, rows, eval_pairs, best, winners, oracle, stats_ac, stats_ab,
        significant_ac, significant_ab, baseline_b, baseline_c, spread_b, spread_c,
    )



def _write_results(
    args: argparse.Namespace,
    calibration: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    eval_pairs: Sequence[Tuple[str, str]],
    best: Dict[str, Any],
    winners: Sequence[Dict[str, Any]],
    oracle: Dict[str, Any],
    stats_ac: Dict[str, Any],
    stats_ab: Dict[str, Any],
    significant_ac: bool,
    significant_ab: bool,
    baseline_b: float,
    baseline_c: float,
    spread_b: float,
    spread_c: float,
) -> Dict[str, Any]:
    """Build and save ``experiments/gated_loop_results.json``."""
    payload = {
        "task": "TASK 30 - retrieval-confidence gate sweep",
        "gate": {
            "enabled": True,
            "top_n": int(args.gate_top_n),
            "metric": "mean cosine of the top-n retrieved neighbours vs the query",
            "fallback": "C = zeros(256) when the gate rejects a query",
            "thresholds_requested": [float(value) for value in args.thresholds],
            "thresholds_run": [float(row["threshold"]) for row in rows],
            "auto_extended": bool(args.auto_extend),
        },
        "config": {
            "steps": int(args.steps),
            "eval_every": int(args.eval_every),
            "eval_size": len(eval_pairs),
            "eval_pair_indices": [args.eval_start, args.eval_start + len(eval_pairs) - 1],
            "num_regions": int(args.num_regions),
            "probe_regions": int(args.probe_regions),
            "max_size": int(args.max_size),
            "score_mode": SCORE_MODE,
            "auto_fit_after": int(args.auto_fit_after),
            "predictor_frozen": True,
            "seed": int(args.seed),
        },
        "arms": {
            "a": "gated memory (C only when the gate trusts the retrieval)",
            "b": "always memory (TASK 29 behaviour)",
            "c": "never memory (C = zeros)",
        },
        "calibration": calibration,
        "baselines": {
            "arm_b_always_memory_final": baseline_b,
            "arm_c_never_memory_final": baseline_c,
            "arm_b_across_pass_spread": spread_b,
            "arm_c_across_pass_spread": spread_c,
        },
        "sweep": [
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in rows
        ],
        "best": {
            "threshold": best["threshold"],
            "rule": (
                "highest mean eval cos over the memory-active checkpoints "
                "(step > auto_fit_after), pre-registered"
            ),
            "eval_cos_final": best["eval_cos_final"],
            "eval_cos_memory_active_mean": best["eval_cos_memory_active_mean"],
            "gate_rate_final": best["gate_rate_final"],
            "gate_rate_memory_active": best["gate_rate_memory_active"],
            "vs_arm_c_final": best["vs_arm_c_final"],
            "vs_arm_b_final": best["vs_arm_b_final"],
            "paired_t_vs_c": best["paired_t_vs_c"],
            "paired_t_vs_b": best["paired_t_vs_b"],
            "memory_size_final": best["memory_size_final"],
        },
        "key_question": {
            "thresholds_where_gated_beats_no_memory": [row["threshold"] for row in winners],
            "any_threshold_beats_no_memory": bool(winners),
            "oracle_upper_bounds": oracle,
        },
        "paired_tests_at_best": {
            "arm_a_vs_c": stats_ac,
            "arm_a_vs_b": stats_ab,
            "arm_a_vs_c_significant_at_0.05": bool(significant_ac),
            "arm_a_vs_b_significant_at_0.05": bool(significant_ab),
            "note": (
                "paired t over the 500 eval pairs; consecutive sentences from one document "
                "are correlated, so this mildly overstates significance"
            ),
        },
        "caveat_memory_vs_model": (
            "the eval pairs are held out from the MEMORY, not from the model: TASK 27 "
            "trained the predictor on all 7,666 val pairs, including these. All arms share "
            "the frozen predictor, so every delta is a clean memory effect."
        ),
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with args.results.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nsaved {args.results}")
    print("=" * 100)
    return payload


if __name__ == "__main__":
    main()

