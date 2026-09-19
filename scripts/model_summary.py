"""TASK 11: exact parameter count and model summary.

GMN at inference is ``VAEHead`` + ``VAEDecoder`` over a frozen sentence encoder,
plus a FAISS index that holds no learned parameters. The trained counts are read
straight off ``checkpoints/vae.pt``'s state dict (which stores only the 10
``head.*`` and ``decoder.*`` tensors) instead of being reconstructed from the
class definitions, so the numbers cannot drift from the shipped checkpoint.

The frozen ``all-MiniLM-L6-v2`` base encoder is loaded only to be counted, and is
reported as excluded from the GMN total.

Writes ``experiments/model_summary.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier.head import ClassifierHead  # noqa: E402

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "vae.pt"
CLASSIFIER_PATH = PROJECT_ROOT / "checkpoints" / "classifier.pt"
RESULTS_PATH = PROJECT_ROOT / "experiments" / "model_summary.json"

#: gpt2 small, as measured in experiments/results.json / results_full_val.json
GPT2_PARAMETERS = 124_439_808


def summarise_prefix(
    state_dict: Dict[str, torch.Tensor], prefix: str
) -> Tuple[int, List[Dict[str, Any]]]:
    """Total parameter count and per-tensor breakdown for one module prefix."""
    breakdown: List[Dict[str, Any]] = []
    total = 0
    for name, tensor in state_dict.items():
        if not name.startswith(prefix):
            continue
        count = int(tensor.numel())
        total += count
        breakdown.append(
            {"tensor": name, "shape": list(tensor.shape), "parameters": count}
        )
    return total, breakdown


def count_module(module: torch.nn.Module, trainable_only: bool = False) -> int:
    return int(
        sum(
            parameter.numel()
            for parameter in module.parameters()
            if (parameter.requires_grad or not trainable_only)
        )
    )


def base_encoder_parameters() -> Tuple[int, bool, bool]:
    """Parameter count of the frozen base encoder, plus ``(is_frozen, loaded)``.

    Returns ``(0, False, False)`` if sentence-transformers is unavailable, so the
    caller can report "unavailable" rather than inventing a number.
    """
    try:
        from src.autoencoder.vae import TextEncoder
    except Exception as error:  # pragma: no cover - import-time dependency issue
        print(f"  (base encoder unavailable: {error})")
        return 0, False, False

    try:
        encoder = TextEncoder()
    except Exception as error:
        print(f"  (could not load the base encoder: {error})")
        return 0, False, False

    parameters = count_module(encoder)
    frozen = all(not p.requires_grad for p in encoder.parameters())
    return parameters, frozen, True


def main() -> None:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"{CHECKPOINT_PATH} not found - run scripts/train_vae.py first"
        )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    config = checkpoint["config"]

    head_parameters, head_breakdown = summarise_prefix(state_dict, "head.")
    decoder_parameters, decoder_breakdown = summarise_prefix(state_dict, "decoder.")

    print(f"checkpoint {CHECKPOINT_PATH.name} | "
          f"{len(state_dict)} stored tensors | "
          f"best val loss {checkpoint.get('final_loss')}")
    print(f"  latent_dim={config['latent_dim']} | "
          f"head_hidden={config['head_hidden_dim']} | "
          f"decoder_hidden={config['decoder_hidden_dim']} | "
          f"base_embed={config['base_embed_dim']}")

    classifier = ClassifierHead()
    classifier_parameters = count_module(classifier)
    classifier_trained = False
    if CLASSIFIER_PATH.exists():
        loaded = torch.load(CLASSIFIER_PATH, map_location="cpu", weights_only=False)
        classifier_trained = True
        print(f"  ClassifierHead checkpoint present ({CLASSIFIER_PATH.name}) but "
              f"unused at inference")

    base_parameters, base_frozen, base_loaded = base_encoder_parameters()

    gmn_parameters = head_parameters + decoder_parameters
    inference_extra = 0  # FAISS index: stored vectors only, no learned parameters

    # ------------------------------------------------------------------ table
    rows: List[Dict[str, Any]] = [
        {
            "component": "Base encoder (frozen)",
            "parameters": base_parameters,
            "notes": "excluded from GMN count",
        },
        {
            "component": "VAE head",
            "parameters": head_parameters,
            "notes": "trained",
        },
        {
            "component": "VAE decoder",
            "parameters": decoder_parameters,
            "notes": "trained",
        },
        {
            "component": "ClassifierHead",
            "parameters": classifier_parameters,
            "notes": "superseded, not used",
        },
        {
            "component": "Total GMN (inference)",
            "parameters": gmn_parameters,
            "notes": "VAE head + decoder only",
        },
        {
            "component": "GPT-2 small",
            "parameters": GPT2_PARAMETERS,
            "notes": "baseline",
        },
    ]

    print("\n" + "=" * 78)
    print("PARAMETER SUMMARY")
    print("=" * 78)
    print(f"  {'Component':<23}| {'Parameters':>11} | Notes")
    print(f"  {'-' * 23}|{'-' * 13}|{'-' * 38}")
    for row in rows:
        value = f"{row['parameters']:,}" if row["parameters"] else "n/a"
        print(f"  {row['component']:<23}| {value:>11} | {row['notes']}")
    print(f"  {'-' * 23}|{'-' * 13}|{'-' * 38}")

    fraction = gmn_parameters / GPT2_PARAMETERS
    reduction = GPT2_PARAMETERS / gmn_parameters
    print(f"\n  GMN is {100 * fraction:.2f}% the size of GPT-2 small "
          f"({reduction:.0f}x smaller, a {reduction:.0f}x parameter reduction)")
    print(f"  Target was GMN < 10,000,000 parameters: "
          f"{'PASS' if gmn_parameters < 10_000_000 else 'FAIL'}")
    print(f"  Inference also runs the frozen base encoder"
          f"{'' if base_loaded else ' (not loaded)'}"
          f"{'' if base_frozen else ' [NOT FROZEN]'} and a FAISS index "
          f"(0 learned parameters)")

    # ------------------------------------------------------------------- save
    payload = {
        "experiment": "TASK 11 model summary",
        "checkpoint": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "config": config,
        "components": rows,
        "counts": {
            "base_encoder_frozen": base_parameters,
            "base_encoder_is_frozen": base_frozen,
            "base_encoder_loaded": base_loaded,
            "vae_head": head_parameters,
            "vae_decoder": decoder_parameters,
            "classifier_head": classifier_parameters,
            "classifier_head_used_at_inference": False,
            "classifier_checkpoint_present": classifier_trained,
            "retrieval_index_learned_parameters": inference_extra,
        },
        "gmn_inference_parameters": gmn_parameters,
        "gpt2_parameters": GPT2_PARAMETERS,
        "gmn_fraction_of_gpt2": fraction,
        "parameter_reduction_factor": reduction,
        "target_under_10m": gmn_parameters < 10_000_000,
        "breakdown": {"vae_head": head_breakdown, "vae_decoder": decoder_breakdown},
        "notes": {
            "gmn_definition": (
                "VAEHead + VAEDecoder. The base sentence encoder is frozen and "
                "excluded; the FAISS memory store holds stored vectors and has no "
                "learned parameters."
            ),
            "source": (
                "counts read from checkpoints/vae.pt model_state_dict so they cannot "
                "drift from the shipped checkpoint"
            ),
        },
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nSaved parameter summary to {RESULTS_PATH}")
    print("Model summary complete.")


if __name__ == "__main__":
    main()

