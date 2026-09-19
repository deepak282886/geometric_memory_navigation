"""SUPERSEDED by geometric scoring in src/eval/evaluate.py.

This script trained a ClassifierHead over ``[ctx position ; nav vector]`` on top
of the frozen geometric pipeline. It converges (train accuracy 100%) but stays
at chance on validation (~26.6% best) because the four candidate endings never
enter the model - the label is a property of the endings, not of the context.

Kept for reference and reused by the TASK 9 ablations. Inference now lives in
``src/eval/evaluate.py::score_example`` (geometric ending scoring, no head).

Original docstring follows.
--------------------------------------------------------------------------
TASK 7: train the classifier head on top of the frozen GMN pipeline.

For every training example the geometric pipeline is recomputed:

    unit query vector (mu, L2-normalised)
      -> store.query_excluding(self)         k=10 neighbours, self removed
      -> outcome weights from the true label (1.0 match / 0.1 mismatch)
      -> parallel_sample                      (weight, direction) per memory
      -> fuse_directions(top_k=3)
      -> navigate                            -> 512-d classifier input

Gradients are accumulated over 32 examples before each optimiser step. The VAE
stays frozen throughout; only the ClassifierHead is trained.

IMPORTANT: ``vae.encode_texts`` returns ``mu`` with norm ~7.2, so every query
vector is L2-normalised ONCE up front and the unit vector is reused for the
FAISS lookup, the sampler, and ``navigate``. The memory store already holds unit
vectors, so all cosine similarities are true cosines.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.autoencoder.vae import VAE  # noqa: E402
from src.classifier.head import ClassifierHead  # noqa: E402
from src.fusion.fuse import fuse_directions, navigate  # noqa: E402
from src.memory.store import MemoryStore, normalize_rows  # noqa: E402
from src.sampler.jev import parallel_sample  # noqa: E402

TRAIN_PATH = PROJECT_ROOT / "data" / "train.json"
VAL_PATH = PROJECT_ROOT / "data" / "val.json"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "vae.pt"
STORE_PATH = PROJECT_ROOT / "checkpoints" / "memory_store.pkl"
CLASSIFIER_PATH = PROJECT_ROOT / "checkpoints" / "classifier.pt"

EPOCHS = 20
ACCUMULATION_STEPS = 32
LEARNING_RATE = 1e-3
RETRIEVAL_K = 10
FUSE_TOP_K = 3
SEED = 42
EVAL_BATCH_SIZE = 256
INFERENCE_OUTCOME_WEIGHT = 1.0

#: kept so our own messages survive while the sampler's chatter is redirected
_REAL_STDOUT = sys.stdout


def log(message: str = "") -> None:
    """Print to the real stdout (the sampler prints once per example)."""
    print(message, file=_REAL_STDOUT, flush=True)


def load_vae(checkpoint_path: Path, device: torch.device) -> Tuple[VAE, Dict[str, Any]]:
    """Rebuild the VAE from the checkpoint config (same pattern as TASK 3/4/6)."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    vae = VAE(
        base_model_name=config["base_model_name"],
        input_dim=config["base_embed_dim"],
        hidden_dim=config["head_hidden_dim"],
        latent_dim=config["latent_dim"],
        decoder_hidden_dim=config["decoder_hidden_dim"],
        logvar_bias=config["logvar_bias"],
        device=device,
    ).to(device)
    missing, unexpected = vae.load_state_dict(checkpoint["model_state_dict"], strict=False)
    real_missing = [key for key in missing if not key.startswith("encoder.")]
    if real_missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={real_missing} unexpected={unexpected}")
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)  # frozen: nothing here is trained
    return vae, checkpoint


def encode_unit_vectors(vae: VAE, texts: Sequence[str]) -> np.ndarray:
    """Frozen encode + L2 normalise -> ``(N, 256)`` unit-norm float32 vectors."""
    with torch.no_grad():
        raw = vae.encode_texts(list(texts)).cpu().numpy()
    return normalize_rows(raw).astype(np.float32)


def build_classifier_input(
    store: MemoryStore,
    query_vector: np.ndarray,
    true_label: int | None = None,
    exclude_idx: int | None = None,
    k: int = RETRIEVAL_K,
    fuse_top_k: int = FUSE_TOP_K,
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[float]]:
    """Retrieve -> sample -> fuse -> navigate for a single example.

    ``exclude_idx`` removes a training example from its own neighbour list;
    ``true_label`` enables the supervised outcome weights. Returns
    ``(classifier_input, memories, outcome_weights)``.
    """
    if exclude_idx is None:
        memories = store.query(query_vector, k=k)
    else:
        memories = store.query_excluding(query_vector, int(exclude_idx), k=k)

    if true_label is None:
        outcome_weights = [INFERENCE_OUTCOME_WEIGHT] * len(memories)
    else:
        outcome_weights = [
            store.get_training_outcome_weight(memory["label"], int(true_label))
            for memory in memories
        ]

    weighted_directions = parallel_sample(query_vector, memories, outcome_weights)
    navigation_vector = fuse_directions(weighted_directions, top_k=fuse_top_k)
    return navigate(query_vector, navigation_vector), memories, outcome_weights


@torch.no_grad()
def batch_accuracy(classifier: ClassifierHead, inputs, labels, batch_size: int = EVAL_BATCH_SIZE) -> float:
    """Argmax accuracy of the classifier over a list of 512-d inputs."""
    classifier.eval()
    correct = 0
    for start in range(0, len(inputs), batch_size):
        chunk = np.asarray(inputs[start : start + batch_size], dtype=np.float32)
        truth = torch.tensor(labels[start : start + batch_size], dtype=torch.long)
        predictions = classifier(torch.from_numpy(chunk)).argmax(dim=1)
        correct += int((predictions == truth).sum())
    return correct / max(len(inputs), 1)


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = torch.device("cpu")
    overall_started = time.perf_counter()

    with TRAIN_PATH.open(encoding="utf-8") as handle:
        train_data = json.load(handle)
    with VAL_PATH.open(encoding="utf-8") as handle:
        val_data = json.load(handle)

    log(f"train examples {len(train_data)} | val examples {len(val_data)}")

    vae, vae_checkpoint = load_vae(CHECKPOINT_PATH, device)
    store = MemoryStore.load(STORE_PATH)
    log(f"VAE loaded (latent_dim={vae_checkpoint['config']['latent_dim']}) | "
        f"encoder frozen {vae.encoder.is_frozen}")

    log("\nPre-encoding all ctx strings with the frozen VAE, then L2-normalising ...")
    pre_started = time.perf_counter()
    train_vectors = encode_unit_vectors(vae, [example["ctx"] for example in train_data])
    val_vectors = encode_unit_vectors(vae, [example["ctx"] for example in val_data])
    train_labels = [int(example["label"]) for example in train_data]
    val_labels = [int(example["label"]) for example in val_data]
    log(f"  train_vectors {train_vectors.shape} | val_vectors {val_vectors.shape} "
        f"in {time.perf_counter() - pre_started:.1f}s")
    train_norms = np.linalg.norm(train_vectors, axis=1)
    val_norms = np.linalg.norm(val_vectors, axis=1)
    log(f"  train row norms min {train_norms.min():.6f} / max {train_norms.max():.6f} "
        f"(unit: {bool(np.allclose(train_norms, 1.0, atol=1e-5))})")
    log(f"  val   row norms min {val_norms.min():.6f} / max {val_norms.max():.6f} "
        f"(unit: {bool(np.allclose(val_norms, 1.0, atol=1e-5))})")
    log(f"  store vectors unit norm: "
        f"{bool(np.allclose(np.linalg.norm(store.vectors, axis=1), 1.0, atol=1e-5))}")
    log(f"  identical train/store vectors: "
        f"{bool(np.allclose(train_vectors, store.vectors, atol=1e-6))}")

    # sanity: a training example must not retrieve itself
    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet):
        self_hits = sum(
            1
            for probe in rng.choice(len(train_data), size=20, replace=False)
            for record in store.query_excluding(train_vectors[int(probe)], int(probe), k=RETRIEVAL_K)
            if record["idx"] == int(probe)
        )
    log(f"  query_excluding self-retrieval over 20 probes: {self_hits} (expect 0)")

    classifier = ClassifierHead()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE)
    log(f"\n{classifier}")
    log(f"classifier parameters : {classifier.count_parameters():,}")
    log(f"optimizer Adam lr={LEARNING_RATE} | accumulation={ACCUMULATION_STEPS} | "
        f"epochs={EPOCHS}")

    # ------------------------------------------------------------- training
    history: List[Dict[str, float]] = []
    best_val_accuracy = -1.0
    best_epoch = 0
    train_accuracy = 0.0
    val_accuracy = 0.0

    log("\n" + "=" * 72)
    log(f"TRAINING ({EPOCHS} epochs, gradient accumulation {ACCUMULATION_STEPS})")
    log("=" * 72)
    log("epoch |   train_loss | train_acc | val_acc")
    log("-" * 58)

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for epoch in range(1, EPOCHS + 1):
                epoch_started = time.perf_counter()
                classifier.train()

                order = rng.permutation(len(train_data))
                epoch_inputs: List[Any] = [None] * len(order)
                running_loss = 0.0
                pending = 0

                for step, raw_index in enumerate(order):
                    index = int(raw_index)
                    query_vector = train_vectors[index]

                    classifier_input, _, _ = build_classifier_input(
                        store,
                        query_vector,
                        true_label=train_labels[index],
                        exclude_idx=index,
                    )
                    epoch_inputs[step] = classifier_input

                    x = torch.tensor(classifier_input, dtype=torch.float32).unsqueeze(0)
                    logits = classifier(x)  # (1, 4) raw logits
                    loss = criterion(
                        logits, torch.tensor([train_labels[index]], dtype=torch.long)
                    )
                    running_loss += float(loss)

                    (loss / ACCUMULATION_STEPS).backward()
                    pending += 1
                    if pending % ACCUMULATION_STEPS == 0:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                # flush the leftover gradients (2000 is not a multiple of 32)
                if pending % ACCUMULATION_STEPS != 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                train_accuracy = batch_accuracy(
                    classifier, epoch_inputs, [train_labels[int(i)] for i in order]
                )

                val_inputs = [
                    build_classifier_input(store, val_vectors[i])[0]
                    for i in range(len(val_data))
                ]
                val_accuracy = batch_accuracy(classifier, val_inputs, val_labels)

                mean_loss = running_loss / max(len(order), 1)
                history.append(
                    {
                        "epoch": float(epoch),
                        "train_loss": mean_loss,
                        "train_acc": train_accuracy,
                        "val_acc": val_accuracy,
                    }
                )

                marker = ""
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_epoch = epoch
                    torch.save(
                        {
                            "config": {
                                "input_dim": classifier.input_dim,
                                "hidden_dim_1": classifier.hidden_dim_1,
                                "hidden_dim_2": classifier.hidden_dim_2,
                                "num_classes": classifier.num_classes,
                                "dropout": classifier.dropout.p,
                            },
                            "model_state_dict": classifier.state_dict(),
                            "epoch": epoch,
                            "val_accuracy": val_accuracy,
                            "train_accuracy": train_accuracy,
                            "train_loss": mean_loss,
                            "parameter_count": classifier.count_parameters(),
                            "note": "L2-normalise query vectors before the GMN pipeline",
                        },
                        CLASSIFIER_PATH,
                    )
                    marker = "  <- best, saved"

                log(
                    f"{epoch:>5d} | {mean_loss:>12.6f} | {train_accuracy:>9.4f} | "
                    f"{val_accuracy:>7.4f}{marker}  ({time.perf_counter() - epoch_started:.1f}s)"
                )
    except Exception:
        # make sure a traceback is never swallowed by the stdout redirect
        sys.stdout = _REAL_STDOUT
        raise

    # ------------------------------------------------------------ reporting
    log("\n" + "=" * 72)
    log("FINAL RESULTS")
    log("=" * 72)
    log(f"  final train accuracy : {train_accuracy:.4f} ({100 * train_accuracy:.2f}%)")
    log(f"  final val accuracy   : {val_accuracy:.4f} ({100 * val_accuracy:.2f}%)")
    log(f"  best val accuracy    : {best_val_accuracy:.4f} ({100 * best_val_accuracy:.2f}%) "
        f"at epoch {best_epoch}")
    log(f"  val accuracy > 30%   : {best_val_accuracy > 0.30}")

    vae_head_parameters = sum(p.numel() for p in vae.head.parameters())
    vae_decoder_parameters = sum(p.numel() for p in vae.decoder.parameters())
    classifier_parameters = classifier.count_parameters()
    total_gmn_parameters = (
        vae_head_parameters + vae_decoder_parameters + classifier_parameters
    )

    log("")
    log(f"  VAE head parameters       : {vae_head_parameters:,}")
    log(f"  VAE decoder parameters    : {vae_decoder_parameters:,}")
    log(f"  ClassifierHead parameters : {classifier_parameters:,}")
    log(f"  Total GMN parameters      : {total_gmn_parameters:,}")
    log(f"  total wall time           : {time.perf_counter() - overall_started:.1f}s")
    log(f"\nBest classifier saved to {CLASSIFIER_PATH}")

    log("\nPer-epoch history:")
    log("  epoch |   train_loss | train_acc | val_acc")
    for row in history:
        log(f"  {int(row['epoch']):>5d} | {row['train_loss']:>12.6f} | "
            f"{row['train_acc']:>9.4f} | {row['val_acc']:>7.4f}")
    log("\nClassifier training complete.")


if __name__ == "__main__":
    main()


