"""Visualise the VAE latent space with UMAP, coloured by activity label (TASK 3).

Pipeline:
    2000 train ``ctx`` strings
      --vae.encode_texts()--> deterministic 256-d ``mu`` vectors
      --UMAP (cosine)--> 2-d embedding  --> notebooks/latent_space.png

Before plotting we also measure whether the geometry is meaningful by comparing
cosine similarity of same-activity-label pairs against different-label pairs.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.autoencoder.vae import VAE  # noqa: E402

DATA_PATH = PROJECT_ROOT / "data" / "train.json"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "vae.pt"
OUTPUT_PATH = PROJECT_ROOT / "notebooks" / "latent_space.png"

TOP_K_LABELS = 20
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "cosine"
UMAP_RANDOM_STATE = 42
DPI = 150
NUM_PAIRS = 200
SEED = 42
ENCODE_BATCH_SIZE = 64
KNN_PURITY_K = 10


def load_vae(checkpoint_path: Path, device: torch.device) -> Tuple[VAE, Dict[str, Any]]:
    """Rebuild the VAE from the checkpoint config and load head/decoder weights."""
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
    # The base encoder is intentionally not stored, so its keys are "missing".
    real_missing = [key for key in missing if not key.startswith("encoder.")]
    if real_missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={real_missing} unexpected={unexpected}")
    vae.eval()
    return vae, checkpoint


@torch.no_grad()
def encode_contexts(vae: VAE, contexts: Sequence[str]) -> np.ndarray:
    """Deterministic 256-d ``mu`` vectors (no sampling) -> ``(N, 256)`` numpy."""
    vectors = vae.encode_texts(list(contexts), batch_size=ENCODE_BATCH_SIZE)
    return vectors.cpu().numpy().astype(np.float64)


def normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def sample_pair_similarities(
    unit_vectors: np.ndarray,
    labels: Sequence[str],
    same_label: bool,
    n_pairs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Cosine similarity of ``n_pairs`` sampled same-label or different-label pairs."""
    by_label: Dict[str, List[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_label[label].append(index)
    multi = [label for label, indices in by_label.items() if len(indices) >= 2]

    similarities: List[float] = []
    while len(similarities) < n_pairs:
        if same_label:
            label = multi[int(rng.integers(len(multi)))]
            pool = by_label[label]
            i, j = rng.choice(len(pool), size=2, replace=False)
            first, second = pool[int(i)], pool[int(j)]
        else:
            first = int(rng.integers(len(labels)))
            second = int(rng.integers(len(labels)))
            if labels[first] == labels[second]:
                continue
        similarities.append(float(unit_vectors[first] @ unit_vectors[second]))
    return np.asarray(similarities)


def describe(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def knn_label_purity(embedding: np.ndarray, labels: Sequence[str], k: int) -> float:
    """Fraction of each point's ``k`` nearest 2-d neighbours that share its label."""
    diff = embedding[:, None, :] - embedding[None, :, :]
    distances = np.sqrt((diff**2).sum(axis=-1))
    np.fill_diagonal(distances, np.inf)
    neighbours = np.argsort(distances, axis=1)[:, :k]
    labels_array = np.asarray(labels)
    return float((labels_array[neighbours] == labels_array[:, None]).mean())


def chance_purity(labels: Sequence[str]) -> float:
    """Purity expected from random labelling: sum over labels of p(label)^2."""
    counts = Counter(labels)
    total = len(labels)
    return float(sum((count / total) ** 2 for count in counts.values()))


def plot_latent(
    embedding: np.ndarray,
    labels: Sequence[str],
    top_labels: Sequence[str],
    counts: Counter,
    same_stats: Dict[str, float],
    diff_stats: Dict[str, float],
    purity: float,
    chance: float,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(15, 11))

    top_set = set(top_labels)
    other_mask = np.array([label not in top_set for label in labels])
    ax.scatter(
        embedding[other_mask, 0],
        embedding[other_mask, 1],
        s=9,
        c="lightgrey",
        alpha=0.55,
        linewidths=0,
        label=f"other activity labels (n={int(other_mask.sum())})",
    )

    colors = plt.cm.tab20(np.linspace(0.0, 1.0, len(top_labels)))
    for color, label in zip(colors, top_labels):
        mask = np.array([item == label for item in labels])
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=20,
            c=[color],
            alpha=0.9,
            linewidths=0,
            label=f"{label} (n={counts[label]})",
        )

    ax.set_title("Latent Space by Activity Label", fontsize=16, fontweight="bold", pad=26)
    ax.text(
        0.5,
        1.018,
        "Top 20 activity labels shown (all remaining labels in grey)",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        style="italic",
        color="dimgray",
    )
    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)
    ax.grid(alpha=0.15, linewidth=0.5)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.055),
        ncol=3,
        fontsize=8.5,
        frameon=False,
        markerscale=1.8,
        handletextpad=0.4,
        columnspacing=1.2,
    )

    summary = (
        f"cosine similarity of mu: same-label {same_stats['mean']:+.3f} "
        f"vs different-label {diff_stats['mean']:+.3f}\n"
        f"10-NN label purity {purity:.3f} (chance {chance:.3f})"
    )
    ax.text(
        0.995,
        0.005,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="lightgrey", alpha=0.9),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(SEED)
    device = torch.device("cpu")

    with DATA_PATH.open("r", encoding="utf-8") as handle:
        train_data = json.load(handle)
    contexts = [example["ctx"] for example in train_data]
    labels = [example["activity_label"] for example in train_data]
    counts = Counter(labels)

    print(f"Loaded {len(train_data)} train examples")
    print(f"Unique activity labels found: {len(counts)}")
    print("\nTop 5 most frequent activity labels:")
    for rank, (label, count) in enumerate(counts.most_common(5), start=1):
        print(f"  {rank}. {label:<45s} {count:>4d}  ({100.0 * count / len(labels):.1f}%)")

    print(f"\nLoading VAE from {CHECKPOINT_PATH} ...")
    vae, checkpoint = load_vae(CHECKPOINT_PATH, device)
    print(f"  latent_dim={checkpoint['config']['latent_dim']} "
          f"| epochs={checkpoint['epochs']} | final_loss={checkpoint['final_loss']:.6f}")
    print(f"  encoder frozen: {vae.encoder.is_frozen}")

    print(f"\nEncoding {len(contexts)} ctx strings with vae.encode_texts() (mu, deterministic) ...")
    vectors = encode_contexts(vae, contexts)
    print(f"  latent matrix: {vectors.shape} (expected ({len(contexts)}, 256))")
    print(f"  finite: {bool(np.isfinite(vectors).all())} | "
          f"min {vectors.min():+.3f} | max {vectors.max():+.3f} | "
          f"per-row std mean {vectors.std(axis=1).mean():.3f}")

    unit = normalize(vectors)
    same_stats = describe(sample_pair_similarities(unit, labels, True, NUM_PAIRS, rng))
    diff_stats = describe(sample_pair_similarities(unit, labels, False, NUM_PAIRS, rng))
    print(f"\nCosine similarity of mu over {NUM_PAIRS} sampled pairs:")
    print(f"  same activity label      : mean {same_stats['mean']:+.4f}  "
          f"std {same_stats['std']:.4f}  range [{same_stats['min']:+.3f}, {same_stats['max']:+.3f}]")
    print(f"  different activity label : mean {diff_stats['mean']:+.4f}  "
          f"std {diff_stats['std']:.4f}  range [{diff_stats['min']:+.3f}, {diff_stats['max']:+.3f}]")
    print(f"  separation (same - diff) : {same_stats['mean'] - diff_stats['mean']:+.4f}")

    import umap  # imported here so the dependency requirement is explicit

    print(f"\nRunning UMAP ({vectors.shape[1]}-d -> 2-d) [n_neighbors={UMAP_N_NEIGHBORS}, "
          f"min_dist={UMAP_MIN_DIST}, metric={UMAP_METRIC}] ...")
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
    )
    embedding = reducer.fit_transform(vectors)
    print(f"  UMAP embedding: {embedding.shape}")

    purity = knn_label_purity(embedding, labels, KNN_PURITY_K)
    chance = chance_purity(labels)
    print("\nCluster diagnostics (2-d embedding):")
    print(f"  {KNN_PURITY_K}-NN label purity : {purity:.4f}  "
          f"(chance {chance:.4f}, x{purity / chance:.2f} over chance)")

    top_labels = [label for label, _ in counts.most_common(TOP_K_LABELS)]
    print(f"\nPlotting top {len(top_labels)} labels, "
          f"{len(counts) - len(top_labels)} remaining labels in grey ...")
    plot_latent(
        embedding, labels, top_labels, counts, same_stats, diff_stats, purity, chance, OUTPUT_PATH
    )

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\nSaved plot: {OUTPUT_PATH} ({size_mb:.3f} MB, dpi={DPI})")
    print("Latent space visualization complete.")


if __name__ == "__main__":
    main()
