"""TASK 10: geometric navigation in latent space, adapted to the additive model.

For 5 validation examples this collects, in the 256-d latent space:

    query position          the encoded context            (red)
    top-3 memory neighbours the contexts fusion selected   (grey)
    correct ending          unit latent of endings[label]  (green)
    wrong endings           unit latents of the other 3    (light grey)
    navigation target       query + navigation_vector      (arrow endpoint)

All of these are projected together with UMAP into 2-D and plotted.

Adaptation from the original TASK 10: the model no longer MOVES to a landing
position - it scores ``cos(ending, ctx) + lambda_nav * cos(ending, nav_vector)``.
So the navigation vector is drawn as an ARROW from the query position rather than
as a landing marker, and the ending positions are shown directly so the reader can
see what the additive score actually compares.

Saves ``notebooks/navigation.png``.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.autoencoder.vae import VAE  # noqa: E402
from src.fusion.fuse import fuse_directions  # noqa: E402
from src.memory.store import MemoryStore, normalize_rows  # noqa: E402
from src.sampler.jev import parallel_sample  # noqa: E402

VAL_FULL_PATH = PROJECT_ROOT / "data" / "val_full.json"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "vae.pt"
STORE_PATH = PROJECT_ROOT / "checkpoints" / "memory_store_full_v2.pkl"
OUTPUT_PATH = PROJECT_ROOT / "notebooks" / "navigation.png"

NUM_EXAMPLES = 5
RETRIEVAL_K = 10
FUSE_TOP_K = 3
UMAP_N_NEIGHBORS = 10  # must stay below the number of collected points
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "cosine"
UMAP_RANDOM_STATE = 42
DPI = 150
SEED = 42


def load_vae(checkpoint_path: Path, device: torch.device) -> VAE:
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
    return vae


def main() -> None:
    with VAL_FULL_PATH.open(encoding="utf-8") as handle:
        val_data: List[Dict[str, Any]] = json.load(handle)

    rng = np.random.default_rng(SEED)
    indices = np.sort(rng.choice(len(val_data), size=NUM_EXAMPLES, replace=False))
    print(f"val split {len(val_data)} | plotting {NUM_EXAMPLES} examples "
          f"(seed {SEED}): {indices.tolist()}")

    vae = load_vae(CHECKPOINT_PATH, torch.device("cpu"))
    store = MemoryStore.load(STORE_PATH)
    print(f"store {store.size} | target={store.default_target} | "
          f"k={RETRIEVAL_K} | fuse top_k={FUSE_TOP_K}")

    queries: List[np.ndarray] = []
    neighbours: List[np.ndarray] = []
    correct_endings: List[np.ndarray] = []
    wrong_endings: List[np.ndarray] = []
    navigation_targets: List[np.ndarray] = []
    per_example: List[Dict[str, Any]] = []

    with contextlib.redirect_stdout(io.StringIO()):
        for index in indices:
            example = val_data[int(index)]
            label = int(example["label"])

            with torch.no_grad():
                query_vector = normalize_rows(
                    vae.encode_texts([example["ctx"]]).cpu().numpy()
                )[0].astype(np.float64)
                ending_vectors = normalize_rows(
                    vae.encode_texts(list(example["endings"])).cpu().numpy()
                ).astype(np.float64)

            memories = store.query(query_vector, k=RETRIEVAL_K)
            weighted = parallel_sample(
                query_vector,
                memories,
                [1.0] * len(memories),
                target=store.default_target,
            )
            navigation_vector = fuse_directions(
                weighted, top_k=FUSE_TOP_K, use_tangent=True, query_vector=query_vector
            )
            chosen = np.argsort([-weight for weight, _ in weighted])[:FUSE_TOP_K]

            queries.append(query_vector)
            neighbours.extend(
                np.asarray(memories[int(i)]["vector"], dtype=np.float64) for i in chosen
            )
            correct_endings.append(ending_vectors[label])
            wrong_endings.extend(
                ending_vectors[k] for k in range(len(ending_vectors)) if k != label
            )
            navigation_targets.append(query_vector + navigation_vector)

            navigation_scores = ending_vectors @ navigation_vector
            context_scores = ending_vectors @ query_vector
            per_example.append(
                {
                    "val_index": int(index),
                    "activity_label": example["activity_label"],
                    "label": label,
                    "prediction_from_context": int(np.argmax(context_scores)),
                    "cos_ctx_nav": float(query_vector @ navigation_vector),
                    "nav_norm": float(np.linalg.norm(navigation_vector)),
                    "correct_ending_gap_on_nav": float(
                        navigation_scores[label] - np.delete(navigation_scores, label).mean()
                    ),
                    "neighbour_activities": [
                        memories[int(i)]["activity_label"] for i in chosen
                    ],
                }
            )

    collected = np.vstack([
        np.asarray(queries),
        np.asarray(neighbours),
        np.asarray(correct_endings),
        np.asarray(wrong_endings),
        np.asarray(navigation_targets),
    ])
    n_query = len(queries)
    n_neighbour = len(neighbours)
    n_correct = len(correct_endings)
    n_wrong = len(wrong_endings)
    n_nav = len(navigation_targets)
    print(f"\ncollected {collected.shape[0]} vectors: "
          f"{n_query} queries + {n_neighbour} neighbours + {n_correct} correct + "
          f"{n_wrong} wrong + {n_nav} navigation targets")

    import umap  # imported here so the dependency requirement is explicit

    n_neighbors = max(2, min(UMAP_N_NEIGHBORS, collected.shape[0] - 2))
    print(f"running UMAP ({collected.shape[1]}-d -> 2-d) "
          f"[n_neighbors={n_neighbors}, min_dist={UMAP_MIN_DIST}, metric={UMAP_METRIC}] ...")
    embedding = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
    ).fit_transform(collected)

    cursor = 0
    query_2d = embedding[cursor : cursor + n_query]
    cursor += n_query
    neighbour_2d = embedding[cursor : cursor + n_neighbour]
    cursor += n_neighbour
    correct_2d = embedding[cursor : cursor + n_correct]
    cursor += n_correct
    wrong_2d = embedding[cursor : cursor + n_wrong]
    cursor += n_wrong
    navigation_2d = embedding[cursor : cursor + n_nav]

    for position, record in enumerate(per_example):
        record["query_2d"] = query_2d[position].tolist()
        record["navigation_target_2d"] = navigation_2d[position].tolist()
        print(f"  [{position}] val {record['val_index']} "
              f"{record['activity_label']!r}: cos(ctx,nav)={record['cos_ctx_nav']:+.4f}, "
              f"nav gap={record['correct_ending_gap_on_nav']:+.4f}, "
              f"neighbours={record['neighbour_activities']}")

    # -------------------------------------------------------------- plot
    from matplotlib.lines import Line2D  # local: only needed for the legend

    fig, ax = plt.subplots(figsize=(13, 10))

    ax.scatter(
        wrong_2d[:, 0], wrong_2d[:, 1], s=55, c="lightgrey", edgecolors="none",
        label=f"wrong endings (n={n_wrong})", zorder=2,
    )
    ax.scatter(
        neighbour_2d[:, 0], neighbour_2d[:, 1], s=80, c="grey",
        edgecolors="white", linewidths=0.6,
        label=f"top-{FUSE_TOP_K} memory neighbour positions (n={n_neighbour})", zorder=3,
    )
    ax.scatter(
        correct_2d[:, 0], correct_2d[:, 1], s=150, c="green",
        edgecolors="black", linewidths=0.7,
        label=f"correct ending vectors (n={n_correct})", zorder=4,
    )
    ax.scatter(
        query_2d[:, 0], query_2d[:, 1], s=170, c="red",
        edgecolors="black", linewidths=0.7,
        label=f"query positions (n={n_query})", zorder=5,
    )

    # navigation direction: an ARROW from the query, not a landing marker
    for position in range(n_query):
        ax.annotate(
            "",
            xy=navigation_2d[position],
            xytext=query_2d[position],
            arrowprops=dict(arrowstyle="->", color="blue", lw=1.7, alpha=0.85),
            zorder=6,
        )

    for position in range(n_query):
        ax.annotate(
            f"q{position}", xy=query_2d[position], xytext=(6, 6),
            textcoords="offset points", fontsize=9, color="darkred", zorder=7,
        )
        ax.annotate(
            f"*{position}", xy=correct_2d[position], xytext=(6, -12),
            textcoords="offset points", fontsize=9, color="darkgreen", zorder=7,
        )

    handles, legend_labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D([0], [0], color="blue", lw=1.7, marker=">", markersize=7,
               label="navigation direction")
    )
    legend_labels.append("navigation direction (query -> query + nav_vector)")

    ax.set_title("Geometric Navigation in Latent Space", fontsize=16, fontweight="bold",
                 pad=26)
    ax.text(
        0.5, 1.015,
        "5 val examples | UMAP of 256-d latents | arrows show the navigation "
        "direction (the additive model never moves the query position)",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=10,
        style="italic", color="dimgray",
    )
    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)
    ax.grid(alpha=0.15, linewidth=0.5)
    ax.legend(handles=handles, labels=legend_labels, loc="best", fontsize=9,
              frameon=True, framealpha=0.9)

    # reference the full-set gap so a 5-example sample cannot look authoritative
    reference_gap = None
    additive_results = PROJECT_ROOT / "experiments" / "results_additive.json"
    if additive_results.exists():
        with additive_results.open(encoding="utf-8") as handle:
            reference_gap = json.load(handle)["gap_terms"]["gap_navigation"]

    summary = (
        f"cos(ctx, nav) = "
        f"{np.mean([r['cos_ctx_nav'] for r in per_example]):+.4f} "
        f"(tangent projection)\n"
        f"correct-ending gap on nav, these 5: "
        f"{np.mean([r['correct_ending_gap_on_nav'] for r in per_example]):+.4f}"
    )
    if reference_gap is not None:
        summary += f"\nfull 10,042-example value: {reference_gap:+.4f}"
    ax.text(
        0.995, 0.005, summary, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="lightgrey",
                              alpha=0.9),
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"\nSaved plot: {OUTPUT_PATH} ({size_kb:.0f} KB, dpi={DPI})")
    print(f"  points: {collected.shape[0]} | UMAP {embedding.shape}")
    print("Navigation visualization complete.")


if __name__ == "__main__":
    main()
