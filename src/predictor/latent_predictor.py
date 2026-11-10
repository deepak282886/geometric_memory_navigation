"""TASK 27: the latent predictor - predict the next POSITION, not the next token.

The VAE (``checkpoints/vae.pt``) maps text to a 256-d unit vector on a semantic
manifold. This module adds the model that walks forward on that manifold:

    P_current (256)          position of the sentence just read
    C         (256)          context from dynamic memory (fused neighbour
                             directions - ``src/fusion/fuse.py``)
    P_predicted = predictor(P_current, C)          <- this module
    P_actual    = encode(next sentence)            <- the VAE, frozen

Architecture (as specified by TASK 27)::

    concat([P_current, C])  (512)
      -> Linear(512, 512) -> LayerNorm -> GELU
      -> Linear(512, 256) -> LayerNorm -> GELU
      -> Linear(256, 256)
      -> L2 normalise           -> 256-d unit vector

``count_parameters`` measures 461,312 parameters - the whole model is ~270x
smaller than GPT-2 small (124,439,808).

Loss (``cosine_prediction_loss``)::

    loss = 1 - cos(P_predicted, P_actual)      0 = perfect, 2 = exactly opposite

which is the natural continuous analogue of next-token cross entropy: there is
exactly one correct answer (the actual next position), so the loss is a
similarity, not a distribution over a 16,000-token vocabulary.

``top1_recall`` answers the task's second question: is the TRUE next position the
closest vector in memory to the prediction? It is a directional check that does
not depend on the predictor's output scale (all vectors are unit-norm), and it
takes an optional ``allowed_mask`` so a caller can exclude an example's own
memory entries (see ``scripts/train_latent_predictor.py``).

Nothing here imports the VAE or the memory store: the predictor is a pure
function of two 256-d vectors, which keeps it usable by TASK 29's prediction loop
and by the training script alike.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#: dimensionality of the VAE latent space (``checkpoints/vae.pt``, latent_dim=256)
LATENT_DIM = 256
#: dimensionality of the fused memory context C (same manifold as the latent)
CONTEXT_DIM = 256
#: 512 = concat(P_current, C)
INPUT_DIM = LATENT_DIM + CONTEXT_DIM
#: (Linear -> LayerNorm -> GELU) widths, followed by Linear(., latent_dim)
HIDDEN_DIMS: Tuple[int, ...] = (512, 256)
#: two unit vectors count as equal when their cosine similarity is this close to 1
RECALL_TOL = 1e-6

ArrayLike = Union[np.ndarray, torch.Tensor]


def _as_numpy(value: ArrayLike) -> np.ndarray:
    """Torch tensor / numpy array -> float64 numpy, gradients detached."""
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(value, dtype=np.float64)



class LatentPredictor(nn.Module):
    """MLP mapping ``(P_current, C)`` to the predicted next latent position.

    ``forward`` accepts ``(latent_dim,)`` / ``(context_dim,)`` vectors or whole
    batches of them, and returns a batch of L2-normalised (unit) vectors, so the
    prediction lives on the same unit sphere as the VAE latents it is compared
    against (``cos`` is then a plain dot product).

    The parameter count is deliberately tiny - the whole point of the design is
    that prediction happens in the 256-d latent space, so no token vocabulary, no
    softmax and no 16,000-way output layer are needed.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        context_dim: int = CONTEXT_DIM,
        hidden_dims: Sequence[int] = HIDDEN_DIMS,
        normalize_output: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.context_dim = int(context_dim)
        self.input_dim = self.latent_dim + self.context_dim
        self.normalize_output = bool(normalize_output)

        layers: list = []
        width = self.input_dim
        for hidden in hidden_dims:
            layers.extend(
                [nn.Linear(width, int(hidden)), nn.LayerNorm(int(hidden)), nn.GELU()]
            )
            width = int(hidden)
        layers.append(nn.Linear(width, self.latent_dim))
        self.mlp = nn.Sequential(*layers)

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        current_vector: ArrayLike,
        context_vector: ArrayLike,
    ) -> torch.Tensor:
        """``(P_current, C) -> P_predicted`` (unit vectors, shape ``(..., 256)``).

        Both inputs must have the same leading dimensions: either a single
        ``(256,)`` pair or a batch ``(B, 256)`` pair. Inputs are converted with
        ``torch.as_tensor``, so numpy arrays are accepted; the caller is
        responsible for matching the module's device (the training script keeps
        everything on CPU, like the rest of this project).
        """
        current = torch.as_tensor(current_vector, dtype=torch.float32)
        context = torch.as_tensor(context_vector, dtype=torch.float32)
        if current.shape[-1] != self.latent_dim:
            raise ValueError(
                f"current_vector has dim {current.shape[-1]}, expected {self.latent_dim}"
            )
        if context.shape[-1] != self.context_dim:
            raise ValueError(
                f"context_vector has dim {context.shape[-1]}, expected {self.context_dim}"
            )
        if current.shape[:-1] != context.shape[:-1]:
            raise ValueError(
                f"batch shape mismatch: current {tuple(current.shape[:-1])} vs "
                f"context {tuple(context.shape[:-1])}"
            )

        hidden = self.mlp(torch.cat([current, context], dim=-1))
        if self.normalize_output:
            # unit sphere: cosine similarity to the actual next position is then
            # simply the dot product used by cosine_prediction_loss
            hidden = F.normalize(hidden, dim=-1, eps=1e-8)
        return hidden

    # ------------------------------------------------------------------ utils
    def count_parameters(self) -> int:
        """Number of trainable parameters (every parameter of this module)."""
        return sum(parameter.numel() for parameter in self.parameters())

    def describe(self) -> str:
        """One-line architecture summary for logs / the DONE block."""
        widths = " -> ".join(
            f"Linear({int(hidden)}) + LayerNorm + GELU" for hidden in HIDDEN_DIMS
        )
        return (
            f"LatentPredictor(input={self.input_dim}: {widths} -> "
            f"Linear({self.latent_dim}) -> L2norm; unit-sphere output)"
        )


def cosine_prediction_loss(
    predicted: ArrayLike,
    actual: ArrayLike,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``loss = 1 - cos(predicted, actual)`` -> ``(loss, cos)``.

    Both terms are returned elementwise (shape ``(B,)`` for a batch, ``()`` for a
    single pair) so a caller can report the per-example distribution as well as
    the mean. ``0`` means a perfect prediction and ``2`` the worst possible one
    (exactly opposite directions); a random unit vector in 256-d already has an
    expected cosine of ~0, i.e. loss ~1, which is the natural reference point.

    ``1 - cos`` is used rather than ``1 - cos^2`` or MSE so the gradient vanishes
    as the prediction becomes exact (like a likelihood) and stays bounded when it
    is completely wrong.
    """
    predicted_tensor = torch.as_tensor(predicted, dtype=torch.float32)
    actual_tensor = torch.as_tensor(actual, dtype=torch.float32)
    if predicted_tensor.shape != actual_tensor.shape:
        raise ValueError(
            f"shape mismatch: predicted {tuple(predicted_tensor.shape)} vs "
            f"actual {tuple(actual_tensor.shape)}"
        )
    cos = F.cosine_similarity(predicted_tensor, actual_tensor, dim=-1)
    return 1.0 - cos, cos


def top1_recall(
    predicted: ArrayLike,
    memory_vectors: ArrayLike,
    actual: ArrayLike,
    allowed_mask: Optional[ArrayLike] = None,
    tol: float = RECALL_TOL,
) -> Union[bool, np.ndarray]:
    """Is the TRUE next position the closest memory vector to the prediction?

    ``memory_vectors`` are the vectors the experience memory holds
    (``(M, 256)``, unit-norm). For every prediction the closest memory row is
    found by cosine similarity and counted as a hit when that row *is* the true
    next position (cosine to ``actual`` within ``tol`` of 1.0).

    This measures whether the prediction is directionally correct - it can be
    satisfied long before the cosine loss in ``cosine_prediction_loss`` is small,
    and it fails when the prediction is close to the right area but not closer to
    the true continuation than to some other stored position.

    ``allowed_mask`` (``(M,)`` or ``(B, M)`` bool) restricts the search to a
    subset of the memory, which the training script uses to exclude an example's
    own entries so that a replayed example cannot be "recalled" from itself.
    Rows excluded this way score ``-inf``; an example with no allowed row is a
    miss by definition.

    Accepts a single ``(256,)`` prediction (returns ``bool``) or a batch
    ``(B, 256)`` (returns a ``(B,)`` bool array).
    """
    prediction = _as_numpy(predicted)
    memory = _as_numpy(memory_vectors)
    target = _as_numpy(actual)

    single = prediction.ndim == 1
    if single:
        prediction = prediction[None, :]
        target = target[None, :]
    if memory.ndim != 2 or memory.shape[1] != prediction.shape[1]:
        raise ValueError(
            f"memory_vectors must be (M, {prediction.shape[1]}), got {memory.shape}"
        )
    if memory.shape[0] == 0:
        return False if single else np.zeros(prediction.shape[0], dtype=bool)

    similarities = prediction @ memory.T  # (B, M) cosine similarities

    allowed: np.ndarray
    if allowed_mask is None:
        allowed = np.ones(similarities.shape[0], dtype=bool)
    else:
        mask = np.asarray(allowed_mask, dtype=bool)
        if mask.ndim == 1:
            mask = np.broadcast_to(mask, similarities.shape)
        allowed = mask.any(axis=1)
        similarities = np.where(mask, similarities, -np.inf)

    best = similarities.argmax(axis=1)
    # a hit when the closest memory row equals the true next position
    hits = ((memory[best] * target).sum(axis=1) >= 1.0 - tol) & allowed
    return bool(hits[0]) if single else hits


def predictor_config(predictor: LatentPredictor) -> dict:
    """Serialisable config for ``predictor`` (mirrors the VAE checkpoint style)."""
    widths = [
        module.out_features
        for module in predictor.mlp
        if isinstance(module, nn.Linear)
    ]
    return {
        "latent_dim": predictor.latent_dim,
        "context_dim": predictor.context_dim,
        # every Linear except the final projection back to latent_dim
        "hidden_dims": widths[:-1],
        "normalize_output": predictor.normalize_output,
    }


def save_latent_predictor(
    predictor: LatentPredictor,
    path: Union[str, Path],
    extras: Optional[dict] = None,
) -> Path:
    """Save ``predictor`` (weights + config + parameter count) to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": predictor_config(predictor),
        "model_state_dict": predictor.state_dict(),
        "parameter_count": predictor.count_parameters(),
    }
    if extras:
        payload.update(extras)
    torch.save(payload, path)
    return path


def load_latent_predictor(
    path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> LatentPredictor:
    """Rebuild a predictor saved by :func:`save_latent_predictor` (eval mode)."""
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    config = checkpoint["config"]
    predictor = LatentPredictor(
        latent_dim=config.get("latent_dim", LATENT_DIM),
        context_dim=config.get("context_dim", CONTEXT_DIM),
        hidden_dims=config.get("hidden_dims", HIDDEN_DIMS),
        normalize_output=config.get("normalize_output", True),
    ).to(device)
    missing, unexpected = predictor.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={missing} unexpected={unexpected}"
        )
    predictor.eval()
    return predictor


