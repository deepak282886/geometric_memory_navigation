"""GMN style encoder: bidirectional trunk + [CLS] projection head (TASK 20/19R).

Thin wrapper over the shared trunk in ``transformer.py``: it adds a projection
head and nothing else, so its only architectural difference from the GPT-2
baseline is the ``causal=False`` flag it hands to the trunk.

Hyper-parameters come from a **config dict** (TASK 19R): pass
``config.MODEL_SMALL`` / ``config.MODEL_MEDIUM`` (or a name, via
``config.model_config("medium")``).  ``config_dict=None`` falls back to
``config.MODEL``.

The output projection is **always** ``Linear(hidden_dim, 256)`` — the latent
space stays 256-d whatever the trunk width is, so the memory store and the
scoring paths built in TASKs 1-17 keep working and SMALL/MEDIUM stay comparable.

Pooling (TASK 20 note 5): the sentence vector is the hidden state at position 0
after prepending the ``[CLS]`` token (id 2, which is ``[BOS]``).  Mean pooling is
deliberately **not** used.

Padding (TASK 19R note 5): :meth:`encode` accepts an optional
``key_padding_mask`` and threads it through every attention block, so a
``[PAD]``-filled batch gives the same sentence vector as a trimmed batch.

    encode(token_ids)                 -> (B, 256) unit vectors
    forward(question_ids, choice_ids) -> question_vec, choice_vecs, scores
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

MODELS_DIR = Path(__file__).resolve().parent
ENVIRONMENT_DIR = MODELS_DIR.parent
TOKENIZER_DIR = ENVIRONMENT_DIR / "tokenizer"
for _directory in (ENVIRONMENT_DIR, MODELS_DIR, TOKENIZER_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import config  # noqa: E402
from bpe import BOS_ID, PAD_ID  # noqa: E402

from transformer import (  # noqa: E402
    CAUSAL_GMN,
    TransformerBase,
    count_unique_parameters,
    parameter_breakdown,
    resolve_model_config,
)

#: Latent width of the sentence vector (the old GMN's 256-d latent space).
LATENT_DIM = 256

#: The pooling token: ``[BOS]`` doubles as ``[CLS]`` (TASK 20 note 5).
CLS_TOKEN_ID = BOS_ID


class GMNStyle(nn.Module):
    """``TransformerBase(causal=False)`` + ``Linear(hidden, 256)`` + L2 norm.

    Parameters
    ----------
    config_dict : mapping, optional
        Model dimensions (``n_layers``, ``n_heads``, ``hidden_dim``, ``ffn_dim``,
        ``vocab_size``, ``max_seq_len``, optional ``dropout``).  ``None`` uses
        ``config.MODEL``.
    latent_dim : int
        Width of the projected sentence vector — fixed at 256 by design.
    projection_bias : bool
        Whether the projection head has a bias.
    vocab_size, hidden_dim : int, optional
        Single-field overrides applied on top of ``config_dict``.
    """

    def __init__(
        self,
        config_dict: Optional[Mapping[str, Any]] = None,
        latent_dim: int = LATENT_DIM,
        projection_bias: bool = True,
        vocab_size: Optional[int] = None,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = resolve_model_config(config_dict)
        if vocab_size is not None:
            self.config["vocab_size"] = int(vocab_size)
        if hidden_dim is not None:
            self.config["hidden_dim"] = int(hidden_dim)

        self.vocab_size = int(self.config["vocab_size"])
        self.hidden_dim = int(self.config["hidden_dim"])
        self.latent_dim = int(latent_dim)

        # causal=False -> dense, fully bidirectional attention.
        self.trunk = TransformerBase.from_config(causal=CAUSAL_GMN, config_dict=self.config)

        self.projection = nn.Linear(self.hidden_dim, self.latent_dim, bias=projection_bias)
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

    @classmethod
    def from_model_name(cls, name: str = "small", **kwargs: Any) -> "GMNStyle":
        """Build directly from a named config: ``from_model_name("medium")``."""
        return cls(config_dict=config.model_config(name), **kwargs)


    # --------------------------------------------------------------- encoding
    def encode(
        self,
        token_ids: torch.Tensor,
        prepend_cls: bool = True,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``token_ids`` (B, T) -> L2-normalised sentence vectors (B, latent_dim).

        With ``prepend_cls=True`` (the default) a ``[CLS]/[BOS]`` id is added at
        position 0 and the sequence is truncated to ``max_seq_len`` afterwards,
        so a caller may pass tokenizer output that is already padded and still
        get a valid input.

        ``key_padding_mask`` (B, T) is ``True`` at padded positions of the input
        *before* ``[CLS]`` is prepended; the ``[CLS]`` column is then added as a
        non-pad key.  Threading the mask through reproduces trimming exactly —
        without it ``[CLS]`` would attend to ``[PAD]`` and the sentence vector
        would be corrupted for a non-architectural reason.
        """
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.bool()
            if tuple(key_padding_mask.shape) != tuple(token_ids.shape):
                raise ValueError(
                    f"key_padding_mask {tuple(key_padding_mask.shape)} must match "
                    f"token_ids {tuple(token_ids.shape)}"
                )

        if prepend_cls:
            cls_column = torch.full(
                (token_ids.shape[0], 1),
                CLS_TOKEN_ID,
                dtype=token_ids.dtype,
                device=token_ids.device,
            )
            token_ids = torch.cat([cls_column, token_ids], dim=1)
            if key_padding_mask is not None:
                # [CLS] is a real token, never a padded key.
                not_pad = torch.zeros_like(key_padding_mask[:, :1])
                key_padding_mask = torch.cat([not_pad, key_padding_mask], dim=1)

        max_len = self.trunk.max_seq_len
        if token_ids.shape[1] > max_len:
            token_ids = token_ids[:, :max_len]
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask[:, :max_len]

        hidden = self.trunk(token_ids, key_padding_mask=key_padding_mask)
        cls_state = hidden[:, 0, :]                     # position 0 == [CLS]
        projected = self.projection(cls_state)
        return F.normalize(projected, p=2, dim=-1)      # unit vectors

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        question_ids: torch.Tensor,
        choice_ids: torch.Tensor,
        prepend_cls: bool = True,
        question_mask: Optional[torch.Tensor] = None,
        choice_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score 4 choices against a question.

        ``question_ids`` (B, T) and ``choice_ids`` (B, 4, T) in, and out:

        * ``question_vec`` (B, latent_dim) — unit vectors
        * ``choice_vecs``  (B, 4, latent_dim) — unit vectors
        * ``scores``       (B, 4) = ``choice_vecs @ question_vec``

        ``question_mask`` (B, T) / ``choice_mask`` (B, 4, T) are optional padding
        masks forwarded to :meth:`encode`.
        """
        if choice_ids.dim() != 3:
            raise ValueError(f"choice_ids must be (B, 4, T), got {tuple(choice_ids.shape)}")

        batch_size, n_choices, seq_len = choice_ids.shape
        question_vec = self.encode(
            question_ids, prepend_cls=prepend_cls, key_padding_mask=question_mask
        )

        flat_choices = choice_ids.reshape(batch_size * n_choices, seq_len)
        flat_mask = (
            choice_mask.reshape(batch_size * n_choices, seq_len)
            if choice_mask is not None
            else None
        )
        flat_vecs = self.encode(flat_choices, prepend_cls=prepend_cls, key_padding_mask=flat_mask)
        choice_vecs = flat_vecs.view(batch_size, n_choices, self.latent_dim)

        scores = torch.einsum("bd,bcd->bc", question_vec, choice_vecs)
        return question_vec, choice_vecs, scores

    # -------------------------------------------------------------- reporting
    def count_parameters(self) -> int:
        """Unique trainable parameters."""
        return count_unique_parameters(self)

    def parameter_breakdown(self) -> Dict[str, int]:
        """Per-component unique parameter counts."""
        return parameter_breakdown(self, depth=2)

    def summary(self) -> Dict[str, Any]:
        """Everything the model tests / trainers want to report."""
        return {
            "name": "GMN style",
            "config": dict(self.config),
            "causal": bool(self.trunk.causal),
            "latent_dim": self.latent_dim,
            "pooling": "[CLS] (position 0)",
            "params": self.count_parameters(),
            "breakdown": self.parameter_breakdown(),
        }


class GMNStyleLM(nn.Module):
    """``TransformerBase(causal=False)`` + a tied next-token LM head.

    The "LLM vs LLM" arm: identical to :class:`gpt2_style.GPT2Style` in every
    respect except the trunk's attention — same layers, same widths, same tied
    ``lm_head``, and therefore the same parameter count.  Data, objective,
    schedule and evaluation are shared too, so **attention direction is the only
    variable**.

    Read this before trusting a perplexity from this model
    -----------------------------------------------------
    :meth:`loss` uses the requested recipe — ``logits[:, :-1]`` against
    ``token_ids[:, 1:]`` — but slicing the **output** does not hide the answer from
    the **input**.  In a bidirectional trunk ``hidden[t]`` attends to position
    ``t + 1``, and that position *is* the token being predicted, so a perplexity
    obtained with ``peek=True`` can be arbitrarily low without the model learning
    any language.  (``test``-style proof: perturbing token ``t + 1`` changes the
    logits at position ``t``.)

    ``peek=False`` builds the trunk with ``no_peek=True``, which blocks query ``t``
    from attending to key ``t + 1`` — only the superdiagonal.  That is the literal
    meaning of "sees full context except the exact token it is predicting", and it
    is the mode a meaningful bidirectional-LM number requires.

        forward(token_ids)                 -> logits (B, T, vocab_size)
        forward(token_ids, target_ids=ids) -> (loss, logits)
        loss(token_ids)                    -> next-token CE, [PAD] ignored
    """

    def __init__(
        self,
        config_dict: Optional[Mapping[str, Any]] = None,
        tie_weights: bool = True,
        peek: bool = True,
        vocab_size: Optional[int] = None,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = resolve_model_config(config_dict)
        if vocab_size is not None:
            self.config["vocab_size"] = int(vocab_size)
        if hidden_dim is not None:
            self.config["hidden_dim"] = int(hidden_dim)

        self.vocab_size = int(self.config["vocab_size"])
        self.hidden_dim = int(self.config["hidden_dim"])
        self.tie_weights = bool(tie_weights)
        self.peek = bool(peek)

        # causal=False -> bidirectional; no_peek blocks only the superdiagonal, so
        # the model keeps full context but cannot read the token it must predict.
        self.trunk = TransformerBase.from_config(
            causal=CAUSAL_GMN, config_dict=self.config, no_peek=not self.peek
        )

        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)
        if self.tie_weights:
            self.lm_head.weight = self.trunk.token_embedding.weight
        else:  # kept only so the size difference can be demonstrated
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

        #: The number of parameters tying saves — same as the causal arm.
        self.tied_parameter_saving = (
            self.vocab_size * self.hidden_dim if self.tie_weights else 0
        )

    @classmethod
    def from_model_name(cls, name: str = "small", **kwargs: Any) -> "GMNStyleLM":
        """Build directly from a named config: ``from_model_name("medium")``."""
        return cls(config_dict=config.model_config(name), **kwargs)

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        token_ids: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """``token_ids`` (B, T) -> logits (B, T, vocab_size).

        With ``target_ids`` (B, T) the next-token loss comes back as well, as
        ``(loss, logits)``.
        """
        logits = self.lm_head(self.trunk(token_ids))
        if target_ids is None:
            return logits
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, self.vocab_size),
            target_ids[:, 1:].reshape(-1),
            ignore_index=PAD_ID,
        )
        return loss, logits

    def hidden_states(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Trunk output before the LM head, ``(B, T, hidden_dim)``."""
        return self.trunk(token_ids)

    def loss(
        self,
        token_ids: torch.Tensor,
        ignore_index: int = PAD_ID,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Next-token cross-entropy: ``logits[:, :-1]`` against ``token_ids[:, 1:]``.

        The shift is the standard one.  What differs between ``peek=True`` and
        ``peek=False`` is whether the trunk could see position ``t + 1`` while
        producing the logits at position ``t``.
        """
        logits = self.lm_head(self.trunk(token_ids))
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, self.vocab_size),
            token_ids[:, 1:].reshape(-1),
            ignore_index=ignore_index,
            reduction=reduction,
        )

    @torch.no_grad()
    def perplexity(self, token_ids: torch.Tensor, ignore_index: int = PAD_ID) -> float:
        """``exp(loss)`` over the batch, a readable training-progress number."""
        return float(torch.exp(self.loss(token_ids, ignore_index=ignore_index)))

    def peekability(self, token_ids: torch.Tensor, position: Optional[int] = None) -> Dict[str, float]:
        """Leak probe: see :func:`peekability`."""
        return peekability(self, token_ids, position=position)

    # -------------------------------------------------------------- reporting
    def count_parameters(self) -> int:
        """Unique trainable parameters (tied LM head counted once)."""
        return count_unique_parameters(self)

    def parameter_breakdown(self) -> Dict[str, int]:
        """Per-component unique parameter counts."""
        return parameter_breakdown(self, depth=2)

    def summary(self) -> Dict[str, Any]:
        """Everything the model tests / trainers want to report."""
        return {
            "name": "GMN style LM",
            "config": dict(self.config),
            "attention": "causal" if self.trunk.causal else "bidirectional",
            "no_peek": bool(self.trunk.no_peek),
            "peek": self.peek,
            "tie_weights": self.tie_weights,
            "tied_parameter_saving": self.tied_parameter_saving,
            "params": self.count_parameters(),
            "breakdown": self.parameter_breakdown(),
        }


# ---------------------------------------------------------------------------
# Leak probe, shared with GPT2Style (the causal arm)
# ---------------------------------------------------------------------------
def peekability(
    model: nn.Module,
    token_ids: torch.Tensor,
    position: Optional[int] = None,
    delta: int = 7,
) -> Dict[str, float]:
    """Measure whether a model can see the token it is asked to predict.

    A next-token loss scores ``ids[p]`` from ``logits[p - 1]``.  This changes
    ``ids[p]`` itself and reports how far ``logits[p - 1]`` moves:

    * causal trunk (GPT-2) or a ``peek=False`` GMN trunk -> exactly ``0.0``;
    * a plain bidirectional trunk -> a large value, because position ``p`` is in
      the attention window of position ``p - 1``.  That is the leak: the model can
      read the answer out of its own input instead of predicting it.

    ``delta`` is added to the token id (modulo ``vocab_size``) to make the change;
    it is clamped to a real token id.
    """
    if token_ids.dim() != 2:
        raise ValueError(f"token_ids must be (B, T), got {tuple(token_ids.shape)}")
    length = int(token_ids.shape[1])
    probe = length // 2 if position is None else int(position)
    probe = max(1, min(length - 1, probe))

    altered = token_ids.clone()
    altered[:, probe] = ((altered[:, probe] + int(delta)) % int(model.vocab_size)).clamp_min(5)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        before = model(token_ids)[:, probe - 1, :]
        after = model(altered)[:, probe - 1, :]
    model.train(was_training)

    difference = (before - after).abs()
    return {
        "probe_position": probe,
        "max_abs_change": float(difference.max()),
        "mean_abs_change": float(difference.mean()),
    }


if __name__ == "__main__":
    for label in ("small", "medium"):
        model = GMNStyle.from_model_name(label).eval()
        cfg = model.config
        seq_len = cfg["max_seq_len"]
        question = torch.randint(5, 1000, (2, seq_len - 10))
        choices = torch.randint(5, 1000, (2, 4, seq_len - 8))
        with torch.no_grad():
            question_vec, choice_vecs, scores = model(question, choices)
        print(
            f"{label:<6s} layers={cfg['n_layers']:<3d} hidden={cfg['hidden_dim']:<4d} "
            f"vocab={cfg['vocab_size']:<6d} params={model.count_parameters():>10,}  "
            f"question_vec={tuple(question_vec.shape)}  "
            f"choice_vecs={tuple(choice_vecs.shape)}  scores={tuple(scores.shape)}"
        )
        del model

    print("\nLLM arm: GMNStyleLM  vs  GPT2Style (same trunk, same tied head)")
    probe_ids = torch.tensor([[BOS_ID, 100, 200, 300, 400, 500, 600, 700]])
    for peek in (True, False):
        lm = GMNStyleLM.from_model_name("small", peek=peek).eval()
        loss, logits = lm(probe_ids, target_ids=probe_ids)
        probe = lm.peekability(probe_ids)
        print(f"  GMNStyleLM peek={peek!s:<5s} no_peek={lm.trunk.no_peek!s:<5s} "
              f"params={lm.count_parameters():>10,}  logits={tuple(logits.shape)}  "
              f"loss={float(loss):.4f}  logit change at p-1 when ids[p] changes: "
              f"{probe['max_abs_change']:.4f}")
        del lm

    from gpt2_style import GPT2Style  # noqa: E402  (only for this demonstration)

    causal = GPT2Style.from_model_name("small").eval()
    causal_probe = peekability(causal, probe_ids)
    print(f"  GPT2Style (causal)           params={causal.count_parameters():>10,}  "
          f"logit change at p-1 when ids[p] changes: "
          f"{causal_probe['max_abs_change']:.4f}")
