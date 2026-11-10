"""GPT-2 style baseline: decoder-only transformer, next-token prediction (TASK 20/19R).

Thin wrapper over the shared trunk in ``transformer.py``: it adds an LM head and
nothing else, so its only architectural difference from the GMN model is the
``causal=True`` flag it hands to the trunk.

Hyper-parameters come from a **config dict** (TASK 19R): pass
``config.MODEL_SMALL`` / ``config.MODEL_MEDIUM`` (or a name, via
``config.model_config("medium")``).  ``config_dict=None`` falls back to
``config.MODEL``.

Weight tying (TASK 20 note 4): ``lm_head.weight`` **is** the token-embedding
matrix, which is standard GPT-2 practice and avoids a duplicate
``vocab_size x hidden_dim`` parameter block (16000 x 256 = 4,096,000 here, i.e.
about half of the TASK 19R SMALL model).  The tied tensor is counted once by
:func:`transformer.count_unique_parameters`.

    forward(token_ids) -> logits (B, T, vocab_size)
    loss(token_ids)    -> cross-entropy on next-token prediction, [PAD] ignored
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

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
from bpe import PAD_ID  # noqa: E402

from transformer import (  # noqa: E402
    CAUSAL_GPT2,
    TransformerBase,
    count_unique_parameters,
    parameter_breakdown,
    resolve_model_config,
)


class GPT2Style(nn.Module):
    """``TransformerBase(causal=True)`` + a tied LM head.

    Parameters
    ----------
    config_dict : mapping, optional
        Model dimensions (``n_layers``, ``n_heads``, ``hidden_dim``, ``ffn_dim``,
        ``vocab_size``, ``max_seq_len``, optional ``dropout``).  ``None`` uses
        ``config.MODEL``.
    tie_weights : bool
        Share ``lm_head.weight`` with the token embedding (default, GPT-2 style).
    vocab_size, hidden_dim : int, optional
        Single-field overrides applied on top of ``config_dict``.
    """

    def __init__(
        self,
        config_dict: Optional[Mapping[str, Any]] = None,
        tie_weights: bool = True,
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

        # causal=True -> strictly lower-triangular attention.  Built via the same
        # classmethod the GMN model uses, so hyperparameters cannot drift.
        self.trunk = TransformerBase.from_config(causal=CAUSAL_GPT2, config_dict=self.config)

        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)
        if self.tie_weights:
            self.lm_head.weight = self.trunk.token_embedding.weight
        else:  # kept only so the size difference can be demonstrated
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

        #: The number of parameters tying saves — reported by the model tests.
        self.tied_parameter_saving = (
            self.vocab_size * self.hidden_dim if self.tie_weights else 0
        )

    @classmethod
    def from_model_name(cls, name: str = "small", **kwargs: Any) -> "GPT2Style":
        """Build directly from a named config: ``from_model_name("medium")``."""
        return cls(config_dict=config.model_config(name), **kwargs)


    # ---------------------------------------------------------------- forward
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """``token_ids`` (B, T) -> logits (B, T, vocab_size)."""
        hidden = self.trunk(token_ids)
        return self.lm_head(hidden)

    def hidden_states(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Trunk output before the LM head, ``(B, T, hidden_dim)``."""
        return self.trunk(token_ids)

    def loss(
        self,
        token_ids: torch.Tensor,
        ignore_index: int = PAD_ID,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Next-token cross-entropy: logits[:, :-1] against targets[:, 1:].

        ``[PAD]`` targets are ignored, which matters because the tokenizer pads
        every sequence out to ``max_seq_len``.
        """
        logits = self(token_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = token_ids[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_targets.view(-1),
            ignore_index=ignore_index,
            reduction=reduction,
        )

    @torch.no_grad()
    def perplexity(self, token_ids: torch.Tensor, ignore_index: int = PAD_ID) -> float:
        """``exp(loss)`` over the batch, a readable training-progress number."""
        value = self.loss(token_ids, ignore_index=ignore_index, reduction="mean")
        return float(torch.exp(value))

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
            "name": "GPT-2 style",
            "config": dict(self.config),
            "causal": bool(self.trunk.causal),
            "tie_weights": self.tie_weights,
            "tied_parameter_saving": self.tied_parameter_saving,
            "params": self.count_parameters(),
            "breakdown": self.parameter_breakdown(),
        }


if __name__ == "__main__":
    from bpe import BOS_ID, EOS_ID

    for label in ("small", "medium"):
        model = GPT2Style.from_model_name(label).eval()
        cfg = model.config
        ids = torch.full((1, cfg["max_seq_len"]), 100, dtype=torch.long)
        ids[0, 0] = BOS_ID
        ids[0, -1] = EOS_ID
        ids[0, -2] = PAD_ID
        with torch.no_grad():
            logits = model(ids)
            loss = float(model.loss(ids))
        print(
            f"{label:<6s} layers={cfg['n_layers']:<3d} hidden={cfg['hidden_dim']:<4d} "
            f"vocab={cfg['vocab_size']:<6d} seq={cfg['max_seq_len']:<3d} "
            f"params={model.count_parameters():>10,}  logits={tuple(logits.shape)}  "
            f"loss={loss:.4f}"
        )
        del model
