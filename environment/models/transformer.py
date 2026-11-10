"""Shared transformer trunk used by BOTH V3 architectures (TASK 20).

The whole point of this module is that both models are built from the *same*
code path, so the parameter budgets match by construction and architecture stops
being a confound.  A single boolean threads through everything:

    causal=True    GPT-2 style — decoder, strictly lower-triangular attention
    causal=False   GMN style   — encoder, dense bidirectional attention

That flag is the ONLY architectural difference (TASK 20 note 1).  FeedForward,
TransformerBlock and TransformerBase are shared verbatim; only the two
task-specific heads live in ``gpt2_style.py`` / ``gmn_style.py``.

Contents
--------
``MultiHeadAttention``  scaled dot-product self-attention, ``causal`` switch
``FeedForward``         Linear -> GELU -> Linear -> dropout
``TransformerBlock``    pre-LayerNorm attention + FFN with residuals
``TransformerBase``     token + LEARNED positional embeddings, N blocks, final LN
``resolve_model_config``   validates an arbitrary config dict (TASK 19R)
``count_unique_parameters`` / ``parameter_breakdown``  reporting helpers
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ``environment/``, ``environment/models/`` and ``environment/tokenizer/`` are
# not package roots, so expose them on sys.path (TASK 18/19 convention) before
# importing siblings by bare module name.
MODELS_DIR = Path(__file__).resolve().parent
ENVIRONMENT_DIR = MODELS_DIR.parent
TOKENIZER_DIR = ENVIRONMENT_DIR / "tokenizer"
for _directory in (ENVIRONMENT_DIR, MODELS_DIR, TOKENIZER_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import config  # noqa: E402

#: Mask value used to silence disallowed attention positions.  Softmax turns
#: ``exp(-1e9)`` into exactly ``0.0`` in float32, which is what lets
#: ``test_models.py`` assert that future positions receive exactly zero weight.
MASK_VALUE = -1e9

#: The one architectural switch, named so both model files read it from here.
CAUSAL_GPT2 = True
CAUSAL_GMN = False

#: GPT-2's weight-init standard deviation.
INIT_STD = 0.02

#: Keys a model config dict must provide (TASK 19R).
REQUIRED_CONFIG_KEYS: Tuple[str, ...] = (
    "n_layers",
    "n_heads",
    "hidden_dim",
    "ffn_dim",
    "vocab_size",
    "max_seq_len",
)

#: Dropout used when a config dict leaves it out.
DEFAULT_DROPOUT = 0.1


def resolve_model_config(config_dict: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Return a validated copy of a model config dict.

    ``None`` means "whatever :data:`config.MODEL` currently is", which keeps the
    ARC-era ``TransformerBase.from_config(causal=...)`` calls of TASK 20 working.
    The WikiText trainers instead pass an explicit dict — ``config.MODEL_SMALL``,
    ``config.MODEL_MEDIUM``, or a name through :func:`config.model_config`.

    Raises
    ------
    KeyError
        a required dimension is missing.
    ValueError
        ``hidden_dim`` is not divisible by ``n_heads``.
    """
    source: Mapping[str, Any] = config.MODEL if config_dict is None else config_dict
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in source]
    if missing:
        raise KeyError(f"model config is missing {missing}; got keys {sorted(source)}")

    resolved: Dict[str, Any] = dict(source)
    resolved.setdefault("dropout", DEFAULT_DROPOUT)

    hidden_dim = int(resolved["hidden_dim"])
    n_heads = int(resolved["n_heads"])
    if hidden_dim % n_heads != 0:
        raise ValueError(f"hidden_dim {hidden_dim} not divisible by n_heads {n_heads}")
    return resolved


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """Scaled dot-product self-attention with an optional causal mask.

    ``causal`` is the single architectural switch that separates the two V3
    models:

    * ``causal=True``  — every position strictly above the diagonal is filled
      with :data:`MASK_VALUE` before the softmax, so token ``t`` can only attend
      to tokens ``<= t`` (GPT-2 style, decoder).
    * ``causal=False`` — no mask at all; attention is dense over all positions
      (GMN style, encoder).

    Nothing else in this class changes between the two models.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        dropout: float = 0.1,
        causal: bool = False,
        no_peek: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim {hidden_dim} not divisible by n_heads {n_heads}")

        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(n_heads)
        self.head_dim = int(hidden_dim) // int(n_heads)
        self.causal = bool(causal)
        self.no_peek = bool(no_peek)
        if self.causal and self.no_peek:
            raise ValueError("causal and no_peek are mutually exclusive attention modes")

        # Q, K, V projections through a single fused weight is faster, but the
        # task asks for explicit projections and they are easier to inspect.
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Dropout on the attention weights, as specified.
        self.attn_dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------ utils
    def _split_heads(self, x: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        """``(B, T, H)`` -> ``(B, n_heads, T, head_dim)``."""
        return x.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    @staticmethod
    def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Boolean ``(T, T)`` mask, ``True`` strictly above the diagonal.

        Exactly the construction from TASK 20 note 2 (only ``device`` added so
        it works on GPU/CPU alike).
        """
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    @staticmethod
    def superdiagonal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Boolean ``(T, T)`` mask, ``True`` only on the superdiagonal (key = query + 1).

        The "no-peek" mask for a bidirectional next-token LM: query ``t`` may
        attend to every position **except** ``t + 1`` — the token it is about to
        predict.  Every other past and future position stays visible, so the model
        keeps full bidirectional context without being able to read the answer out
        of its own input.
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        diagonal = torch.arange(max(0, seq_len - 1), device=device)
        mask[diagonal, diagonal + 1] = True
        return mask

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """``x`` (B, T, H) -> (B, T, H), plus attention maps when requested.

        ``key_padding_mask`` is ``(B, T)`` bool, ``True`` at padded *keys*.  It is
        optional, OFF by default, and never enabled for one model and not the
        other — it is an input mask, not the causal-vs-bidirectional difference.
        """
        batch_size, seq_len, _ = x.shape

        q = self._split_heads(self.q_proj(x), batch_size, seq_len)
        k = self._split_heads(self.k_proj(x), batch_size, seq_len)
        v = self._split_heads(self.v_proj(x), batch_size, seq_len)

        # Scaled dot-product scores: (B, n_heads, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if self.causal:
            scores = scores.masked_fill(self.causal_mask(seq_len, x.device), MASK_VALUE)

        if self.no_peek:
            scores = scores.masked_fill(self.superdiagonal_mask(seq_len, x.device), MASK_VALUE)

        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], MASK_VALUE)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        out = self.out_proj(context)

        if need_weights:
            return out, attn_weights
        return out


# ---------------------------------------------------------------------------
# Feed-forward and block
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    """Position-wise MLP: ``Linear(H, F) -> GELU -> Linear(F, H) -> dropout``.

    Shared by both models with no flags at all — identical in every respect.
    """

    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Pre-LayerNorm transformer block.

    ``x = x + attn(ln1(x))`` then ``x = x + ffn(ln2(x))``.  Pre-LN is used
    because it trains more stably from scratch than the original post-LN layout.

    The class is identical for both models; the only thing that varies is the
    ``causal`` flag handed to its :class:`MultiHeadAttention`.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        causal: bool = False,
        no_peek: bool = False,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadAttention(
            hidden_dim, n_heads, dropout=dropout, causal=causal, no_peek=no_peek
        )
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, ffn_dim, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if need_weights:
            attn_out, weights = self.attn(
                self.ln1(x), key_padding_mask=key_padding_mask, need_weights=True
            )
        else:
            attn_out, weights = self.attn(
                self.ln1(x), key_padding_mask=key_padding_mask, need_weights=False
            ), None

        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return (x, weights) if need_weights else x


# ---------------------------------------------------------------------------
# Trunk
# ---------------------------------------------------------------------------
class TransformerBase(nn.Module):
    """The shared trunk: embeddings + N blocks + final LayerNorm.

    ``forward(token_ids)`` returns ``(B, T, hidden_dim)`` hidden states.  The
    ``causal`` flag selects the attention type and nothing else.

    Positional information uses **learned** embeddings
    (``nn.Embedding(max_seq_len, hidden_dim)``), not sinusoidal encodings, and
    both models use this same class.
    """

    def __init__(
        self,
        vocab_size: int = int(config.MODEL["vocab_size"]),
        n_layers: int = int(config.MODEL["n_layers"]),
        n_heads: int = int(config.MODEL["n_heads"]),
        hidden_dim: int = int(config.MODEL["hidden_dim"]),
        ffn_dim: int = int(config.MODEL["ffn_dim"]),
        max_seq_len: int = int(config.MODEL["max_seq_len"]),
        dropout: float = float(config.MODEL["dropout"]),
        causal: bool = False,
        no_peek: bool = False,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.hidden_dim = int(hidden_dim)
        self.ffn_dim = int(ffn_dim)
        self.max_seq_len = int(max_seq_len)
        self.dropout = float(dropout)
        self.causal = bool(causal)
        self.no_peek = bool(no_peek)

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        # LEARNED positional embeddings (TASK 20 note 3) — not sinusoidal.
        self.positional_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.embed_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            TransformerBlock(hidden_dim, n_heads, ffn_dim, dropout=dropout, causal=causal,
                             no_peek=no_peek)
            for _ in range(n_layers)
        )
        self.ln_final = nn.LayerNorm(hidden_dim)

        self.apply(self._init_weights)
        self._rescale_residual_projections()

    # ------------------------------------------------------------ construction
    @classmethod
    def from_config(
        cls,
        causal: bool,
        config_dict: Optional[Mapping[str, Any]] = None,
        no_peek: bool = False,
    ) -> "TransformerBase":
        """Build a trunk from a model config dict (TASK 19R).

        Both models construct their trunk through this one classmethod, so the
        hyperparameters cannot drift apart.  ``config_dict=None`` falls back to
        :data:`config.MODEL`; the WikiText trainers pass ``config.MODEL_SMALL`` or
        ``config.MODEL_MEDIUM`` explicitly.

        ``no_peek=True`` (with ``causal=False``) blocks only the superdiagonal, the
        bidirectional-LM mask used by :class:`gmn_style.GMNStyleLM`.
        """
        resolved = resolve_model_config(config_dict)
        return cls(
            vocab_size=int(resolved["vocab_size"]),
            n_layers=int(resolved["n_layers"]),
            n_heads=int(resolved["n_heads"]),
            hidden_dim=int(resolved["hidden_dim"]),
            ffn_dim=int(resolved["ffn_dim"]),
            max_seq_len=int(resolved["max_seq_len"]),
            dropout=float(resolved["dropout"]),
            causal=causal,
            no_peek=no_peek,
        )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """GPT-2 style init: N(0, 0.02) weights, zero biases, identity LayerNorm."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _rescale_residual_projections(self) -> None:
        """Scale the two residual-output projections by ``1/sqrt(2 * n_layers)``.

        GPT-2's trick for keeping activation variance stable with depth; it
        changes no parameter counts.
        """
        std = INIT_STD / math.sqrt(2 * self.n_layers)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and (
                name.endswith("out_proj") or name.endswith("fc2")
            ):
                nn.init.normal_(module.weight, mean=0.0, std=std)


    # ---------------------------------------------------------------- forward
    def forward(
        self,
        token_ids: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """``token_ids`` (B, T) -> hidden states (B, T, hidden_dim)."""
        if token_ids.dim() != 2:
            raise ValueError(f"token_ids must be (B, T), got {tuple(token_ids.shape)}")

        batch_size, seq_len = token_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )
        del batch_size

        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        x = self.token_embedding(token_ids) + self.positional_embedding(positions)
        x = self.embed_dropout(x)

        weights: List[torch.Tensor] = []
        for block in self.blocks:
            if need_weights:
                x, block_weights = block(
                    x, key_padding_mask=key_padding_mask, need_weights=True
                )
                weights.append(block_weights)
            else:
                x = block(x, key_padding_mask=key_padding_mask, need_weights=False)

        x = self.ln_final(x)
        return (x, weights) if need_weights else x

    # -------------------------------------------------------------- reporting
    def attention_maps(self, token_ids: torch.Tensor) -> List[torch.Tensor]:
        """Per-layer attention maps ``(B, n_heads, T, T)``, eval mode, no dropout."""
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                _, weights = self(token_ids, need_weights=True)
        finally:
            self.train(was_training)
        return weights

    def count_parameters(self) -> int:
        return count_unique_parameters(self)


# ---------------------------------------------------------------------------
# Parameter reporting helpers
# ---------------------------------------------------------------------------
def count_unique_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters, deduplicating shared/tied weights.

    A tied weight appears twice in ``named_parameters`` (once as
    ``token_embedding.weight`` and once as ``lm_head.weight``) but is one
    tensor, so it must be counted once.
    """
    seen: Set[int] = set()
    total = 0
    for parameter in module.parameters():
        if id(parameter) in seen:
            continue
        if trainable_only and not parameter.requires_grad:
            continue
        seen.add(id(parameter))
        total += parameter.numel()
    return total


def parameter_breakdown(module: nn.Module, depth: int = 1) -> Dict[str, int]:
    """Unique parameter counts grouped by the first ``depth`` name components."""
    totals: Dict[str, int] = {}
    seen: Set[int] = set()
    for name, parameter in module.named_parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        key = ".".join(name.split(".")[:depth])
        totals[key] = totals.get(key, 0) + parameter.numel()
    return dict(sorted(totals.items()))


if __name__ == "__main__":  # tiny smoke test; full checks live in test_models.py
    from bpe import BOS_ID, EOS_ID

    for causal in (CAUSAL_GPT2, CAUSAL_GMN):
        trunk = TransformerBase.from_config(causal=causal)
        dummy = torch.tensor([[BOS_ID, 100, 200, EOS_ID]])
        hidden = trunk(dummy)
        print(
            f"causal={causal!s:<5s}  hidden={tuple(hidden.shape)}  "
            f"params={trunk.count_parameters():,}  "
            f"attn_map={tuple(trunk.attention_maps(dummy)[0].shape)}"
        )
