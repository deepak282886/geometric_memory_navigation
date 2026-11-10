"""Verification script for the TASK 20 shared transformer blocks.

Run from the repository root::

    python environment/models/test_models.py

It builds both models from ``config.MODEL``, runs dummy forward passes and
asserts the properties TASK 20 requires:

1. both models instantiate and forward without NaN/Inf;
2. correct output shapes — GPT-2 logits ``(B, T, vocab)``, GMN scores ``(B, 4)``;
3. **causal** GPT-2 attention gives *exactly* zero weight to future positions,
   and a change at position 6 cannot alter the logits at position 5;
4. **bidirectional** GMN attention gives non-zero weight to the same future
   position;
5. the two trunks are architecturally identical (same state-dict keys/shapes) —
   only the ``causal`` flag differs;
6. parameter counts are within 5%, LM-head weight tying is real, positional
   embeddings are LEARNED, GMN vectors are unit length, and ``[PAD]`` is ignored
   by the GPT-2 loss.

Exits non-zero if any check fails.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Tuple

import torch
import torch.nn as nn

MODELS_DIR = Path(__file__).resolve().parent
ENVIRONMENT_DIR = MODELS_DIR.parent
TOKENIZER_DIR = ENVIRONMENT_DIR / "tokenizer"
for _directory in (ENVIRONMENT_DIR, MODELS_DIR, TOKENIZER_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import config  # noqa: E402
from bpe import BOS_ID, EOS_ID, PAD_ID  # noqa: E402

from gmn_style import LATENT_DIM, GMNStyle  # noqa: E402
from gpt2_style import GPT2Style  # noqa: E402
from transformer import TransformerBase  # noqa: E402

BATCH_SIZE = 2
SEQ_LEN = 16

#: Position pair used for the attention-direction checks (TASK 20 note 7).
QUERY_POS = 5
FUTURE_POS = 6


class Report:
    """Collects PASS/FAIL lines and decides the exit code."""

    def __init__(self) -> None:
        self.checks: List[Tuple[str, bool, str]] = []

    def check(self, label: str, ok: bool, detail: str = "") -> bool:
        self.checks.append((label, bool(ok), detail))
        mark = "PASS" if ok else "FAIL"
        suffix = f"  ({detail})" if detail else ""
        print(f"  [{mark}] {label}{suffix}")
        return bool(ok)

    @property
    def passed(self) -> bool:
        return all(ok for _, ok, _ in self.checks)

    @property
    def n_failed(self) -> int:
        return sum(1 for _, ok, _ in self.checks if not ok)


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * 74)


def make_dummy_batch() -> torch.Tensor:
    """Deterministic dummy token ids: [BOS] x4 [EOS] without using [PAD]."""
    generator = torch.Generator().manual_seed(int(config.TRAIN["seed"]))
    body = torch.randint(5, 1000, (BATCH_SIZE, SEQ_LEN - 3), generator=generator)
    bos = torch.full((BATCH_SIZE, 1), BOS_ID, dtype=torch.long)
    eos = torch.full((BATCH_SIZE, 2), EOS_ID, dtype=torch.long)
    return torch.cat([bos, body, eos], dim=1)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_shapes(report: Report, gpt2: GPT2Style, gmn: GMNStyle, ids: torch.Tensor):
    section("Output shapes")
    with torch.no_grad():
        logits = gpt2(ids)
        questions = ids
        choices = ids.unsqueeze(1).repeat(1, 4, 1)
        question_vec, choice_vecs, scores = gmn(questions, choices)

    print(f"  GPT-2 logits       : {tuple(logits.shape)}")
    print(f"  GMN question_vec   : {tuple(question_vec.shape)}")
    print(f"  GMN choice_vecs    : {tuple(choice_vecs.shape)}")
    print(f"  GMN scores         : {tuple(scores.shape)}")

    report.check(
        "GPT-2 logits shape == (batch, seq, vocab)",
        tuple(logits.shape) == (BATCH_SIZE, SEQ_LEN, int(config.MODEL["vocab_size"])),
        f"{tuple(logits.shape)}",
    )
    report.check(
        "GMN scores shape == (batch, 4)",
        tuple(scores.shape) == (BATCH_SIZE, 4),
        f"{tuple(scores.shape)}",
    )
    report.check(
        f"GMN question/choice vectors are ({BATCH_SIZE}, {LATENT_DIM}) / "
        f"({BATCH_SIZE}, 4, {LATENT_DIM})",
        tuple(question_vec.shape) == (BATCH_SIZE, LATENT_DIM)
        and tuple(choice_vecs.shape) == (BATCH_SIZE, 4, LATENT_DIM),
    )
    return logits, question_vec, choice_vecs, scores


def check_forward_consistency(
    report: Report,
    gpt2: GPT2Style,
    gmn: GMNStyle,
    ids: torch.Tensor,
    logits: torch.Tensor,
    scores: torch.Tensor,
    question_vec: torch.Tensor,
) -> None:
    section("Numerical health and forward-pass consistency")
    report.check("GPT-2 logits are all finite", bool(torch.isfinite(logits).all()))
    report.check("GMN scores are all finite", bool(torch.isfinite(scores).all()))
    report.check(
        "GMN vectors are L2-normalised (unit length)",
        bool(torch.allclose(question_vec.norm(dim=-1), torch.ones(BATCH_SIZE), atol=1e-5)),
        f"norms={[round(v, 5) for v in question_vec.norm(dim=-1).tolist()]}",
    )
    report.check(
        "GMN scores are cosine similarities in [-1, 1]",
        bool((scores.abs() <= 1.0 + 1e-5).all()),
        f"range=[{float(scores.min()):.4f}, {float(scores.max()):.4f}]",
    )

    choices = ids.unsqueeze(1).repeat(1, 4, 1)
    with torch.no_grad():
        batched = gpt2(ids)
        single = gpt2(ids[:1])
        encode_batched = gmn.encode(ids)
        encode_single = gmn.encode(ids[:1])
        _, choice_vecs, _ = gmn(ids, choices)

    report.check(
        "GPT-2 is deterministic: two eval-mode passes are bitwise identical",
        bool(torch.equal(batched, gpt2(ids))),
    )
    # Row 0 of a batch must equal row 0 run alone — a genuine batch-invariance
    # check (comparing gpt2(ids) against a *repeated* row 0 would be comparing
    # against row 1's tokens as well).
    report.check(
        "GPT-2 is batch-invariant: gpt2(ids)[:1] == gpt2(ids[:1])",
        bool(torch.equal(batched[:1], single)),
    )
    report.check(
        "GMN encode is batch-invariant: encode(ids)[:1] == encode(ids[:1])",
        bool(torch.allclose(encode_batched[:1], encode_single, atol=1e-6, rtol=0)),
        f"max diff = {float((encode_batched[:1] - encode_single).abs().max()):.2e}",
    )

    # forward() flattens (B, 4, T) -> (B*4, T); each choice must score exactly as
    # it would on its own.
    with torch.no_grad():
        per_choice_diff = max(
            float((choice_vecs[b, c] - gmn.encode(choices[b, c])[0]).abs().max())
            for b in range(BATCH_SIZE) for c in range(choices.shape[1])
        )
    report.check(
        "GMN choice vecs from forward() match encode() on the same choice",
        per_choice_diff < 1e-6,
        f"max diff = {per_choice_diff:.2e}",
    )
    report.check(
        "GMN scores equal choice_vecs @ question_vec",
        bool(torch.allclose(
            scores,
            torch.einsum("bd,bcd->bc", question_vec, choice_vecs),
            atol=1e-6, rtol=0,
        )),
    )



def check_parameter_counts(report: Report, gpt2: GPT2Style, gmn: GMNStyle) -> None:
    section("Parameter counts")
    gpt2_params = gpt2.count_parameters()
    gmn_params = gmn.count_parameters()
    difference = gmn_params - gpt2_params
    relative = 100.0 * abs(difference) / gpt2_params

    print(f"  GPT-2 style parameters : {gpt2_params:>12,}")
    print(f"  GMN style parameters   : {gmn_params:>12,}")
    print(f"  difference             : {difference:>12,}  ({relative:.2f}% of GPT-2)")
    print(f"  GPT-2 breakdown        : {gpt2.parameter_breakdown()}")
    print(f"  GMN breakdown          : {gmn.parameter_breakdown()}")
    print(f"  LM-head tying saves    : {gpt2.tied_parameter_saving:>12,} "
          f"({gpt2.vocab_size} x {gpt2.hidden_dim})")

    report.check(
        "parameter difference is within 5%",
        relative < 5.0,
        f"{relative:.2f}%",
    )
    report.check(
        "GPT-2 LM head weight IS the token embedding",
        gpt2.lm_head.weight is gpt2.trunk.token_embedding.weight,
    )
    untied = GPT2Style(tie_weights=False).count_parameters()
    print(f"  GPT-2 if UNTIED        : {untied:>12,}  "
          f"(+{untied - gpt2_params:,} = vocab_size x hidden_dim)")
    report.check(
        "weight tying removes exactly vocab_size x hidden_dim params",
        untied - gpt2_params == gpt2.vocab_size * gpt2.hidden_dim,
        f"{untied - gpt2_params:,}",
    )


def check_causal_masking(report: Report, gpt2: GPT2Style, ids: torch.Tensor) -> None:
    section(f"Causal masking — can position {QUERY_POS} see position {FUTURE_POS}?")
    maps = gpt2.trunk.attention_maps(ids)          # list of (B, H, T, T)
    layer0 = maps[0]

    future_weight = layer0[:, :, QUERY_POS, FUTURE_POS:]
    past_weight = layer0[:, :, QUERY_POS, : QUERY_POS + 1]
    print(f"  layer-0 weights at query {QUERY_POS}: "
          f"positions <= {QUERY_POS} sum to {float(past_weight.sum(-1).mean()):.6f}, "
          f"positions >= {FUTURE_POS} sum to {float(future_weight.sum(-1).mean()):.6f}")

    report.check(
        f"position {QUERY_POS} gives EXACTLY 0 weight to every position > {QUERY_POS}",
        bool((future_weight == 0).all()),
        f"max future weight = {float(future_weight.abs().max())}",
    )
    report.check(
        f"position {QUERY_POS} does attend to positions 0..{QUERY_POS}",
        bool((past_weight > 0).all()),
        f"min past weight = {float(past_weight.min()):.6f}",
    )

    upper = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool), diagonal=1)
    report.check(
        "every layer is strictly lower-triangular",
        all(bool((layer[..., upper] == 0).all()) for layer in maps),
        f"{len(maps)} layers checked",
    )
    row_sums = torch.stack([layer.sum(-1) for layer in maps])
    report.check(
        "attention rows still sum to 1 (no fully-masked position)",
        bool(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)),
        f"min row sum = {float(row_sums.min()):.6f}",
    )

    # Behavioural proof, stronger than inspecting the mask: perturbing a token
    # can only influence logits at that position or later.
    altered = ids.clone()
    altered[:, FUTURE_POS] = (int(ids[0, FUTURE_POS]) + 1) % 1000 + 5
    with torch.no_grad():
        before = gpt2(ids)
        after = gpt2(altered)
    report.check(
        f"changing token {FUTURE_POS} leaves logits at positions <= {QUERY_POS} unchanged",
        bool(torch.allclose(before[:, :FUTURE_POS], after[:, :FUTURE_POS], atol=1e-6)),
    )
    report.check(
        f"changing token {FUTURE_POS} DOES change logits at positions >= {FUTURE_POS}",
        not bool(torch.allclose(before[:, FUTURE_POS:], after[:, FUTURE_POS:], atol=1e-6)),
    )


def check_bidirectional_attention(report: Report, gmn: GMNStyle, ids: torch.Tensor) -> None:
    section(f"Bidirectional attention — can position {QUERY_POS} see position {FUTURE_POS}?")
    maps = gmn.trunk.attention_maps(ids)
    layer0 = maps[0]

    future_weight = layer0[:, :, QUERY_POS, FUTURE_POS:]
    print(f"  layer-0 weights at query {QUERY_POS}: "
          f"position {FUTURE_POS} = {float(layer0[0, 0, QUERY_POS, FUTURE_POS]):.6f}, "
          f"positions >= {FUTURE_POS} sum to {float(future_weight.sum(-1).mean()):.6f}")

    report.check(
        f"position {QUERY_POS} gives NON-ZERO weight to position {FUTURE_POS}",
        bool(layer0[:, :, QUERY_POS, FUTURE_POS].gt(0).all()),
        f"min = {float(layer0[:, :, QUERY_POS, FUTURE_POS].min()):.6f}",
    )
    report.check(
        f"position {QUERY_POS} spreads weight over all {SEQ_LEN} positions",
        bool((layer0[:, :, QUERY_POS, :] > 0).all()),
    )
    report.check(
        "no triangular structure in any layer (unlike GPT-2)",
        not all(bool((layer[..., torch.triu(
            torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool), diagonal=1)].abs().sum() < 1e-8)
        ) for layer in maps),
    )
    row_sums = torch.stack([layer.sum(-1) for layer in maps])
    report.check(
        "attention rows sum to 1",
        bool(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)),
    )

    # Behavioural proof: information flows backwards into the [CLS] position.
    altered = ids.clone()
    altered[:, FUTURE_POS] = (int(ids[0, FUTURE_POS]) + 1) % 1000 + 5
    with torch.no_grad():
        vec_before = gmn.encode(ids)
        vec_after = gmn.encode(altered)
        cls_hidden = gmn.trunk(ids)[:, 0, :]
    report.check(
        f"changing token {FUTURE_POS} DOES change the [CLS] vector at position 0",
        not bool(torch.allclose(vec_before, vec_after, atol=1e-6)),
    )
    report.check(
        "GMN encode() returns unit vectors for the dummy batch",
        bool(torch.allclose(vec_before.norm(dim=-1), torch.ones(BATCH_SIZE), atol=1e-5)),
    )
    report.check(
        "GMN trunk hidden states are finite",
        bool(torch.isfinite(cls_hidden).all()),
    )


def check_shared_architecture(report: Report, gpt2: GPT2Style, gmn: GMNStyle) -> None:
    section("Shared vs model-specific layers")
    gpt2_shapes = {name: tuple(t.shape) for name, t in gpt2.trunk.state_dict().items()}
    gmn_shapes = {name: tuple(t.shape) for name, t in gmn.trunk.state_dict().items()}

    report.check(
        "both trunks are instances of the same class",
        type(gpt2.trunk) is TransformerBase and type(gmn.trunk) is TransformerBase,
    )
    report.check(
        "trunks expose identical state-dict keys",
        set(gpt2_shapes) == set(gmn_shapes),
        f"{len(gpt2_shapes)} tensors",
    )
    report.check("trunks expose identical tensor shapes", gpt2_shapes == gmn_shapes)
    report.check(
        "both models use the same TransformerBlock / MultiHeadAttention / FeedForward classes",
        all(type(a) is type(b) for a, b in zip(gpt2.trunk.blocks, gmn.trunk.blocks))
        and all(
            type(a.attn) is type(b.attn) and type(a.ffn) is type(b.ffn)
            for a, b in zip(gpt2.trunk.blocks, gmn.trunk.blocks)
        ),
    )

    gpt2_flags = {block.attn.causal for block in gpt2.trunk.blocks}
    gmn_flags = {block.attn.causal for block in gmn.trunk.blocks}
    report.check(
        "the ONLY difference is the causal flag: GPT-2={True}, GMN={False}",
        gpt2_flags == {True} and gmn_flags == {False} and gpt2.trunk.causal is True
        and gmn.trunk.causal is False,
        f"gpt2={gpt2_flags}, gmn={gmn_flags}",
    )

    # Rebuild both trunks from the same seed: if the code path really is shared,
    # every single weight must come out elementwise identical.
    torch.manual_seed(int(config.TRAIN["seed"]))
    seeded_causal = TransformerBase.from_config(causal=True)
    torch.manual_seed(int(config.TRAIN["seed"]))
    seeded_bidir = TransformerBase.from_config(causal=False)
    identical = all(
        torch.equal(tensor, seeded_bidir.state_dict()[name])
        for name, tensor in seeded_causal.state_dict().items()
    )
    report.check(
        "same seed -> both trunks initialise to IDENTICAL weights",
        identical,
        "proves the trunks share one code path",
    )

    report.check(
        "model-specific parts differ as designed",
        isinstance(gpt2.lm_head, nn.Linear) and isinstance(gmn.projection, nn.Linear)
        and not hasattr(gmn, "lm_head") and not hasattr(gpt2, "projection"),
        "GPT-2: lm_head | GMN: projection",
    )


def check_positional_embeddings(report: Report, gpt2: GPT2Style, gmn: GMNStyle,
                                ids: torch.Tensor) -> None:
    section("Positional embeddings (learned, shared class)")
    max_seq_len = int(config.MODEL["max_seq_len"])
    hidden_dim = int(config.MODEL["hidden_dim"])

    for name, trunk in (("GPT-2", gpt2.trunk), ("GMN", gmn.trunk)):
        table = trunk.positional_embedding
        report.check(
            f"{name}: positional embedding is an nn.Embedding (learned, not sinusoidal)",
            isinstance(table, nn.Embedding),
            type(table).__name__,
        )
        report.check(
            f"{name}: positional table shape == (max_seq_len, hidden_dim)",
            tuple(table.weight.shape) == (max_seq_len, hidden_dim),
            f"{tuple(table.weight.shape)}",
        )
        report.check(
            f"{name}: positional embeddings are trainable",
            bool(table.weight.requires_grad),
        )

    buffers = [n for n, _ in gpt2.trunk.named_buffers()] + [
        n for n, _ in gmn.trunk.named_buffers()
    ]
    report.check(
        "no sinusoidal / static positional buffer is registered",
        not any("pe" in n or "position" in n or "sin" in n for n in buffers),
        f"buffers={buffers or 'none'}",
    )

    swapped = ids.clone()
    swapped[:, 1], swapped[:, 2] = ids[:, 2].clone(), ids[:, 1].clone()
    with torch.no_grad():
        report.check(
            "swapping two tokens changes the output (positions are actually used)",
            not bool(torch.allclose(gpt2.trunk(ids), gpt2.trunk(swapped), atol=1e-6)),
        )

    gpt2.zero_grad(set_to_none=True)
    gpt2(ids).sum().backward()
    grad = gpt2.trunk.positional_embedding.weight.grad
    report.check(
        "positional embedding receives a non-zero gradient",
        grad is not None and bool((grad.abs() > 0).any()),
    )
    gpt2.zero_grad(set_to_none=True)


def check_padding_ignored(report: Report, gpt2: GPT2Style) -> None:
    section("GPT-2 loss ignores [PAD]")
    ids = make_dummy_batch()
    padded = torch.cat([ids, torch.full((BATCH_SIZE, 4), PAD_ID, dtype=torch.long)], dim=1)

    per_token = gpt2.loss(padded, reduction="none").view(BATCH_SIZE, -1)
    targets = padded[:, 1:]
    report.check(
        "per-token loss is exactly 0 where the target is [PAD]",
        bool((per_token[targets == PAD_ID] == 0).all()),
        f"{(targets == PAD_ID).sum().item()} padded targets",
    )
    report.check(
        "per-token loss is > 0 where the target is a real token",
        bool((per_token[targets != PAD_ID] > 0).all()),
    )
    report.check(
        "mean loss and perplexity over a padded batch are finite",
        bool(torch.isfinite(gpt2.loss(padded)))
        and torch.isfinite(torch.tensor(gpt2.perplexity(padded))).item(),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    torch.manual_seed(int(config.TRAIN["seed"]))

    print("=" * 74)
    print("TASK 20 — shared transformer blocks (verification)")
    print("=" * 74)
    print(f"config.MODEL   : {config.MODEL}")
    print(f"architecture   : {config.MODEL['n_layers']} layers, "
          f"{config.MODEL['n_heads']} heads, hidden={config.MODEL['hidden_dim']}, "
          f"ffn={config.MODEL['ffn_dim']}, seq={config.MODEL['max_seq_len']}, "
          f"vocab={config.MODEL['vocab_size']}")

    report = Report()
    gpt2 = GPT2Style().eval()
    gmn = GMNStyle().eval()
    ids = make_dummy_batch()
    print(f"dummy batch    : {tuple(ids.shape)}  (seed={config.TRAIN['seed']})")

    logits, question_vec, choice_vecs, scores = check_shapes(report, gpt2, gmn, ids)
    check_forward_consistency(report, gpt2, gmn, ids, logits, scores, question_vec)
    check_parameter_counts(report, gpt2, gmn)
    check_causal_masking(report, gpt2, ids)
    check_bidirectional_attention(report, gmn, ids)
    check_shared_architecture(report, gpt2, gmn)
    check_positional_embeddings(report, gpt2, gmn, ids)
    check_padding_ignored(report, gpt2)

    gpt2_params = gpt2.count_parameters()
    gmn_params = gmn.count_parameters()
    difference = gmn_params - gpt2_params

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print(f"GPT-2 style parameters : {gpt2_params:>12,}")
    print(f"GMN style parameters   : {gmn_params:>12,}")
    print(f"difference             : {difference:>12,}  "
          f"({100.0 * abs(difference) / gpt2_params:.2f}% of GPT-2)")
    print(f"GPT-2 output shape     : (batch, seq, vocab) -> {tuple(logits.shape)}")
    print(f"GMN scores shape       : (batch, 4)          -> {tuple(scores.shape)}")
    print(f"GMN question_vec shape : (batch, latent)     -> {tuple(question_vec.shape)}")
    print("\nlayers SHARED by both models:")
    print("  TransformerBase, TransformerBlock, MultiHeadAttention, FeedForward,")
    print("  learned positional embedding, final LayerNorm, GPT-2 init")
    print("layers model-specific:")
    print("  GPT-2 : linear LM head, tied to token_embedding.weight")
    print("  GMN   : Linear(hidden_dim, 256) projection on the [CLS] position")
    print("the single architectural switch:")
    print("  MultiHeadAttention.forward: causal=True (GPT-2) vs causal=False (GMN)")
    print(f"\nchecks passed : {len(report.checks) - report.n_failed}/{len(report.checks)}")
    print(f"ALL CHECKS PASSED" if report.passed else f"FAILED: {report.n_failed} check(s)")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
