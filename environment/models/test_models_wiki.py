"""Verification script for TASK 19R: the WikiText-scale architectures.

Run from the repository root::

    python environment/models/test_models_wiki.py

It runs the **same 47 checks** as ``test_models.py`` (TASK 20), but built from

* ``config.MODEL_SMALL`` — 6 layers / 8 heads / 256 hidden / ffn 1024 /
  vocab 16000 / seq_len 64 (TASK 19R note 1), and
* the TASK 18R **WikiText** tokenizer ``bpe_wiki_vocab.json`` (vocab 16000), not
  the 8000-token ARC one.

and adds the checks TASK 19R asks for on top:

* ``key_padding_mask`` reproduces a trimmed batch exactly for the bidirectional
  model (a non-architectural correctness property, not the causal switch);
* the GMN projection head is ``Linear(hidden_dim, 256)`` for **every** width, so
  the latent space stays 256-d;
* both models and the trunk honour an arbitrary config dict;
* formula/measured parameter counts for SMALL and MEDIUM, and TASK 19R's size
  targets (a missed target is reported, never silently tuned);
* forward-pass shapes on real WikiText sentences at seq_len 64 / vocab 16000,
  for SMALL and for MEDIUM.

Exits non-zero if any check fails.

RAM note (TASK 18R constraint): the 3.8M-sentence ``train_sentences.json`` is
never loaded wholesale — :func:`read_json_array` streams it line by line, one
element per line, which is the pattern TASK 20R/21R trainers must use too.
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

MODELS_DIR = Path(__file__).resolve().parent
ENVIRONMENT_DIR = MODELS_DIR.parent
TOKENIZER_DIR = ENVIRONMENT_DIR / "tokenizer"
for _directory in (ENVIRONMENT_DIR, MODELS_DIR, TOKENIZER_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import config  # noqa: E402
from bpe import BOS_ID, EOS_ID, PAD_ID, BPETokenizer  # noqa: E402

from gmn_style import LATENT_DIM, GMNStyle  # noqa: E402
from gpt2_style import GPT2Style  # noqa: E402
from model_sizes import gmn_parameters, gpt2_parameters  # noqa: E402
from test_models import Report, section  # noqa: E402
from transformer import (  # noqa: E402
    TransformerBase,
    resolve_model_config,
)

#: The TASK 19R config under test, plus MEDIUM for the size/forward checks.
SMALL: Dict[str, Any] = config.model_config("small")
MEDIUM: Dict[str, Any] = config.model_config("medium")

#: A third, deliberately odd width: the GMN head must still emit 256 numbers.
#: 192 is chosen so all three widths (SMALL 256, MEDIUM 384, this one) differ.
CUSTOM: Dict[str, Any] = dict(SMALL, n_layers=2, hidden_dim=192, ffn_dim=512, max_seq_len=16)

BATCH_SIZE = 2
SEQ_LEN = int(SMALL["max_seq_len"])            # 64
VOCAB_SIZE = int(SMALL["vocab_size"])          # 16000
ROUND_TRIP_SENTENCES = 64

#: Position pair used by the attention-direction checks (same as TASK 20's tests).
QUERY_POS = 5
FUTURE_POS = 6

#: TASK 19R size targets (see environment/models/model_sizes.py).  MEDIUM's ~22M
#: became reachable when DECISION 1 corrected its fields to 9 x 384 / ffn 1536.
SIZE_TARGETS = {"small": 9_000_000, "medium": 22_000_000}
SIZE_TOLERANCE = 0.10


def read_json_array(path: Path, limit: int) -> List[Any]:
    """Stream the first ``limit`` elements of a one-element-per-line JSON array.

    TASK 18R wrote both sentence files in this shape precisely so they can be read
    without materialising 1.5 GB of JSON (TASK 18R RAM constraint).  Each line
    carries one element plus, optionally, the array's separators, so the line is
    handed to :meth:`json.JSONDecoder.raw_decode`, which parses the leading value
    and ignores whatever punctuation follows it.
    """
    decoder = json.JSONDecoder()
    items: List[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if not items and line.startswith("["):     # the array's own opening bracket
                line = line[1:]
            value, _ = decoder.raw_decode(line)
            items.append(value)
            if len(items) >= limit:
                break
    return items


def encode_batch(
    tokenizer: BPETokenizer,
    texts: Sequence[str],
    pad_to: int = SEQ_LEN,
    add_special_tokens: bool = True,
) -> torch.Tensor:
    """Tokenize ``texts`` into a ``(len(texts), pad_to)`` long tensor.

    Truncation to ``pad_to`` is applied *before* padding, and ``[BOS]/[EOS]``
    count towards the budget, exactly as a trainer should do it.
    """
    rows: List[List[int]] = []
    for text in texts:
        ids = tokenizer.encode_unpadded(text, max_length=None)
        if add_special_tokens:
            ids = [BOS_ID, *ids, EOS_ID]
        ids = ids[:pad_to]
        rows.append(ids + [PAD_ID] * (pad_to - len(ids)))
    return torch.tensor(rows, dtype=torch.long)


def padding_mask(ids: torch.Tensor) -> torch.Tensor:
    """``(B, T)`` bool mask, ``True`` at ``[PAD]`` positions."""
    return ids.eq(PAD_ID)


# ---------------------------------------------------------------------------
# The 47 checks shared with test_models.py (TASK 20), run on MODEL_SMALL
# ---------------------------------------------------------------------------
def make_dummy_batch(seq_len: int = SEQ_LEN) -> torch.Tensor:
    """Deterministic dummy ids: [BOS] body... [EOS] [EOS], without using [PAD]."""
    generator = torch.Generator().manual_seed(int(config.TRAIN["seed"]))
    body = torch.randint(5, VOCAB_SIZE, (BATCH_SIZE, seq_len - 3), generator=generator)
    bos = torch.full((BATCH_SIZE, 1), BOS_ID, dtype=torch.long)
    eos = torch.full((BATCH_SIZE, 2), EOS_ID, dtype=torch.long)
    return torch.cat([bos, body, eos], dim=1)


def check_shapes(report: Report, gpt2: GPT2Style, gmn: GMNStyle, ids: torch.Tensor):
    section(f"Output shapes (MODEL_SMALL: vocab {VOCAB_SIZE:,}, seq_len {SEQ_LEN})")
    with torch.no_grad():
        logits = gpt2(ids)
        choices = ids.unsqueeze(1).repeat(1, 4, 1)
        question_vec, choice_vecs, scores = gmn(ids, choices)

    print(f"  GPT-2 logits       : {tuple(logits.shape)}")
    print(f"  GMN question_vec   : {tuple(question_vec.shape)}")
    print(f"  GMN choice_vecs    : {tuple(choice_vecs.shape)}")
    print(f"  GMN scores         : {tuple(scores.shape)}")

    report.check(
        "GPT-2 logits shape == (batch, seq, vocab)",
        tuple(logits.shape) == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE),
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
    # against row 1's tokens as well).  Bitwise equality is not required: at
    # seq_len 64 with a 16000-token vocabulary, CPU matmul reassociates over the
    # larger batch and the last float32 bit can move.  TASK 20's smaller suite
    # asserted bitwise equality; here the tolerance is stated and the diff shown.
    batch_diff = float((batched[:1] - single).abs().max())
    report.check(
        "GPT-2 is batch-invariant: gpt2(ids)[:1] ~= gpt2(ids[:1])",
        batch_diff < 1e-4,
        f"max abs diff = {batch_diff:.2e} (float32 reassociation)",
    )
    report.check(
        "GMN encode is batch-invariant: encode(ids)[:1] == encode(ids[:1])",
        bool(torch.allclose(encode_batched[:1], encode_single, atol=1e-6, rtol=0)),
        f"max diff = {float((encode_batched[:1] - encode_single).abs().max()):.2e}",
    )

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


def check_parameter_counts(report: Report, gpt2: GPT2Style, gmn: GMNStyle) -> Dict[str, int]:
    section(f"Parameter counts (MODEL_SMALL: {SMALL['n_layers']} layers, "
            f"hidden {SMALL['hidden_dim']}, ffn {SMALL['ffn_dim']}, vocab {VOCAB_SIZE:,})")
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

    report.check("parameter difference is within 5%", relative < 5.0, f"{relative:.2f}%")
    report.check(
        "GPT-2 LM head weight IS the token embedding",
        gpt2.lm_head.weight is gpt2.trunk.token_embedding.weight,
    )
    untied = GPT2Style(config_dict=gpt2.config, tie_weights=False).count_parameters()
    print(f"  GPT-2 if UNTIED        : {untied:>12,}  "
          f"(+{untied - gpt2_params:,} = vocab_size x hidden_dim)")
    report.check(
        "weight tying removes exactly vocab_size x hidden_dim params",
        untied - gpt2_params == gpt2.vocab_size * gpt2.hidden_dim,
        f"{untied - gpt2_params:,}",
    )
    return {"gpt2": gpt2_params, "gmn": gmn_params}


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

    # Behavioural proof, stronger than inspecting the mask: a change at position
    # FUTURE_POS cannot influence the logits at positions <= QUERY_POS.
    altered = ids.clone()
    altered[:, FUTURE_POS] = torch.clamp((ids[:, FUTURE_POS] + 1) % VOCAB_SIZE, min=5)
    with torch.no_grad():
        before = gpt2(ids)
        after = gpt2(altered)
    report.check(
        f"changing token {FUTURE_POS} leaves logits at positions <= {QUERY_POS} "
        f"bitwise unchanged",
        bool(torch.equal(before[:, :FUTURE_POS], after[:, :FUTURE_POS])),
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
    altered[:, FUTURE_POS] = torch.clamp((ids[:, FUTURE_POS] + 1) % VOCAB_SIZE, min=5)
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

    # Rebuild both trunks from one seed: if the code path really is shared, every
    # single weight must come out elementwise identical.
    torch.manual_seed(int(config.TRAIN["seed"]))
    seeded_causal = TransformerBase.from_config(causal=True, config_dict=SMALL)
    torch.manual_seed(int(config.TRAIN["seed"]))
    seeded_bidir = TransformerBase.from_config(causal=False, config_dict=SMALL)
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
    del seeded_causal, seeded_bidir


def check_positional_embeddings(report: Report, gpt2: GPT2Style, gmn: GMNStyle,
                                ids: torch.Tensor) -> None:
    section("Positional embeddings (learned, shared class)")
    max_seq_len = int(SMALL["max_seq_len"])
    hidden_dim = int(SMALL["hidden_dim"])

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
    section(f"GPT-2 loss ignores [PAD] (batch padded up to max_seq_len {SEQ_LEN})")
    # 60 real tokens + 4 [PAD] == SEQ_LEN, so the padded batch exercises the
    # max_seq_len boundary instead of merely exceeding it.
    ids = make_dummy_batch(seq_len=SEQ_LEN - 4)
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
    gpt2.zero_grad(set_to_none=True)


# ---------------------------------------------------------------------------
# Checks TASK 19R adds on top of TASK 20's 47
# ---------------------------------------------------------------------------
def check_padding_mask(
    report: Report,
    gmn: GMNStyle,
    tokenizer: BPETokenizer,
    sentences: Sequence[str],
) -> None:
    """``key_padding_mask`` must reproduce a trimmed batch for the encoder.

    Masking is an *input* property, not the causal/bidirectional switch: with the
    mask, a ``[PAD]``-filled batch must give exactly the vector a trimmed batch
    gives; without it ``[CLS]`` attends to ``[PAD]`` and the sentence vector is
    measurably corrupted.
    """
    section("GMN key_padding_mask reproduces a trimmed batch")
    texts = list(sentences[:BATCH_SIZE])

    worst_abs = 0.0
    worst_cos = 1.0
    worst_unmasked_cos = 1.0
    for text in texts:
        padded = encode_batch(tokenizer, [text])                 # (1, 64)
        mask = padding_mask(padded)
        real_len = int((~mask).sum())
        trimmed = padded[:, :real_len]
        with torch.no_grad():
            masked_vec = gmn.encode(padded, key_padding_mask=mask)
            trimmed_vec = gmn.encode(trimmed)
            unmasked_vec = gmn.encode(padded)
        masked_cos = float(torch.nn.functional.cosine_similarity(masked_vec, trimmed_vec))
        unmasked_cos = float(torch.nn.functional.cosine_similarity(unmasked_vec, trimmed_vec))
        worst_abs = max(worst_abs, float((masked_vec - trimmed_vec).abs().max()))
        worst_cos = min(worst_cos, masked_cos)
        worst_unmasked_cos = min(worst_unmasked_cos, unmasked_cos)
        print(f"  real {real_len:>2d}/64 tokens : masked-vs-trimmed cos {masked_cos:.6f}, "
              f"unmasked-vs-trimmed cos {unmasked_cos:.4f}")

    report.check(
        "encode(padded, key_padding_mask) == encode(trimmed) exactly",
        worst_abs < 1e-5,
        f"max abs diff = {worst_abs:.2e}",
    )
    report.check(
        "and cosine similarity is 1.0",
        worst_cos > 1.0 - 1e-5,
        f"min cosine = {worst_cos:.6f}",
    )
    report.check(
        "without the mask the vector is different (the mask does real work)",
        worst_unmasked_cos < 0.999,
        f"worst unmasked cosine = {worst_unmasked_cos:.4f}",
    )

    padded = encode_batch(tokenizer, texts)
    mask = padding_mask(padded)
    with torch.no_grad():
        _, weights = gmn.trunk(padded, key_padding_mask=mask, need_weights=True)
    layer0 = weights[0]                                   # (B, H, T, T)
    padded_key_weights = layer0.masked_select(mask[:, None, None, :])
    report.check(
        "masked key positions receive EXACTLY zero attention weight",
        float(padded_key_weights.abs().max()) == 0.0,
        f"max weight at a padded key = {float(padded_key_weights.abs().max())}",
    )
    row_sums = layer0.sum(-1)
    report.check(
        "attention rows still sum to 1 under the mask",
        bool(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)),
        f"min row sum = {float(row_sums.min()):.6f}",
    )

    choices = padded.unsqueeze(1).repeat(1, 4, 1)
    choice_mask = mask.unsqueeze(1).repeat(1, 4, 1)
    with torch.no_grad():
        question_vec, choice_vecs, scores = gmn(
            padded, choices, question_mask=mask, choice_mask=choice_mask
        )
    report.check(
        "forward() with question_mask/choice_mask returns finite 256-d vectors and scores",
        bool(torch.isfinite(question_vec).all() and torch.isfinite(choice_vecs).all()
             and torch.isfinite(scores).all())
        and tuple(question_vec.shape) == (BATCH_SIZE, LATENT_DIM)
        and tuple(scores.shape) == (BATCH_SIZE, 4),
        f"scores={tuple(scores.shape)}",
    )


def check_latent_dim_fixed(report: Report) -> None:
    """The GMN head is ``Linear(hidden_dim, 256)`` whatever the trunk width is."""
    section(f"GMN projection is Linear(hidden_dim, {LATENT_DIM}) for every width")
    rows: List[Dict[str, Any]] = []
    for label, cfg in (("small (h256)", SMALL), ("medium (h512)", MEDIUM),
                       (f"custom (h{CUSTOM['hidden_dim']})", CUSTOM)):
        gmn = GMNStyle(config_dict=cfg)
        gpt2 = GPT2Style(config_dict=cfg)
        seq = min(SEQ_LEN, int(cfg["max_seq_len"]))
        gmn.eval()
        with torch.no_grad():
            vecs = gmn.encode(make_dummy_batch(seq_len=seq))
        rows.append({
            "label": label,
            "hidden": int(cfg["hidden_dim"]),
            "weight_shape": tuple(gmn.projection.weight.shape),
            "latent_dim": int(gmn.latent_dim),
            "vector_shape": tuple(vecs.shape),
            "norm_error": float((vecs.norm(dim=-1) - 1.0).abs().max()),
            "difference": gmn.count_parameters() - gpt2.count_parameters(),
            "expected_difference": int(cfg["hidden_dim"]) * LATENT_DIM + LATENT_DIM,
        })
        print(f"  {label:<15s} hidden {cfg['hidden_dim']:<4d} -> projection "
              f"{rows[-1]['weight_shape']}  encode -> {rows[-1]['vector_shape']}  "
              f"(GMN - GPT-2 = {rows[-1]['difference']:,})")
        del gmn, gpt2
        gc.collect()

    report.check(
        "projection weight is always (latent_dim, hidden_dim)",
        all(row["weight_shape"] == (LATENT_DIM, row["hidden"]) for row in rows),
        f"{[row['weight_shape'] for row in rows]}",
    )
    report.check(
        "model.latent_dim == 256 for every config",
        all(row["latent_dim"] == LATENT_DIM for row in rows),
    )
    report.check(
        "encode() output width is 256 for every config",
        all(row["vector_shape"] == (BATCH_SIZE, LATENT_DIM) for row in rows),
        f"{[row['vector_shape'] for row in rows]}",
    )
    report.check(
        "every config still returns unit vectors",
        all(row["norm_error"] < 1e-5 for row in rows),
        f"max |norm - 1| = {max(row['norm_error'] for row in rows):.2e}",
    )
    report.check(
        "GMN - GPT-2 == hidden_dim * 256 + 256 for every config",
        all(row["difference"] == row["expected_difference"] for row in rows),
        f"{[(row['hidden'], row['difference']) for row in rows]}",
    )
    report.check(
        "the widths really differ (so the check is not vacuous)",
        len({row["hidden"] for row in rows}) == 3,
        f"widths={[row['hidden'] for row in rows]}",
    )


def check_config_plumbing(report: Report) -> None:
    """Both models and the trunk must honour an arbitrary config dict (TASK 19R)."""
    section("Config-dict plumbing")
    trunk = TransformerBase.from_config(causal=True, config_dict=CUSTOM)
    actual = (trunk.n_layers, trunk.n_heads, trunk.hidden_dim, trunk.ffn_dim,
              trunk.vocab_size, trunk.max_seq_len, trunk.dropout)
    expected = (CUSTOM["n_layers"], CUSTOM["n_heads"], CUSTOM["hidden_dim"],
                CUSTOM["ffn_dim"], CUSTOM["vocab_size"], CUSTOM["max_seq_len"],
                CUSTOM["dropout"])
    print(f"  custom dict   : {CUSTOM}")
    print(f"  trunk built   : layers={actual[0]} heads={actual[1]} hidden={actual[2]} "
          f"ffn={actual[3]} vocab={actual[4]} seq={actual[5]} dropout={actual[6]}")
    report.check(
        "trunk built from a custom dict has exactly those dimensions",
        actual == expected,
        f"{actual}",
    )

    missing_rejected = False
    try:
        resolve_model_config({"n_layers": 1, "hidden_dim": 16})
    except KeyError:
        missing_rejected = True
    report.check("a dict missing required keys raises KeyError", missing_rejected)

    heads_rejected = False
    try:
        resolve_model_config(dict(SMALL, hidden_dim=250, n_heads=8))
    except ValueError:
        heads_rejected = True
    report.check(
        "hidden_dim not divisible by n_heads raises ValueError", heads_rejected
    )

    gpt2 = GPT2Style(config_dict=CUSTOM)
    gmn = GMNStyle(config_dict=CUSTOM)
    same_trunks = (
        {n: tuple(t.shape) for n, t in gpt2.trunk.state_dict().items()}
        == {n: tuple(t.shape) for n, t in gmn.trunk.state_dict().items()}
    )
    report.check(
        "both models built from one dict share identical trunk shapes",
        same_trunks,
        f"{len(gpt2.trunk.state_dict())} tensors",
    )

    default_model = GPT2Style()
    report.check(
        "config_dict=None falls back to config.MODEL (the SMALL WikiText contract)",
        default_model.config == config.model_config("small")
        and gpt2.config == resolve_model_config(CUSTOM)
        and "config" in default_model.summary() and "config" in gmn.summary(),
        f"default vocab={default_model.config['vocab_size']}, "
        f"seq={default_model.config['max_seq_len']}",
    )
    del trunk, gpt2, gmn, default_model
    gc.collect()


def check_size_targets(report: Report) -> Dict[str, Dict[str, int]]:
    """Measured SMALL/MEDIUM sizes vs the closed-form formula and the targets."""
    section("Parameter counts for SMALL and MEDIUM: formula and targets")
    measured: Dict[str, Dict[str, Any]] = {}
    for label, cfg in (("small", SMALL), ("medium", MEDIUM)):
        gpt2 = GPT2Style(config_dict=cfg)
        gmn = GMNStyle(config_dict=cfg)
        row = {
            "gpt2": gpt2.count_parameters(),
            "gmn": gmn.count_parameters(),
            "formula_gpt2": gpt2_parameters(cfg),
            "formula_gmn": gmn_parameters(cfg),
        }
        del gpt2, gmn
        gc.collect()
        measured[label] = row

        target = SIZE_TARGETS[label]
        delta = 100.0 * (row["gpt2"] - target) / target
        verdict = "MET" if abs(delta) <= 100 * SIZE_TOLERANCE else "NOT MET"
        row["verdict"] = verdict
        print(f"  {label.upper():<7s} measured GPT-2 {row['gpt2']:>12,} "
              f"GMN {row['gmn']:>12,}")
        print(f"          formula  GPT-2 {row['formula_gpt2']:>12,} "
              f"GMN {row['formula_gmn']:>12,}  |  target ~{target:,} -> "
              f"{verdict} ({delta:+.1f}%)")

    for label in ("small", "medium"):
        row = measured[label]
        report.check(
            f"closed-form formula matches the measured {label.upper()} counts",
            row["gpt2"] == row["formula_gpt2"] and row["gmn"] == row["formula_gmn"],
            f"{row['gpt2']:,} / {row['gmn']:,}",
        )
    report.check(
        "the two architectures are within 5% of each other at both sizes",
        all(
            100.0 * abs(row["gmn"] - row["gpt2"]) / row["gpt2"] < 5.0
            for row in measured.values()
        ),
        ", ".join(
            f"{label}={100.0 * abs(row['gmn'] - row['gpt2']) / row['gpt2']:.2f}%"
            for label, row in measured.items()
        ),
    )
    report.check(
        "SMALL is within 10% of its ~9M target",
        measured["small"]["verdict"] == "MET",
        f"{measured['small']['gpt2']:,} vs ~{SIZE_TARGETS['small']:,}",
    )
    report.check(
        "MEDIUM is within 10% of its ~22M target (DECISION 1 fields)",
        measured["medium"]["verdict"] == "MET",
        f"{measured['medium']['gpt2']:,} vs ~{SIZE_TARGETS['medium']:,}",
    )

    for label, cfg in (("SMALL", SMALL), ("MEDIUM", MEDIUM)):
        row = measured[label.lower()]
        embedding = cfg["vocab_size"] * cfg["hidden_dim"]
        positions = cfg["max_seq_len"] * cfg["hidden_dim"]
        final_ln = 2 * cfg["hidden_dim"]
        per_block = (row["formula_gpt2"] - embedding - positions - final_ln) // cfg["n_layers"]
        print(f"  {label} = tied embedding {embedding:,} + positions {positions:,} + "
              f"{cfg['n_layers']} x {per_block:,} blocks + {final_ln:,} = {row['gpt2']:,}")
        if row["verdict"] != "MET":
            print(f"    -> ~{SIZE_TARGETS[label.lower()]:,} is UNREACHABLE with these "
                  f"fields (the blocks alone are {cfg['n_layers'] * per_block / 1e6:.1f}M); "
                  "see model_sizes.py for measured alternatives.")
    return measured


def check_wiki_tokenizer(
    report: Report,
    tokenizer: BPETokenizer,
    sentences: Sequence[str],
) -> torch.Tensor:
    """The architecture must be paired with the TASK 18R WikiText tokenizer."""
    section(f"TASK 18R WikiText tokenizer (vocab {VOCAB_SIZE:,}, not the ARC one)")
    print(f"  wiki tokenizer : {config.path('tokenizer_wiki')}  "
          f"vocab={tokenizer.vocab_size:,}  merges={len(tokenizer.merges):,}")
    arc_size: Optional[int] = None
    arc_path = config.path("tokenizer")
    if arc_path.exists():
        arc = BPETokenizer.load(arc_path)
        arc_size = arc.vocab_size
        print(f"  ARC tokenizer  : {arc_path}  vocab={arc_size:,}  "
              f"merges={len(arc.merges):,}")
    else:
        print(f"  ARC tokenizer  : {arc_path} (absent)")

    report.check(
        f"wiki vocab_size == config vocab_size ({VOCAB_SIZE:,}) for SMALL and MEDIUM",
        tokenizer.vocab_size == int(SMALL["vocab_size"]) == int(MEDIUM["vocab_size"]),
        f"{tokenizer.vocab_size:,}",
    )
    report.check(
        "the wiki tokenizer is NOT the 8000-token ARC tokenizer",
        arc_size is None or arc_size != tokenizer.vocab_size,
        f"wiki={tokenizer.vocab_size:,}, ARC={arc_size}",
    )

    ids = encode_batch(tokenizer, sentences[:BATCH_SIZE])
    mask = padding_mask(ids)
    print(f"  real batch     : {tuple(ids.shape)}  "
          f"({int(mask.sum()):,} of {ids.numel():,} positions are [PAD])")
    report.check(
        f"real sentences encode to (B, {SEQ_LEN}) with [BOS] first and [PAD] present",
        tuple(ids.shape) == (BATCH_SIZE, SEQ_LEN)
        and bool((ids[:, 0] == BOS_ID).all())
        and bool(mask.any()),
        f"{tuple(ids.shape)}",
    )

    def pads_form_a_suffix(row: torch.Tensor) -> bool:
        padded = row.bool()
        if not bool(padded.any()):
            return True
        return bool(padded[int(torch.nonzero(padded, as_tuple=False)[0]):].all())

    report.check(
        "[PAD] only ever appears as a suffix (no holes inside a sequence)",
        all(pads_form_a_suffix(row) for row in mask),
    )
    real_lengths = (~mask).sum(dim=1).tolist()
    report.check(
        "the batch is not degenerate: every real sentence is longer than 5 tokens",
        min(real_lengths) > 5,
        f"real lengths={real_lengths}",
    )
    return ids


def check_medium_forward(
    report: Report,
    tokenizer: BPETokenizer,
    sentences: Sequence[str],
) -> None:
    """MEDIUM must forward on real data too (TASK 19R: verify both configs)."""
    section(f"Forward pass shapes: MODEL_MEDIUM ({MEDIUM['n_layers']} x "
            f"{MEDIUM['hidden_dim']}, ffn {MEDIUM['ffn_dim']}, seq {SEQ_LEN})")
    gpt2 = GPT2Style(config_dict=MEDIUM).eval()
    gmn = GMNStyle(config_dict=MEDIUM).eval()
    ids = encode_batch(tokenizer, sentences[:BATCH_SIZE])
    choices = ids.unsqueeze(1).repeat(1, 4, 1)
    with torch.no_grad():
        logits = gpt2(ids)
        loss = float(gpt2.loss(ids))
        question_vec, choice_vecs, scores = gmn(ids, choices)

    print(f"  GPT-2 params {gpt2.count_parameters():>11,}  logits {tuple(logits.shape)}  "
          f"loss {loss:.4f}")
    print(f"  GMN   params {gmn.count_parameters():>11,}  question_vec "
          f"{tuple(question_vec.shape)}  choice_vecs {tuple(choice_vecs.shape)}  "
          f"scores {tuple(scores.shape)}")

    report.check(
        f"MEDIUM GPT-2 logits are ({BATCH_SIZE}, {SEQ_LEN}, {VOCAB_SIZE}) and finite",
        tuple(logits.shape) == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        and bool(torch.isfinite(logits).all())
        and torch.isfinite(torch.tensor(loss)).item(),
        f"{tuple(logits.shape)}",
    )
    report.check(
        f"MEDIUM GMN vectors are {LATENT_DIM}-d, scores are ({BATCH_SIZE}, 4), all finite",
        tuple(question_vec.shape) == (BATCH_SIZE, LATENT_DIM)
        and tuple(choice_vecs.shape) == (BATCH_SIZE, 4, LATENT_DIM)
        and tuple(scores.shape) == (BATCH_SIZE, 4)
        and bool(torch.isfinite(scores).all()),
        f"scores={tuple(scores.shape)}",
    )
    del gpt2, gmn
    gc.collect()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    torch.manual_seed(int(config.TRAIN["seed"]))

    print("=" * 74)
    print("TASK 19R — WikiText-scale architectures (verification)")
    print("=" * 74)
    print(f"MODEL_SMALL  : {SMALL}")
    print(f"MODEL_MEDIUM : {MEDIUM}")
    print(f"torch        : {torch.__version__}")

    tokenizer = BPETokenizer.load(config.path("tokenizer_wiki"))
    sentences = read_json_array(config.path("wiki_val_sentences"), ROUND_TRIP_SENTENCES)
    print(f"streamed     : {len(sentences)} val sentences, line by line "
          f"(the 3.8M-sentence train file is never loaded)")

    report = Report()
    gpt2 = GPT2Style(config_dict=SMALL).eval()
    gmn = GMNStyle(config_dict=SMALL).eval()
    ids = make_dummy_batch()
    print(f"dummy batch  : {tuple(ids.shape)}  (seed={config.TRAIN['seed']})\n")

    logits, question_vec, choice_vecs, scores = check_shapes(report, gpt2, gmn, ids)
    check_forward_consistency(report, gpt2, gmn, ids, logits, scores, question_vec)
    check_parameter_counts(report, gpt2, gmn)
    check_causal_masking(report, gpt2, ids)
    check_bidirectional_attention(report, gmn, ids)
    check_shared_architecture(report, gpt2, gmn)
    check_positional_embeddings(report, gpt2, gmn, ids)
    check_padding_ignored(report, gpt2)
    shared_checks = len(report.checks)
    print(f"\n  {shared_checks} checks so far — TASK 20's suite, rebuilt on MODEL_SMALL")

    check_padding_mask(report, gmn, tokenizer, sentences)
    check_latent_dim_fixed(report)
    check_config_plumbing(report)
    measured = check_size_targets(report)
    check_wiki_tokenizer(report, tokenizer, sentences)
    check_medium_forward(report, tokenizer, sentences)

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    for label, row in measured.items():
        print(f"  {label.upper():<7s} GPT-2 {row['gpt2']:>12,} ({row['gpt2'] / 1e6:>5.2f}M)   "
              f"GMN {row['gmn']:>12,} ({row['gmn'] / 1e6:>5.2f}M)   "
              f"target ~{SIZE_TARGETS[label]:,} -> {row['verdict']}")
    print(f"\n  shared checks (TASK 20's suite on MODEL_SMALL) : {shared_checks}")
    print(f"  new checks (TASK 19R)                          : "
          f"{len(report.checks) - shared_checks}")
    print(f"  checks passed : {len(report.checks) - report.n_failed}/{len(report.checks)}")
    print("ALL CHECKS PASSED" if report.passed else f"FAILED: {report.n_failed} check(s)")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
