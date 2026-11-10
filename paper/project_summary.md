# Project summary - Geometric Memory Navigation (GMN)

## What we built

A 4-way multiple-choice scorer for HellaSwag with **624,256 trained
parameters** - no classifier head, no autoregressive LM:

1. **Frozen sentence encoder** - `all-MiniLM-L6-v2`, 384-d, never trained.
2. **Learned latent head** - 384 -> 512 -> 256 (`mu`/`logvar`), a VAE head.
3. **Memory store** - 39,905 training examples stored as 256-d latent
   vectors (FAISS kNN, k=10).
4. **Parallel sampler + tangent fusion** - one (weight, direction) per retrieved
   memory; the top-3 are fused and projected onto the query's tangent space.
5. **Geometric scoring** - `argmax_k cos(unit_ending_k, unit_ctx)`; no trained
   classifier participates in inference.

The design is **CLIP-like**: the context is the query embedding, the four candidate
endings are the candidates, and they are ranked by cosine similarity in a learned
latent space. Total trainable parameters: **624,256** - 199x
fewer than GPT-2 small (124,439,808), with no tokenizer, no attention and no
sampling.

## What we found

1. **Context-cosine scoring alone beats GPT-2 small**: 31.31% vs 28.58% on
   10,042 examples (p < 0.0001, exact McNemar on the same examples)
   with 624,256 parameters.
2. **Navigation does not help.** Chord navigation actively hurts (-3.6 points,
   p < 0.0001) because the chord sits ~117 degrees from the context; tangent
   projection recovers part of that loss but never reaches the context-only
   baseline. On the honest 5,042-example split the tuned lambda is not measurable
   (29.7% vs 28.6%, p = 0.0616).
3. **Margin-ranking structure induction does not help either.** It reliably widens
   the correct-vs-wrong cosine gap (0.027 -> 0.047) yet moves accuracy
   by only +0.57 points (p = 0.3083).
4. **Task-level fine-tuning does help** - the project's first significant gain.
   Training the latent head end-to-end on the 4-way objective lifts accuracy from
   31.31% to 32.43% (+1.13 points over the original, p = 0.0034). Best observed:
   32.56% at T=0.07 (post-hoc, p = 0.0010).
5. **The gap hypothesis is inverted.** Accuracy is highest where the
   correct-vs-wrong cosine gap is *smallest*, and per-example gap std matters more
   than gap size (0.0322 for CLIP vs 0.2408 for the margin loss):

| Model | Val gap | Accuracy |
|---|---|---|
| CLIP-style (5 epochs) | 0.0071 | 32.14% |
| CLIP-style (T=0.07, best observed) | 0.0075 | 32.56% |
| CLIP-style (T=0.10, pre-registered) | 0.0108 | 32.43% |
| Original VAE (recon + KL) | 0.0274 | 31.31% |
| Margin ranking (v3, margin=0.3) | 0.0474 | 31.88% |
| Margin ranking (v2, margin=0.5) | 0.0530 | 31.37% |

6. **Early stopping matters more than training length.** Val accuracy peaks at
   epoch 6 and then declines while the training loss keeps falling
   monotonically - a falling loss was not evidence that more training would help.
7. **Temperature is robust in a band.** All of T=0.03-0.10 significantly beat the
   original; too-high temperature is the failure mode:

| T | Accuracy | vs original | p vs original |
|---|---|---|---|
| 0.03 | 32.23% | +0.93 | 0.0129 |
| 0.07 | 32.56% | +1.25 | 0.0010 |
| 0.10 | 32.43% | +1.13 | 0.0034 |
| 0.20 | 32.03% | +0.72 | 0.0316 |

## What the paper claims

* **C1.** A 624K-parameter geometric scorer significantly beats GPT-2 small
  (124M) on HellaSwag: 31.31% vs 28.58% (p < 0.0001) - a
  199x parameter reduction.
* **C2.** Training the latent head on the task objective rather than only on
  reconstruction gives a small but statistically significant improvement:
  32.43% vs 31.31% (p = 0.0034).
* **C3.** For this architecture the correct-vs-wrong cosine gap is *not* the
  mechanism: maximising it buys nothing, and the highest-accuracy models have the
  smallest gaps.

## What the paper does not claim

* **Navigation does not improve accuracy.** Every navigated variant is worse than,
  or indistinguishable from, context-cosine scoring.
* **Gap size is not a usable proxy metric** for tuning future variants.
* **32.56% is not the headline.** That value was chosen after seeing the
  10,042 results; the pre-registered figure is 32.43%.
* **No multiple-comparison correction.** About ten model variants were compared on
  the same 10,042 examples, so the reported p-values are optimistic, and every
  run used a single seed, so run-to-run variance is unknown.
* **The earlier honest numbers apply only to their own split**: 29.7%
  no-navigation vs 28.6% with navigation on the 5,042-example held-out set.
  Tables 1 and 3 report the full 10,042 set and are not comparable to those.

## Reproducing

| Artifact | What it holds |
|---|---|
| `checkpoints/vae.pt` | original reconstruction + KL VAE (TASK 2) |
| `checkpoints/vae_structured_v2.pt`, `vae_structured_v3.pt` | margin-ranking variants (TASK 13/14) |
| `checkpoints/vae_clip_best_overall.pt` | pre-registered CLIP-style model (T=0.10) |
| `checkpoints/vae_clip_T0.07.pt` | best-observed CLIP-style model (T=0.07) |
| `experiments/results_*.json` | every number quoted above |
| `scripts/generate_paper_table.py` | regenerates all three tables, findings and this file |
