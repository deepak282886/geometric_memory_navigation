# TASKS V2 — Structure Induction

These tasks extend the GMN architecture with a key insight:
**induce structure in the latent space so navigation becomes trivial.**

The VAE currently trains on reconstruction + KL only.
This gives a smooth manifold but no meaningful geometry for navigation.
Adding a contrastive structure loss forces correct endings to be
geometrically closer to their contexts than wrong endings are.

If no_navigation accuracy improves significantly with the structured VAE,
the core claim is proven: **induced structure makes navigation trivial.**

Execute one task at a time. Print a DONE block after each. Wait for confirmation.

---

## TASK 13 — Structured VAE Training
**Status: CURRENT**

### The Loss

Add a margin ranking loss to the existing VAE loss:

```
L_structure = mean(max(0,
    cos(ctx_vector, wrong_ending_vector) -
    cos(ctx_vector, correct_ending_vector) + margin))
```

Where margin = 0.5 (tunable).

This directly forces:
  cos(ctx, correct_ending) > cos(ctx, wrong_ending) + 0.5

The geometry now encodes what the scorer needs.

### Instructions

1. Update src/autoencoder/vae.py:

   Add function structure_loss(ctx_vectors, correct_vectors,
                               wrong_vectors, margin=0.5):
     - ctx_vectors:     (N, 256) tensor — encoded contexts
     - correct_vectors: (N, 256) tensor — encoded correct endings
     - wrong_vectors:   (N, 256) tensor — encoded ONE random wrong ending per example
     - All inputs are mu vectors (deterministic, not sampled z)
     - Compute cosine similarities (use F.cosine_similarity)
     - Return mean(max(0, cos_wrong - cos_correct + margin))
     - Also return: mean cos_correct, mean cos_wrong (for logging)

2. Write scripts/train_vae_structured.py:

   Setup:
   - Load data/train_full.json (39,905 examples)
   - Initialize a FRESH VAE (do not load checkpoints/vae.pt)
   - Same architecture as before (all-MiniLM-L6-v2 frozen base)
   - Optimizer: Adam lr=1e-3
   - Save to checkpoints/vae_structured.pt

   Training texts:
   - Same as before: ctx + all 4 endings per example
   - Pre-compute all base embeddings ONCE (frozen encoder)
   - Store as numpy array to avoid recomputing each epoch

   Loss per batch:
     total = recon_loss
           + 0.001 * kl_loss          (same as before)
           + alpha * structure_loss    (new)

   Alpha schedule:
     Epoch 1-2:  alpha = 0.0   (warm up reconstruction first)
     Epoch 3-5:  alpha = 0.1   (introduce structure gently)
     Epoch 6-10: alpha = 0.5   (full structure induction)

   Per batch for structure loss:
     - Sample one ctx per batch item
     - Get its correct ending index
     - Get one random wrong ending (not the correct one)
     - Encode both with VAE head (using mu)
     - Compute structure_loss

   Log every 200 steps:
     total_loss | recon_loss | kl_loss | struct_loss |
     mean_cos_correct | mean_cos_wrong | margin_violations

   Save best model by lowest structure_loss to checkpoints/vae_structured.pt
   Also save final model regardless.

   Train 10 epochs.

   After training print:
     Initial:  mean_cos_correct vs mean_cos_wrong (epoch 1)
     Final:    mean_cos_correct vs mean_cos_wrong (epoch 10)
     Gap:      cos_correct - cos_wrong (should be > 0 and growing)
     Margin violations: what % of pairs still violate the margin

3. After training run a quick geometry check:
   scripts/check_structured_geometry.py:

   Load vae_structured.pt
   Encode 500 val examples (ctx + all 4 endings)
   For each example compute:
     cos(ctx, correct_ending)
     cos(ctx, wrong_ending_1)
     cos(ctx, wrong_ending_2)
     cos(ctx, wrong_ending_3)
     rank of correct ending by cosine similarity (1=best, 4=worst)

   Print:
     Mean cos(ctx, correct):  XX.XXXX
     Mean cos(ctx, wrong):    XX.XXXX
     Gap:                     XX.XXXX
     Correct ending rank 1:   XX.X%   <- this is no_navigation accuracy
     Correct ending rank 2:   XX.X%
     Correct ending rank 3:   XX.X%
     Correct ending rank 4:   XX.X%

   Compare against original VAE (from experiments/results_honest.json):
     Original no_navigation accuracy: 29.73% (on 5,042 examples)

   Save to experiments/structured_geometry.json

After completion print DONE block with:
  - Final mean_cos_correct vs mean_cos_wrong
  - Gap between them
  - Correct-ending-rank-1 accuracy on 500 val examples
  - Whether this beats original VAE no_navigation (29.73%)
  - Margin violation % at end of training

Do not proceed to TASK 14 until confirmed.

---

## TASK 14 — Full Evaluation of Structured VAE
**Status: WAITING**

### Instructions

1. Build a new memory store from the structured VAE:
   scripts/build_structured_store.py:
   - Load vae_structured.pt
   - Encode all 39,905 train ctx strings
   - Build MemoryStore (same as before)
   - Save to checkpoints/memory_store_structured.pkl
   - Print encoding time

2. Run full evaluation on 10,042 val examples:
   scripts/run_structured_eval.py:

   Run these conditions on ALL 10,042 val examples:

   a) no_navigation with ORIGINAL vae
      (cached: 31.31% on 10,042 examples)

   b) no_navigation with STRUCTURED vae
      (encode val with vae_structured, score endings geometrically)

   c) navigation (lambda=0.25) with STRUCTURED vae + structured store
      (use the best lambda from TASK 10 honest eval)

   d) GPT-2 small
      (cached: 28.58% on 10,042 examples)

   Print table:
     Condition                        | Accuracy | vs GPT-2 | p-value
     no_nav original VAE (cached)     | 31.31%   | +2.73    | <0.0001
     no_nav STRUCTURED VAE            | XX.XX%   | XX.XX    | XX
     nav structured VAE (λ=0.25)      | XX.XX%   | XX.XX    | XX
     GPT-2 small (cached)             | 28.58%   | baseline | -

   McNemar tests:
     structured no_nav vs original no_nav
     structured no_nav vs GPT-2 small
     structured nav vs structured no_nav

   Save to experiments/results_structured.json

After completion print DONE block with:
  - All accuracy numbers
  - Whether structured VAE beats original VAE
  - Whether navigation NOW helps with structured VAE
    (this is the key question)
  - p-values

Do not proceed to TASK 15 until confirmed.

---

## TASK 15 — Ablation: What Does Structure Induction Buy?
**Status: WAITING**

### Instructions

Run a margin sweep to understand sensitivity:

scripts/run_margin_sweep.py:
  Retrain VAE with different margin values: [0.1, 0.3, 0.5, 0.7, 1.0]
  For each margin:
    - Train 10 epochs (use alpha=0.5 from epoch 3)
    - Record: cos_correct, cos_wrong, gap, no_nav accuracy on 500 val
  
  Print table:
    margin | cos_correct | cos_wrong | gap   | no_nav_acc
    0.0    | (original)  |           |       | 29.73%
    0.1    | XX.XX       | XX.XX     | XX.XX | XX.XX%
    0.3    | XX.XX       | XX.XX     | XX.XX | XX.XX%
    0.5    | XX.XX       | XX.XX     | XX.XX | XX.XX%
    0.7    | XX.XX       | XX.XX     | XX.XX | XX.XX%
    1.0    | XX.XX       | XX.XX     | XX.XX | XX.XX%

  Find optimal margin.
  Save to experiments/margin_sweep.json

Also run alpha sweep (structure loss weight) at best margin:
  alpha values: [0.1, 0.3, 0.5, 1.0, 2.0]
  Same metrics.
  Save to experiments/alpha_sweep.json

After completion print DONE block with:
  - Best margin and alpha combination
  - Best no_nav accuracy achieved
  - Whether gap correlates monotonically with accuracy
    (if yes: gap is a valid proxy metric for future tuning)

Do not proceed to TASK 16 until confirmed.

---

## TASK 16 — Isotropy Analysis and SimCSE Comparison
**Status: WAITING**

### Instructions

The original VAE may have anisotropy (all vectors pointing in similar
directions) which limits navigation. Measure and fix this.

1. scripts/measure_isotropy.py:

   Load original vae.pt and vae_structured.pt
   Encode 2000 train examples with each
   
   Measure isotropy (average cosine similarity between random pairs):
     isotropy = mean(cos(v_i, v_j)) for random i,j pairs (n=5000 pairs)
     Perfect isotropy = 0.0
     Fully collapsed = 1.0
   
   Also measure:
     - Average pairwise cosine similarity
     - Variance of singular values of the embedding matrix
       (high variance = anisotropic, low = isotropic)
     - Intrinsic dimensionality estimate
   
   Print:
     Original VAE isotropy:    XX.XXXX (0=perfect, 1=collapsed)
     Structured VAE isotropy:  XX.XXXX
     Does structure induction also improve isotropy?

2. Add SimCSE-style contrastive loss as an alternative:
   scripts/train_vae_simcse.py:

   Same text twice through encoder with different dropout masks
   = positive pair (same meaning, different representation)
   Other texts in batch = negative pairs

   L_simcse = -log(
     exp(cos(z_i, z_i+) / τ) /
     Σ_j exp(cos(z_i, z_j+) / τ)
   )

   Where τ = 0.05 (temperature)
   This is the InfoNCE / NT-Xent loss

   Train 10 epochs, save to checkpoints/vae_simcse.pt
   Report isotropy and no_nav accuracy

   Print comparison:
     Method          | Isotropy | Gap  | no_nav_acc
     Original VAE    | XX.XX    | XX.XX | 29.73%
     Structure loss  | XX.XX    | XX.XX | XX.XX%
     SimCSE loss     | XX.XX    | XX.XX | XX.XX%
     Both combined   | XX.XX    | XX.XX | XX.XX%

   Save to experiments/isotropy_analysis.json

After completion print DONE block with:
  - Isotropy scores for each method
  - Which method gives best no_nav accuracy
  - Whether isotropy correlates with accuracy

Do not proceed to TASK 17 until confirmed.

---

## TASK 17 — Update Paper Tables and Findings
**Status: WAITING**

### Instructions

Update paper/results_table.tex and paper/findings.md with new results.

Add a new TABLE 3 — Structure Induction Results:

\begin{table}[h]
\centering
\begin{tabular}{lccc}
\toprule
Method & VAE Training & Accuracy & vs Original \\
\midrule
GMN (original VAE) & Recon + KL & 29.7\% & baseline \\
GMN (structure loss) & + Margin ranking & XX.X\% & +X.X\% \\
GMN (SimCSE loss) & + Contrastive & XX.X\% & +X.X\% \\
GMN (both losses) & + Both & XX.X\% & +X.X\% \\
\midrule
GPT-2 small & Autoregressive & 28.6\% & -X.X\% \\
\bottomrule
\end{tabular}
\caption{Effect of structure induction on HellaSwag accuracy.
All GMN variants use 624K parameters. Structure losses induce
geometric organization where correct endings are closer to their
contexts than wrong endings, making navigation trivially readable
from local geometry.}
\label{tab:structure}
\end{table}

Add to paper/findings.md:

## Finding 6: Structure Induction
Inducing geometric structure via margin ranking loss
improved no-navigation accuracy from 29.7% to XX.X%,
confirming that the latent space geometry — not the
navigation mechanism — is the primary bottleneck.
The correct ending cosine gap increased from XX.XX
(original) to XX.XX (structured), directly predicting
the accuracy improvement.

## Finding 7: Navigation With Structured Space
With a structured latent space, navigation (λ=0.25)
[helped/did not help] compared to no-navigation,
suggesting that [induced structure makes navigation
cues readable / the task does not require relational
navigation even with structured geometry].

Save updated paper/results_table.tex and paper/findings.md

---

## HOW TO USE WITH CLINE

Paste this at the start of each session:

"Read README.md, TASKS.md and TASKS_V2.md carefully.
TASKS 1-12 from TASKS.md are complete and verified.
The key finding: no_navigation with original VAE scores 29.73%
on honest eval, beating GPT-2 small (25.60%) at 199x fewer params.
Navigation adds no significant benefit with the current VAE.

Hypothesis for V2: the VAE latent space lacks induced structure.
Adding a margin ranking loss should force correct endings
geometrically closer to their contexts, making navigation trivial.

Execute TASK [N] only.
After completion print a DONE block.
Do not proceed until confirmed."

Replace [N] with current task number.