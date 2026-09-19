# Geometric Memory Navigation

## Core Idea
Input is encoded as a position in a frozen 256-d latent space.
Past experiences (memories) are stored as latent vectors.
kNN retrieves the nearest memories; each memory suggests a direction to navigate.
Directions are fused into a single navigation vector.
The correct ending is then chosen by **cosine similarity**, not by a classifier:

```
score(ending) = cos(ending, ctx) + lambda_nav * cos(ending, navigation_vector)
```

## Architecture (as built)
1. VAE: `all-MiniLM-L6-v2` (frozen) 384-d → VAE head/decoder → 256-d latent
2. Memory store: 39,905 HellaSwag train examples as unit-norm latent vectors
3. kNN: top-10 nearest neighbours (FAISS)
4. Parallel sampler: (weight, direction) per memory, `target=correct_ending`
5. Fusion: top-3 weighted sum, orthogonalised against the query (`use_tangent=True`)
6. Scoring: additive cosine similarity over the 4 candidate endings

Total parameters: **624,256** (199x smaller than GPT-2 small's 124,439,808).

## Results

Honest validation numbers — `lambda` chosen on the first 5,000 val examples,
accuracy reported only on the remaining 5,042 (`scripts/run_lambda_honest.py`):

| Condition | Accuracy | vs GPT-2 |
|---|---|---|
| no_navigation (lambda = 0) | 29.73% | +4.13 |
| GMN (lambda = 0.50, honest) | **28.58%** | +2.98 |
| GPT-2 small (124M params) | 25.60% | baseline |

McNemar exact tests on those 5,042 examples:

| Comparison | p | Significant |
|---|---|---|
| GMN vs no_navigation | 0.0616 | no |
| GMN vs GPT-2 small | 0.0009 | yes |
| no_navigation vs GPT-2 small | <0.0001 | yes |

### What this means
- GMN with 624K parameters significantly beats GPT-2 small (124M) — the headline claim.
- **Navigation does not significantly help yet.** Its best setting loses 1.15 points
  to the context-only baseline, and lambda is not reliably tunable: the tune-set
  spread across lambda is 0.90 points while the eval set swings 1.27 points, so the
  chosen lambda is noise. `results_additive.json`'s 31.49% is therefore optimistic
  (it tuned lambda on the same examples it reported); 28.58% is the honest figure.
- An earlier apparent "sqrt(2) blend penalty" was an illusion: `argmax(ctx+nav)` and
  `argmax((ctx+nav)/sqrt(2))` are identical, so all of the (noise-level) gain came
  from down-weighting navigation, not from normalisation.

## Dataset
HellaSwag via HuggingFace `datasets`: full train (39,905) and val (10,042) splits.
`data/{train,val}.json` are the first 2000/500 rows, kept for the early tasks;
`data/{train_full,val_full}.json` are the full splits used for all reported numbers.
Note: HellaSwag's own `activity_label` field mixes fine-grained WikiHow titles
(e.g. "Baking cookies") with coarse categories (e.g. "Family Life"), which matters
when interpreting neighbourhood-overlap diagnostics but not accuracy.

## Stack
- Python 3.12.6, torch 2.14.0+cpu
- sentence-transformers 6.1.0, transformers 5.17.0
- datasets 5.0.1, faiss-cpu 1.15.1
- umap-learn 0.5.12, matplotlib 3.11.2

## Key Scripts
| Script | Purpose |
|---|---|
| `scripts/load_full_data.py`, `scripts/load_full_val.py` | fetch full splits |
| `scripts/train_vae.py` | train the VAE (TASK 2) |
| `scripts/visualize_latent.py` | TASK 3 latent-space plot |
| `scripts/build_full_store.py` | build 39,905-example memory store |
| `scripts/precompute_val_full_vectors.py` | cache val latents |
| `scripts/full_val_eval.py` | chord/landing evaluation (TASK 8) |
| `scripts/run_ablations.py`, `scripts/run_{tangent,continuation,additive}_eval.py` | ablations |
| `scripts/run_lambda_honest.py` | **STEP 1: honest lambda tune/eval split** |
| `scripts/visualize_navigation.py` | **TASK 10: navigation figure** |
| `scripts/gpt2_full_val.py` | GPT-2 small baseline (detached, ~24 min) |
| `scripts/model_summary.py` | **TASK 11: exact parameter counts** |
| `scripts/generate_paper_table.py` | **TASK 12: paper tables + `paper/findings.md`** |

## Tasks
See TASKS.md
