# TASKS

Execute one task at a time. Do not proceed to the next task until instructed.
After each task print a DONE summary with key outputs and shapes.

---

## TASK 1 — Setup & Data
**Status: DONE**

### Instructions
1. Create requirements.txt with:
```
torch
transformers
sentence-transformers
datasets
faiss-cpu
scikit-learn
umap-learn
matplotlib
wandb
tqdm
numpy
```

2. Create .env.example with:
```
WANDB_API_KEY=your_key_here
```

3. Create all __init__.py files in src/ subdirectories

4. Write scripts/load_data.py that:
   - Loads HellaSwag from HuggingFace datasets
   - Takes first 2000 examples from train split
   - Takes first 500 examples from validation split
   - For each example store:
     - ctx: the context string (ctx field)
     - endings: list of 4 ending strings
     - label: correct ending index (integer 0-3)
     - activity_label: activity label string
   - Saves to data/train.json and data/val.json
   - Prints 2 sample examples to verify structure
   - Prints label distribution to verify balance

### Verify Before Proceeding
- data/train.json exists with 2000 examples
- data/val.json exists with 500 examples
- Each example has ctx, endings (list of 4), label (0-3), activity_label
- Print one full example

---

## TASK 2 — VAE Encoder
**Status: DONE**

### Instructions
Write src/autoencoder/vae.py with:

1. Class `TextEncoder`:
   - Loads sentence-transformers `all-MiniLM-L6-v2` as base
   - Forward: text string → 384-dim base embedding (frozen)

2. Class `VAEHead`:
   - Input: 384-dim base embedding
   - FC layer → 512 → mu (256-dim) and logvar (256-dim)
   - Reparameterization trick: z = mu + eps * exp(0.5 * logvar)
   - Output: z (256-dim latent vector), mu, logvar

3. Class `VAEDecoder`:
   - Input: 256-dim latent vector
   - FC layers: 256 → 384
   - Output: reconstructed 384-dim embedding

4. Class `VAE` combining all three:
   - encode(text) → z, mu, logvar
   - decode(z) → reconstructed embedding
   - forward(text) → z, mu, logvar, reconstructed embedding

5. Function `vae_loss(reconstructed, original, mu, logvar)`:
   - reconstruction_loss = MSE(reconstructed, original)
   - kl_loss = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
   - total = reconstruction_loss + 0.001 * kl_loss
   - return total, reconstruction_loss, kl_loss

6. Write scripts/train_vae.py:
   - Load data/train.json
   - Concatenate ctx + " " + each ending as training texts
   - Also encode just ctx alone
   - Total training texts ~ 2000 * 5 = ~10000 strings
   - Train VAE for 10 epochs batch size 64
   - Log loss every 100 steps
   - Save to checkpoints/vae.pt
   - Print final loss

### Verify Before Proceeding
- Run scripts/train_vae.py
- Loss should decrease
- Print shape of z: should be (batch, 256)
- Encode one text string and print z shape

---

## TASK 3 — Latent Space Visualization
**Status: DONE**

### Instructions
Write scripts/visualize_latent.py that:

1. Loads trained VAE from checkpoints/vae.pt
2. Encodes all 2000 training ctx strings → 256-dim vectors
3. Runs UMAP: 256-dim → 2-dim
4. Plots scatter plot:
   - Color by activity_label
   - Each unique activity label gets a different color
   - Add legend
   - Title: "Latent Space by Activity Label"
5. Saves to notebooks/latent_space.png
6. Prints: number of unique activity labels found

### Verify Before Proceeding
- notebooks/latent_space.png exists
- Plot shows some clustering structure (not pure noise)
- If completely random noise — VAE needs more training epochs, go back to Task 2
- If reasonable clusters visible — proceed

---

## TASK 4 — Memory Store
**Status: DONE**

### Instructions
Write src/memory/store.py with:

1. Class `MemoryStore`:

   `__init__(vae, train_data)`:
   - Encode all train ctx strings with VAE → 256-dim vectors
   - Store as numpy array: self.vectors (2000, 256)
   - Store self.labels (correct answer index per example)
   - Store self.activity_labels (string per example)
   - Store self.endings (4 endings per example)
   - Build FAISS IndexFlatIP (inner product) index
   - Add all vectors to index
   - Print: "Memory store built with N vectors"

   `query(vector, k=10)`:
   - Input: 256-dim query vector (numpy)
   - FAISS search top-k
   - Return list of dicts:
     ```
     {
       vector: 256-dim numpy array,
       label: correct answer index,
       activity_label: string,
       endings: list of 4 strings,
       distance: float,
       outcome_weight: 1.0 if label matches query label else 0.1
     }
     ```
   - Note: outcome_weight at query time is 1.0 for all
     (we don't know query label at inference)
     Use 1.0 for all neighbors at inference time
     Use actual outcome weights only during training

   `get_outcome_weight(memory_label, query_label)`:
   - Returns 1.0 if memory_label == query_label else 0.1

2. Write scripts/test_store.py:
   - Build memory store from train data
   - Pick 3 random val examples
   - Query with their ctx vectors
   - Print top-3 neighbors for each
   - Print their activity labels vs query activity label
   - Verify neighbors make semantic sense

### Verify Before Proceeding
- Memory store builds without error
- Query returns 10 neighbors
- Neighbors look semantically related to query
- Print one full query result

---

## TASK 5 — Parallel Jev Sampler
**Status: DONE**

### Instructions
Write src/sampler/jev.py with:

1. Function `cosine_similarity(a, b)`:
   - Pure numpy
   - Input: two 256-dim vectors
   - Output: scalar similarity

2. Function `sample_memory(query_vector, memory, outcome_weight)`:
   - Computes weight = cosine_similarity(query, memory.vector) * outcome_weight
   - Computes direction = memory.vector - query_vector
   - Returns (weight, direction)

3. Function `parallel_sample(query_vector, memories, outcome_weights)`:
   - Input:
     - query_vector: 256-dim numpy array
     - memories: list of memory dicts from store.query()
     - outcome_weights: list of floats (1.0 or 0.1)
   - Uses ThreadPoolExecutor to process each memory in parallel
   - Calls sample_memory for each
   - Returns list of (weight, direction) tuples
   - Print: sampled N memories in parallel

### Verify Before Proceeding
- Write scripts/test_sampler.py
- Load store, query one example, run parallel_sample
- Print weights and direction norms for each memory
- Verify weights are positive floats
- Verify directions are 256-dim vectors

---

## TASK 6 — Direction Fusion
**Status: DONE**

### Instructions
Write src/fusion/fuse.py with:

1. Function `fuse_directions(weighted_directions, top_k=3)`:
   - Input: list of (weight, direction) tuples
   - Sort by weight descending
   - Take top_k
   - Normalize weights: weights = weights / sum(weights)
   - Weighted sum: navigation_vector = sum(w * d for w, d in top_k)
   - Normalize result to unit vector
   - Return 256-dim navigation vector

2. Function `navigate(query_vector, navigation_vector)`:
   - Returns concatenation of [query_vector, navigation_vector]
   - Output: 512-dim vector
   - This is the input to the classifier

### Verify Before Proceeding
- Write scripts/test_fusion.py
- Run full pipeline: encode → query → sample → fuse → navigate
- Print shape of final navigation output: should be (512,)
- Verify no NaN or inf values

---

## TASK 7 — Classifier Head
**Status: DONE (head replaced by geometric ending scoring; see TASK 8)**

### Instructions
Write src/classifier/head.py with:

1. Class `ClassifierHead(nn.Module)`:
   - Input: 512-dim (concatenated position + navigation)
   - Layers:
     ```
     Linear(512, 256)
     ReLU
     Dropout(0.1)
     Linear(256, 128)
     ReLU
     Dropout(0.1)
     Linear(128, 4)
     ```
   - Forward: returns raw logits (4-dim)
   - No softmax in forward (use CrossEntropyLoss which includes it)

2. Write scripts/train_classifier.py:

   Full training loop:
   - Load data/train.json
   - Load VAE from checkpoints/vae.pt (frozen — do not train)
   - Build memory store from train data
   - Initialize ClassifierHead
   - For each training example:
     - Encode ctx with VAE → query_vector
     - Query memory store top-10
     - Assign outcome_weights based on whether memory label == true label
     - Parallel sample
     - Fuse directions
     - Navigate: concat [query_vector, nav_vector]
     - Classifier forward → logits
     - CrossEntropyLoss against true label
   - Train for 20 epochs
   - Batch size: 32 (accumulate gradients)
   - Optimizer: Adam lr=1e-3
   - Log loss every 50 steps
   - Save best model to checkpoints/classifier.pt
   - Print train accuracy per epoch

### Verify Before Proceeding
- Training runs without error
- Loss decreases over epochs
- Train accuracy should reach at least 40%+ by epoch 10
- If stuck at 25% (random) — there is a bug, do not proceed

---

## TASK 8 — Evaluation
**Status: DONE**

### Instructions
Write src/eval/evaluate.py with:

1. Function `evaluate_gmn(val_data, vae, store, classifier)`:
   - For each val example:
     - Encode ctx → query_vector
     - Query store top-10
     - Parallel sample (outcome_weights all 1.0 at inference)
     - Fuse
     - Navigate
     - Classifier forward → argmax → predicted label
   - Return accuracy

2. Function `evaluate_random(val_data)`:
   - Random choice from 0-3 for each example
   - Return accuracy (should be ~25%)

3. Function `evaluate_gpt2(val_data)`:
   - Load GPT-2 small from HuggingFace
   - For each example score each of 4 endings:
     - Concatenate ctx + ending
     - Get GPT-2 log probability of ending given ctx
     - Pick highest log probability ending
   - Return accuracy

4. Write scripts/run_eval.py:
   - Run all three evaluations on same 500 val examples
   - Print results table:
     ```
     Model          | Params  | Accuracy
     Random         | -       | XX.X%
     GPT-2 small    | 117M    | XX.X%
     GMN (ours)     | <10M    | XX.X%
     ```
   - Count GMN parameters and print
   - Save results to experiments/results.json

### Verify Before Proceeding
- Random baseline ~25% ✓
- GPT-2 scores ~50% ✓
- GMN scores reported
- If GMN < 30% there is a problem — check sampler and fusion

---

## TASK 9 — Ablations
**Status: DONE**

### Instructions
Write scripts/run_ablations.py:

Run 4 ablation conditions on val set:

1. `no_navigation`:
   - Skip sampler and fusion
   - Classifier input: concat([query_vector, zeros(256)])
   - Reports accuracy without navigation

2. `uniform_weights`:
   - Set all outcome_weights = 1.0 during training
   - Retrain classifier with uniform weights
   - Reports accuracy without outcome weighting

3. `random_fusion`:
   - In fuse_directions: pick random 3 instead of top-3
   - Reports accuracy without top-k selection

4. `no_hyperedge_bonus`:
   - Remove activity_label bonus from weights
   - Reports accuracy without relational structure

5. `full_model`:
   - All components active
   - Reports full accuracy

Print results table:
```
Ablation              | Accuracy | Delta vs Full
No navigation         | XX.X%    | -X.X%
Uniform weights       | XX.X%    | -X.X%
Random fusion         | XX.X%    | -X.X%
No hyperedge bonus    | XX.X%    | -X.X%
Full model            | XX.X%    | baseline
```

Save to experiments/ablations.json

---

## TASK 10 — Navigation Visualization
**Status: DONE (adapted for the additive model — see note below)**

### Note on the adaptation
The additive scorer never MOVES the query, so there is no landing position:
`argmax(cos(ending, ctx) + lambda * cos(ending, nav_vector))` reads the navigation
vector as a *direction*, not a destination. The shipped figure therefore draws the
navigation vector as an arrow from the query and shows the ending vectors directly.
See also STEP 1 (`scripts/run_lambda_honest.py`), which showed lambda is not
reliably tunable (tune-set spread 0.90 points vs 1.27 points of eval-set swing).

### Instructions
Write scripts/visualize_navigation.py:

1. Pick 5 val examples (seeded RNG over the full 10,042-example val split)
2. For each:
   - Encode ctx → query_vector (unit-norm)
   - Query top-10 memories, fuse top-3 → navigation_vector
   - navigation target = query_vector + navigation_vector (arrow endpoint only)
3. Collect all points:
   - query positions (red dots)
   - top-3 memory neighbour positions (grey dots)
   - correct ending vectors (green dots)
   - wrong ending vectors (light grey dots)
   - Draw arrows from query → navigation target
4. Run UMAP on all 45 collected vectors → 2D
5. Plot with:
   - Red: query position
   - Grey: memory neighbour positions
   - Green: correct ending vectors
   - Light grey: wrong ending vectors
   - Blue arrows: navigation direction
   - Title: "Geometric Navigation in Latent Space"
6. Save to notebooks/navigation.png

---

## TASK 11 — Parameter Count & Model Summary
**Status: DONE**

### Instructions
Write scripts/model_summary.py:

Count parameters in each component:
- VAE encoder head (not base sentence-transformer)
- VAE decoder
- Classifier head

Print:
```
Component          | Parameters
VAE head           | X,XXX,XXX
VAE decoder        | X,XXX,XXX
Classifier head    | X,XXX,XXX
Total (GMN)        | X,XXX,XXX
GPT-2 small        | 117,000,000

GMN is XX% the size of GPT-2 small
```

Target: GMN total < 10,000,000 parameters

---

## TASK 12 — Paper Results Table
**Status: DONE (adapted - see note below)**

### Instructions
Write scripts/generate_paper_table.py:

Load experiments/results.json and experiments/ablations.json

Print LaTeX table ready to paste into paper:

```latex
\begin{table}[h]
\centering
\begin{tabular}{lcc}
\toprule
Model & Parameters & HellaSwag Acc. \\
\midrule
Random baseline & -- & 25.0\% \\
GPT-2 small & 117M & XX.X\% \\
\midrule
GMN w/o navigation & XM & XX.X\% \\
GMN w/o outcome weights & XM & XX.X\% \\
GMN w/o top-k fusion & XM & XX.X\% \\
GMN full model & XM & \textbf{XX.X\%} \\
\bottomrule
\end{tabular}
\caption{Accuracy on HellaSwag validation subset (500 examples).}
\label{tab:results}
\end{table}
```

Save to paper/results_table.tex

### Note on the adaptation
The shipped tables supersede this sketch:

- `experiments/results.json` and `ablations.json` describe the superseded
  classifier / geometric-recall ablations, so the tables instead source the
  navigation experiments (`results_honest.json`, `results_full_val.json`,
  `results_tangent.json`, `results_continuation.json`, `results_additive.json`).
- Table 1 uses the honest held-out split (5,042 examples, lambda tuned on a
  disjoint 5,000-example split).
- Table 2 is split into two panels, because chord/tangent/continuation were
  measured on the full 10,042 set while the additive model was measured on the
  honest 5,042 sub-split. Each panel compares against its own same-set baseline,
  so no cross-set delta is reported.
- GPT-2 small is counted at the measured 124,439,808 parameters, not the quoted
  117M.

Also written: `experiments/paper_tables.json` and `paper/findings.md`.

---

## HOW TO USE WITH CLINE

Paste this at the start of each Cline session:

```
Read README.md and TASKS.md carefully.
Understand the full architecture before writing any code.
Execute TASK [N] only.
After completion print a DONE block with:
  - Files created
  - Key output shapes
  - Any numbers printed
Do not proceed to TASK [N+1] until I confirm.
```

Replace [N] with current task number.