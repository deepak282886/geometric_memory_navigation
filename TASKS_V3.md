# TASKS V3 — From Scratch: GPT-2 vs GMN

Build a clean environment that trains two architectures
from scratch on identical data and compares them head to head.

Same parameters. Same data. Same compute.
Architecture is the only variable.

**The claim to prove:**
> Geometric contrastive training outperforms autoregressive
> next-token prediction on reasoning tasks at equal parameter count.

Execute one task at a time. Print DONE block after each. Wait for confirmation.

---

## TASK 18 — Environment Setup & Data
**Status: CURRENT**

### Instructions

1. Create the directory structure:

```
environment/
├── tokenizer/
│   ├── __init__.py
│   └── bpe.py
├── models/
│   ├── __init__.py
│   ├── transformer.py      # shared transformer blocks
│   ├── gpt2_style.py       # decoder (causal attention)
│   └── gmn_style.py        # encoder (bidirectional attention)
├── training/
│   ├── __init__.py
│   ├── train_gpt2.py
│   └── train_gmn.py
├── data/
│   ├── __init__.py
│   ├── arc/
│   ├── conceptnet/
│   └── loader.py
├── eval/
│   ├── __init__.py
│   ├── eval_gpt2.py
│   └── eval_gmn.py
├── compare.py
├── config.py               # all hyperparameters in one place
└── README_V3.md
```

2. Write environment/config.py:

```python
# ALL hyperparameters in one place
# Both models use identical values

MODEL = dict(
    n_layers     = 6,
    n_heads      = 8,
    hidden_dim   = 256,
    ffn_dim      = 1024,
    max_seq_len  = 128,
    vocab_size   = 8000,    # set after tokenizer training
    dropout      = 0.1,
)

TRAIN = dict(
    batch_size   = 32,
    lr           = 1e-3,
    epochs       = 20,
    warmup_steps = 200,
    grad_clip    = 1.0,
    seed         = 42,
)

GMN = dict(
    temperature  = 0.07,
    lambda_nav   = 0.0,     # start with no navigation
    top_k        = 10,
)

PATHS = dict(
    arc_train    = "environment/data/arc/train.json",
    arc_val      = "environment/data/arc/val.json",
    arc_test     = "environment/data/arc/test.json",
    conceptnet   = "environment/data/conceptnet/triples.json",
    tokenizer    = "environment/tokenizer/bpe_vocab.json",
    gpt2_ckpt    = "environment/checkpoints/gpt2_style.pt",
    gmn_ckpt     = "environment/checkpoints/gmn_style.pt",
    results      = "environment/results/comparison.json",
)
```

3. Download and preprocess ARC dataset:
   environment/data/loader.py

   Download ARC from:
   https://huggingface.co/datasets/ai2_arc
   
   Load ARC-Challenge split:
     train: ~1119 examples
     validation: ~299 examples
     test: ~1172 examples
   
   Each example has:
     question: str
     choices: list of 4 strings (A/B/C/D)
     answerKey: str (A/B/C/D)
   
   Convert answerKey to int label (A=0, B=1, C=2, D=3)
   
   Save as:
     environment/data/arc/train.json
     environment/data/arc/val.json
     environment/data/arc/test.json
   
   Each record:
   {
     "question": str,
     "choices": [str, str, str, str],
     "label": int (0-3)
   }

4. Download ConceptNet triples (for GMN relational signal):
   
   Download from HuggingFace datasets:
   "conceptnet5" or use:
   wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
   
   Filter to English only (/c/en/)
   Filter to useful relations:
     /r/Causes, /r/IsA, /r/HasProperty,
     /r/PartOf, /r/UsedFor, /r/CapableOf,
     /r/CausesDesire, /r/HasPrerequisite
   
   Extract: (subject, relation, object) triples
   Save ~100K triples to:
   environment/data/conceptnet/triples.json
   
   Format: [{"subject": str, "relation": str, "object": str}]

5. Write README_V3.md explaining the experiment.

After completion print DONE block with:
  - ARC train/val/test counts
  - Label distribution (should be ~25% each)
  - ConceptNet triples count after filtering
  - Sample ARC example
  - Sample ConceptNet triple

Do not proceed to TASK 19 until confirmed.

---

## TASK 19 — BPE Tokenizer From Scratch
**Status: WAITING**

### Instructions

Build a BPE tokenizer trained on ARC + ConceptNet text only.
No pretrained tokenizer. No borrowed vocabulary.

Write environment/tokenizer/bpe.py:

1. Class BPETokenizer:

   train(texts, vocab_size=8000):
     - Start with character-level vocabulary
     - Add special tokens:
         [PAD]=0, [UNK]=1, [BOS]=2, [EOS]=3, [MASK]=4
     - Count all character pairs in texts
     - Iteratively merge most frequent pair
     - Until vocab_size reached
     - Save merge rules and vocab to config.PATHS.tokenizer
   
   encode(text) -> list[int]:
     - Apply learned merge rules
     - Return token ids
     - Max length: config.MODEL.max_seq_len
     - Truncate if longer, pad if shorter
   
   decode(ids) -> str:
     - Reverse the merge rules
     - Return text string
   
   save(path) / load(path):
     - JSON serializable format

2. Write environment/tokenizer/train_tokenizer.py:
   
   Collect all text:
     - ARC questions (all splits)
     - ARC answer choices (all splits)
     - ConceptNet triples as sentences:
         "subject relation object"
         e.g. "fire causes heat"
   
   Train BPE on this text corpus
   Target vocab_size = 8000
   
   Print after training:
     Vocab size: XXXX
     Sample tokenizations:
       "photosynthesis" → [tokens]
       "evaporation" → [tokens]
       "causes" → [tokens]
       "The trophy didn't fit" → [tokens]
     Average tokens per ARC question: XX.X

3. Verify round-trip:
   For 100 random ARC questions:
     decoded = tokenizer.decode(tokenizer.encode(q))
     assert decoded.strip() == q.strip()
   Print: round-trip success rate (target: >95%)

After completion print DONE block with:
  - Final vocab size
  - Sample tokenizations (5 examples)
  - Round-trip success rate
  - Average tokens per ARC question/answer

Do not proceed to TASK 20 until confirmed.

---

## TASK 20 — Shared Transformer Blocks
**Status: WAITING**

### Instructions

Build the shared transformer architecture.
Both GPT-2 style and GMN style use these blocks.
The ONLY difference: causal mask vs bidirectional attention.

Write environment/models/transformer.py:

1. Class MultiHeadAttention(nn.Module):
   - n_heads, hidden_dim from config
   - Q, K, V projections
   - Scaled dot-product attention
   - causal=True: mask future positions (GPT-2 style)
   - causal=False: full attention (GMN style)
   - Dropout on attention weights

2. Class FeedForward(nn.Module):
   - Linear(hidden_dim, ffn_dim) → GELU → Linear(ffn_dim, hidden_dim)
   - Dropout

3. Class TransformerBlock(nn.Module):
   - Pre-LayerNorm (more stable than post)
   - MultiHeadAttention (causal flag passed through)
   - FeedForward
   - Residual connections

4. Class TransformerBase(nn.Module):
   - Token embedding (vocab_size, hidden_dim)
   - Positional embedding (max_seq_len, hidden_dim)
   - N TransformerBlocks
   - Final LayerNorm
   - causal flag determines attention type
   - forward(token_ids) -> hidden_states (B, T, hidden_dim)

5. Write environment/models/gpt2_style.py:
   
   Class GPT2Style(nn.Module):
     - TransformerBase(causal=True)
     - LM head: Linear(hidden_dim, vocab_size, bias=False)
       weight tied to embedding (standard practice)
     - forward(token_ids) -> logits (B, T, vocab_size)
     - loss(token_ids) -> cross entropy on next token prediction
     - count_parameters() -> int

6. Write environment/models/gmn_style.py:
   
   Class GMNStyle(nn.Module):
     - TransformerBase(causal=False)
     - Projection head: Linear(hidden_dim, 256)
       projects [CLS] token to 256-dim latent
     - L2 normalize output
     - encode(token_ids) -> (B, 256) unit vectors
     - forward(question_ids, choice_ids) ->
         question_vec (B, 256)
         choice_vecs (B, 4, 256)
         scores (B, 4) = choice_vecs @ question_vec
     - count_parameters() -> int

7. Write environment/models/test_models.py:
   Instantiate both models
   Run forward pass with dummy data
   Print:
     GPT-2 style parameters: XX,XXX,XXX
     GMN style parameters:   XX,XXX,XXX
     GPT-2 output shape: (batch, seq, vocab)
     GMN scores shape:   (batch, 4)
     Both should be ~same parameter count

After completion print DONE block with:
  - GPT-2 style parameter count
  - GMN style parameter count
  - Difference (target: < 5%)
  - Forward pass shapes verified
  - No NaN in outputs

Do not proceed to TASK 21 until confirmed.

---

## TASK 21 — Training: GPT-2 Style
**Status: WAITING**

### Instructions

Write environment/training/train_gpt2.py:

Training objective: next-token prediction on ARC text.

Data preparation:
  For each ARC training example build sequences:
    [BOS] question [EOS] choice_A [EOS]
    [BOS] question [EOS] choice_B [EOS]
    [BOS] question [EOS] choice_C [EOS]
    [BOS] question [EOS] choice_D [EOS]
  
  The model sees all 4 choices as training text.
  It predicts next tokens for all of them.
  
  Also include ConceptNet triples as text:
    [BOS] subject relation object [EOS]
  
  Total sequences: ~1119*4 + ~100K = ~104K

Training loop:
  Epochs: 20
  Batch: 32
  Optimizer: AdamW lr=1e-3, weight_decay=0.01
  LR schedule: linear warmup 200 steps then cosine decay
  Gradient clip: 1.0
  
  Loss: CrossEntropy on next token prediction
        ignore [PAD] tokens in loss
  
  Log every 100 steps:
    step | loss | perplexity | lr
  
  Evaluate every epoch on val set:
    For each val question:
      Score each choice by sum of log probs
      of choice tokens given question tokens
      Predict argmax
    Report accuracy on val set
  
  Save best model by val accuracy
  Save to config.PATHS.gpt2_ckpt

After training print:
  Per-epoch: train_loss | train_ppl | val_accuracy
  Best val accuracy achieved
  Final train perplexity

After completion print DONE block with:
  - Per-epoch val accuracy (all 20 epochs)
  - Best val accuracy and epoch
  - Final train perplexity
  - Whether val accuracy > 25% (above random)

Do not proceed to TASK 22 until confirmed.

---

## TASK 22 — Training: GMN Style
**Status: WAITING**

### Instructions

Write environment/training/train_gmn.py:

Training objective: two signals combined.

Signal 1 — Contrastive scoring (primary, weight=1.0):
  For each ARC training example:
    question_vec = encode(question)          # (256,)
    choice_vecs  = encode(each choice)       # (4, 256)
    scores = choice_vecs @ question_vec      # (4,)
    loss = CrossEntropy(scores / T, label)
    T = 0.07

Signal 2 — Relational consistency (secondary, weight=0.1):
  Sample ConceptNet triple (subj, rel, obj)
  subj_vec = encode(subject)
  obj_vec  = encode(object)
  rel_vec  = encode(relation_name)
  
  Consistency loss:
    direction = obj_vec - subj_vec
    Same relation type should point same direction:
      If (A causes B) and (C causes D):
        direction(A→B) ≈ direction(C→D)
    
    Batch all triples of same relation type
    Loss = variance of direction vectors
           within each relation type
           (minimize = make directions consistent)

Combined loss:
  total = signal_1 + 0.1 * signal_2

Training loop:
  Epochs: 20
  Batch: 32
  Optimizer: AdamW lr=1e-3, weight_decay=0.01
  LR schedule: same as GPT-2 style
  
  Log every 100 steps:
    step | contrastive_loss | relational_loss | val_acc
  
  Evaluate every epoch:
    For each val question:
      scores = choice_vecs @ question_vec
      predict = argmax(scores)
    Report accuracy
  
  Save best model by val accuracy
  Save to config.PATHS.gmn_ckpt

After training print:
  Per-epoch: contrastive_loss | relational_loss | val_accuracy
  Best val accuracy and epoch
  Whether relational loss decreased (directions became consistent)

After completion print DONE block with:
  - Per-epoch val accuracy (all 20 epochs)
  - Best val accuracy and epoch
  - Contrastive loss curve
  - Relational loss curve (did directions become consistent?)
  - Whether GMN val accuracy > GPT-2 val accuracy

Do not proceed to TASK 23 until confirmed.

---

## TASK 23 — Geometry Visualization
**Status: WAITING**

### Instructions

Before evaluating on test set, verify the geometry
is actually meaningful.

Write environment/eval/visualize_geometry.py:

1. Load trained GMN encoder
   Encode all ARC val questions + all choices
   
2. UMAP visualization:
   
   Plot A — Question space:
     Encode all 299 val questions
     UMAP to 2D
     Color by correct answer index (0/1/2/3)
     Save to environment/results/question_space.png
     
     If geometry is good: questions with same
     correct answer position should NOT cluster
     (the question, not the position, determines the answer)

   Plot B — Answer geometry:
     For each val question encode question + 4 choices
     In 2D UMAP space draw:
       Question as center point (red)
       Correct choice as green dot
       Wrong choices as grey dots
     Show 20 random examples
     Save to environment/results/answer_geometry.png
     
     If geometry is good: green dots should be
     consistently closer to red than grey dots

   Plot C — Relation directions:
     For 5 relation types from ConceptNet
     Sample 20 triples each
     Encode subjects and objects
     Draw arrows from subject to object
     Save to environment/results/relation_directions.png
     
     If relational training worked:
       Arrows of same relation type should
       point in similar directions (consistent)
       Arrows of different types should differ

3. Compute geometry metrics:
   
   Mean cos(question, correct_choice):  XX.XXXX
   Mean cos(question, wrong_choice):    XX.XXXX
   Gap:                                 XX.XXXX
   Gap std:                             XX.XXXX
   
   Per relation type direction consistency:
     (mean cosine similarity between direction
      vectors of same relation type)
     Causes:      XX.XXXX
     IsA:         XX.XXXX
     HasProperty: XX.XXXX
     (higher = more consistent = relational training worked)
   
   Save to environment/results/geometry_metrics.json

After completion print DONE block with:
  - All geometry metrics
  - Whether relation directions are consistent
    (target: same-relation cosine > 0.3)
  - Subjective assessment of UMAP plots
  - Path to saved plots

Do not proceed to TASK 24 until confirmed.

---

## TASK 24 — Head To Head Evaluation
**Status: WAITING**

### Instructions

Write environment/compare.py:

Evaluate BOTH models on ARC-Challenge TEST set (1172 examples).

Neither model has seen the test set during training.

For GPT-2 style:
  For each test question:
    For each choice k:
      full_text = [BOS] question [EOS] choice_k [EOS]
      score_k = sum of log probs of choice_k tokens
                given question tokens
    predict = argmax(score_0, score_1, score_2, score_3)

For GMN style:
  For each test question:
    question_vec = encode(question)
    choice_vecs  = encode(each choice)
    scores = choice_vecs @ question_vec
    predict = argmax(scores)

Print comparison table:

  Model         | Params  | Val Acc | Test Acc | Training Objective
  ------------- |---------|---------|----------|-----------------------
  Random        | --      | 25.0%   | 25.0%    | --
  GPT-2 style   | XX.XM   | XX.X%   | XX.X%    | Next-token prediction
  GMN style     | XX.XM   | XX.X%   | XX.X%    | Contrastive + relational

McNemar test on test set:
  GMN vs GPT-2: p = XX.XXXX

Also run per-category analysis:
  ARC has question categories.
  Report accuracy per category for both models.
  Where does GMN beat GPT-2?
  Where does GPT-2 beat GMN?

Save full results to environment/results/comparison.json

After completion print DONE block with:
  - Both test accuracies
  - McNemar p-value
  - Parameter counts
  - Per-category breakdown
  - The headline: does GMN beat GPT-2 on ARC-Challenge?

Do not proceed to TASK 25 until confirmed.

---

## TASK 25 — Add Navigation To GMN
**Status: WAITING**

### Instructions

The trained GMN encoder + memory navigation.

1. Build memory store from ARC training set:
   - Encode all 1119 train questions with trained GMN encoder
   - Store with correct answer label as outcome weight
   - FAISS index
   - Save to environment/checkpoints/arc_memory_store.pkl

2. Add navigation to evaluation:
   environment/eval/eval_gmn_nav.py

   For each test question:
     question_vec = encode(question)
     
     # Navigation
     neighbors = store.query(question_vec, k=10)
     nav_vector = fuse_directions(neighbors, top_k=3,
                                  use_tangent=True,
                                  query_vector=question_vec)
     
     # Additive scoring (proven best from earlier work)
     choice_vecs = encode(choices)
     ctx_scores = choice_vecs @ question_vec
     nav_scores = choice_vecs @ nav_vector
     scores = ctx_scores + lambda_nav * nav_scores
   
   Sweep lambda: [0.0, 0.25, 0.5, 1.0, 2.0]
   
   Print table:
     lambda | val_acc | test_acc
     0.00   | XX.X%   | XX.X%
     0.25   | XX.X%   | XX.X%
     0.50   | XX.X%   | XX.X%
     1.00   | XX.X%   | XX.X%
     2.00   | XX.X%   | XX.X%

Key question: with a properly trained encoder
(not frozen/borrowed), does navigation help now?

Save to environment/results/navigation_results.json

After completion print DONE block with:
  - Lambda sweep results
  - Best lambda and accuracy
  - Whether navigation helps (p-value vs lambda=0)
  - Comparison vs GPT-2 style at best lambda

Do not proceed to TASK 26 until confirmed.

---

## TASK 26 — Final Paper Table
**Status: WAITING**

### Instructions

Generate the definitive comparison table.

environment/results/final_table.tex:

\begin{table}[h]
\centering
\begin{tabular}{lcccc}
\toprule
Model & Params & Training Objective & Val & Test \\
\midrule
Random & -- & -- & 25.0\% & 25.0\% \\
\midrule
GPT-2 style (ours) & XXM & Next-token prediction & XX.X\% & XX.X\% \\
GMN (no nav) & XXM & Contrastive & XX.X\% & XX.X\% \\
GMN + navigation & XXM & Contrastive + relational & XX.X\% & XX.X\% \\
\bottomrule
\end{tabular}
\caption{ARC-Challenge results. Both models trained from scratch
on identical data with identical parameter budgets. Architecture
and training objective are the only variables. McNemar p-value
(GMN vs GPT-2): p = X.XXXX.}
\label{tab:arc}
\end{table}

Also write environment/results/v3_findings.md:

## V3 Findings

### Finding 1: Training Objective Matters More Than Architecture
[fill from results]

### Finding 2: Navigation With Trained Encoder
[fill from results — does it help now?]

### Finding 3: Relational Consistency
[did ConceptNet training induce consistent directions?]

### Finding 4: Per-Category Analysis
[where does geometric training help most?]

---

## HOW TO USE WITH CLINE

Paste at start of each session:

"Read README.md, TASKS.md, TASKS_V2.md and TASKS_V3.md.
TASKS 1-17 from TASKS.md/V2 are complete.

V3 is a new controlled experiment:
  Train GPT-2 style AND GMN style from scratch
  on identical data (ARC-Challenge + ConceptNet)
  with identical parameter budgets (~22M each).
  
  Architecture and training objective
  are the ONLY variables.
  
  Goal: prove geometric contrastive training
  outperforms autoregressive next-token prediction
  on reasoning tasks.

Execute TASK [N] from TASKS_V3.md only.
Print DONE block after completion.
Do not proceed until confirmed."

Replace [N] with current task number (start at 18).