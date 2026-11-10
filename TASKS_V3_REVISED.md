# TASKS V3 REVISED — Pretraining from Scratch: GPT-2 vs GMN

**The controlled experiment:**

Same data: WikiText-103 (103M tokens)
Same parameter count: ~22M each
Same compute budget
Different objective only

GPT-2 style: next-token prediction
GMN style:   next-sentence contrastive prediction

Evaluate both on HellaSwag (existing baselines ready).

**The claim to prove:**
> Geometric next-sentence prediction outperforms
> autoregressive next-token prediction on reasoning tasks
> at equal parameter count and equal training data.

Stop all current training. Kill the ARC training job.
Execute one task at a time. Print DONE block. Wait for confirmation.

---

## TASK 18R — Stop ARC Training, Setup WikiText-103
**Status: CURRENT**

### Instructions

1. Stop any running ARC training:
   Kill any running train_gpt2.py process.
   Delete or archive logs/gpt2_train.out.log

2. Download WikiText-103:

   from datasets import load_dataset
   ds = load_dataset("wikitext", "wikitext-103-raw-v1")

   Splits:
     train: ~103M tokens, ~1.8M sentences
     validation: ~217K tokens
     test: ~245K tokens

3. Preprocess into sentences:
   environment/data/wikitext/preprocess.py

   For each document in train split:
     Split into sentences by period/newline
     Clean: strip whitespace, skip empty, skip < 5 words
     Keep sentence pairs: (S_n, S_n+1) within same document
     Do NOT pair across document boundaries

   Save:
     environment/data/wikitext/train_sentences.json
       list of sentences (for GPT-2 LM training)
     environment/data/wikitext/train_pairs.json
       list of [S_n, S_n+1] pairs (for GMN training)
     environment/data/wikitext/val_sentences.json
     environment/data/wikitext/val_pairs.json

   Target after cleaning:
     train: ~500K-1M sentences
     val: ~10K-20K sentences

4. Update environment/config.py:

   Add WIKITEXT section:
   WIKITEXT = dict(
       train_sentences = "environment/data/wikitext/train_sentences.json",
       train_pairs     = "environment/data/wikitext/train_pairs.json",
       val_sentences   = "environment/data/wikitext/val_sentences.json",
       val_pairs       = "environment/data/wikitext/val_pairs.json",
   )

   Update TRAIN section:
       batch_size   = 64,     # larger for in-batch negatives
       lr           = 3e-4,   # standard for pretraining
       epochs       = 10,     # pretraining needs fewer epochs
       warmup_steps = 1000,   # longer warmup for pretraining
       grad_clip    = 1.0,
       seq_len      = 64,     # sentence-level, not doc-level

5. Retrain BPE tokenizer on WikiText-103:
   environment/tokenizer/train_tokenizer_wiki.py

   Corpus: sample 500K sentences from train_sentences.json
   (do not use all 1M — BPE training will take too long)
   vocab_size = 16000   (larger vocab for general text)
   Save to environment/tokenizer/bpe_wiki_vocab.json

   Print after training:
     Vocab size achieved
     Sample tokenizations:
       "the president announced"
       "scientists discovered"
       "the economy grew"
       "photosynthesis"  (should still be 1-2 tokens)
     Average tokens per sentence (target: 10-20)
     Round-trip success rate on 1000 val sentences

After completion print DONE block with:
  - Train sentence count
  - Train pair count
  - Val sentence count
  - Sample sentence pair (S_n, S_n+1)
  - Tokenizer vocab size
  - Average tokens per sentence
  - Round-trip success rate

Do not proceed to TASK 19R until confirmed.

---

## TASK 19R — Rebuild Architecture For WikiText Scale
**Status: WAITING**

### Instructions

WikiText-103 is 100x more data than ARC.
Architecture needs to scale accordingly.

Update environment/config.py MODEL section:

SMALL config (fast iteration, verify pipeline):
  MODEL_SMALL = dict(
      n_layers   = 6,
      n_heads    = 8,
      hidden_dim = 256,
      ffn_dim    = 1024,
      vocab_size = 16000,
      max_seq_len = 64,
      dropout    = 0.1,
  )

MEDIUM config (actual experiment):
  MODEL_MEDIUM = dict(
      n_layers   = 12,
      n_heads    = 8,
      hidden_dim = 512,
      ffn_dim    = 2048,
      vocab_size = 16000,
      max_seq_len = 64,
      dropout    = 0.1,
  )

Start with MODEL_SMALL to verify the pipeline.
Switch to MODEL_MEDIUM for the real experiment.

Update environment/models/transformer.py:
  Make hidden_dim, n_layers, n_heads, ffn_dim,
  vocab_size, max_seq_len all configurable
  (pass config dict to constructor)
  Currently they read from config.MODEL —
  make them accept any config dict

Update environment/models/gpt2_style.py:
  Accept config dict
  Same architecture, just configurable dims

Update environment/models/gmn_style.py:
  Accept config dict
  Output projection: Linear(hidden_dim, 256)
  → always projects to 256-dim regardless of hidden_dim
  This keeps the latent space fixed at 256-dim
  for compatibility with the existing memory store

Print parameter counts for both configs:

  Config    | GPT-2 params | GMN params
  SMALL     | XX.XM        | XX.XM
  MEDIUM    | XX.XM        | XX.XM

Target for MEDIUM: ~22M params each (matching GPT-2 small)

After completion print DONE block with:
  - Parameter counts for SMALL and MEDIUM configs
  - Whether MEDIUM is close to 22M
  - Forward pass shapes verified for both configs

Do not proceed to TASK 20R until confirmed.

---

## TASK 20R — Train GPT-2 Style on WikiText-103
**Status: WAITING**

### Instructions

Write environment/training/train_gpt2_wiki.py

Training objective: next-token prediction on WikiText-103.

Data:
  Load train_sentences.json
  For each sentence build:
    [BOS] sentence tokens [EOS]
  Concatenate ALL sentences into one long token stream
  (standard LM training — no sentence boundaries needed)
  Chunk into seq_len=64 blocks
  Each block is one training example

Training loop:
  Start with MODEL_SMALL to verify (2 epochs)
  Then switch to MODEL_MEDIUM for full training (10 epochs)

  Optimizer: AdamW lr=3e-4, weight_decay=0.1
  LR schedule: linear warmup 1000 steps, cosine decay
  Gradient clip: 1.0
  Batch: 64

  Loss: cross entropy on next token
        ignore [PAD] tokens

  Log every 200 steps:
    step | loss | perplexity | lr

  Evaluate every epoch:
    Perplexity on val_sentences.json
    (standard LM eval — lower is better)

  Save best model by val perplexity
  → environment/checkpoints/gpt2_wiki.pt
  → environment/results/gpt2_wiki_history.json

Target perplexity after 10 epochs on MEDIUM:
  < 100 (proving the model learned something)
  < 50 (proving it learned well)
  GPT-2 small from scratch on WikiText-103: ~37

After completion print DONE block with:
  - Per-epoch: train_loss | train_ppl | val_ppl
  - Best val perplexity and epoch
  - Total training time
  - Whether val_ppl < 100

Do not proceed to TASK 21R until confirmed.

---

## TASK 21R — Train GMN Style on WikiText-103
**Status: WAITING**

### Instructions

Write environment/training/train_gmn_wiki.py

Training objective: next-sentence contrastive prediction.

Data:
  Load train_pairs.json
  Each pair: [S_n, S_n+1] from same document

  For each pair:
    anchor   = encode(S_n)    → (256,) unit vector
    positive = encode(S_n+1)  → (256,) unit vector

  Negatives: IN-BATCH
    In a batch of 64 pairs:
      anchor_i's positive = pair_i's S_n+1
      anchor_i's negatives = all other S_n+1 in the batch (63 negatives)
    This gives 64-way softmax per example
    Much harder than random negatives

  Loss: InfoNCE (NT-Xent)
    For each anchor i:
      scores_i = [cos(anchor_i, pos_j) for j in batch] / T
      loss_i = -log(exp(scores_i[i]) / sum(exp(scores_i)))
    total_loss = mean(loss_i)
    Temperature T = 0.07

  This is exactly SimCSE / CLIP applied to sentence pairs.

Training loop:
  Start with MODEL_SMALL to verify (2 epochs)
  Then MODEL_MEDIUM for full training (10 epochs)

  Optimizer: AdamW lr=3e-4, weight_decay=0.1
  LR schedule: same as GPT-2
  Batch: 64 PAIRS (128 sentences total)
  Gradient clip: 1.0

  Log every 200 steps:
    step | contrastive_loss | alignment | uniformity

  Where:
    alignment   = mean cos(anchor, positive)
                  (higher = positives are closer)
    uniformity  = log(mean exp(-2 * pairwise_distances))
                  (lower = embeddings more spread out)
    These two metrics diagnose representation collapse.

  Evaluate every epoch on val_pairs.json:
    For each val pair (S_n, S_n+1):
      Find the correct S_n+1 among 99 random distractors
      (1 positive + 99 random negatives from val set)
      Recall@1 = fraction where correct S_n+1 is top-1
    Report Recall@1 (target: > 20% after training, >> 1% random)

  Save best model by Recall@1
  → environment/checkpoints/gmn_wiki.pt
  → environment/results/gmn_wiki_history.json

  COLLAPSE DETECTION:
    If alignment > 0.99 AND uniformity > -1:
      Representations have collapsed to a single point
      Stop training, reduce T or increase batch size

After completion print DONE block with:
  - Per-epoch: loss | alignment | uniformity | Recall@1
  - Best Recall@1 and epoch
  - Whether collapse was detected
  - Total training time
  - Whether Recall@1 > 20%

Do not proceed to TASK 22R until confirmed.

---

## TASK 22R — Geometry Verification
**Status: WAITING**

### Instructions

Before evaluating on HellaSwag, verify both models
learned something meaningful.

Write environment/eval/verify_geometry.py:

1. GPT-2 style verification:
   Sample 100 sentence pairs from val set
   For each (S_n, S_n+1):
     Score S_n+1 by log prob given S_n
     Score 9 random sentences from val set
     Rank of correct S_n+1 among 10 (1=best)
   Report: mean rank of correct continuation
   Target: mean rank < 3 (better than random = 5.5)

2. GMN style verification:
   Same 100 sentence pairs
   For each (S_n, S_n+1):
     cos(encode(S_n), encode(S_n+1)) vs random
     Rank of correct S_n+1 among 10
   Report: mean rank of correct continuation
   Target: mean rank < 3

3. UMAP visualization:
   Encode 1000 random val sentences with GMN
   UMAP to 2D
   Color by document (sentences from same doc
   should cluster together if geometry is good)
   Save to environment/results/wiki_geometry.png

4. Semantic directions (the key test):
   Sample 100 sentence pairs where S_n contains
   a specific topic word ("president", "science",
   "war", "economy")
   
   For each topic:
     direction = mean(encode(S_n+1) - encode(S_n))
                 across all pairs with that topic
   
   Check: are topic directions consistent?
     cos(direction_A, direction_B) for same topic
     should be > cos(direction_A, direction_B) for diff topics
   
   Print:
     Topic direction consistency (same topic cosine):
       president: XX.XX
       science:   XX.XX
       war:       XX.XX
       economy:   XX.XX
     Cross-topic cosine (should be lower): XX.XX

   This directly tests whether the geometry encodes
   meaningful semantic directions — the core claim.

Save to environment/results/geometry_verification.json

After completion print DONE block with:
  - GPT-2 mean rank of correct continuation
  - GMN mean rank of correct continuation
  - Topic direction consistency scores
  - Whether directions are meaningful (same > cross)
  - Path to UMAP plot

Do not proceed to TASK 23R until confirmed.

---

## TASK 23R — Evaluate on HellaSwag
**Status: WAITING**

### Instructions

This is the main result. Both models evaluated on
HellaSwag — the same benchmark from Paper 1.

Write environment/eval/eval_hellaswag.py:

Load the existing HellaSwag val set:
  data/val_full.json (10,042 examples, already exists)

For GPT-2 style:
  For each example score each of 4 endings:
    full_text = ctx + " " + ending_k
    tokenize with NEW wiki tokenizer
    log_prob_k = sum log probs of ending tokens
                 given ctx tokens
  predict = argmax(log_prob_0..3)
  Report accuracy on all 10,042 examples

For GMN style:
  For each example:
    ctx_vec     = encode(ctx)           # (256,)
    ending_vecs = encode(each ending)   # (4, 256)
    scores      = ending_vecs @ ctx_vec # (4,)
  predict = argmax(scores)
  Report accuracy on all 10,042 examples

Also evaluate:
  Random baseline: 25.0% (theoretical)
  Paper 1 GMN (original, with pretrained encoder): 31.3%
  Paper 1 GPT-2 small (pretrained, 124M): 28.6%

Print the definitive comparison table:

  Model                        | Params | Pretrain data    | HellaSwag
  ----------------------------|--------|------------------|----------
  Random baseline             | --     | --               | 25.0%
  GPT-2 small (pretrained)    | 124M   | 40GB web text    | 28.6%*
  GMN Paper 1 (frozen enc)    | 0.62M  | borrowed encoder | 31.3%*
  GPT-2 style (ours, scratch) | XX.XM  | WikiText-103     | XX.X%
  GMN style (ours, scratch)   | XX.XM  | WikiText-103     | XX.X%
  (* from Paper 1 experiments)

McNemar test:
  GMN scratch vs GPT-2 scratch: p = XX.XXXX
  GMN scratch vs random:        p = XX.XXXX

Save to environment/results/hellaswag_results.json

After completion print DONE block with:
  - Both from-scratch accuracies on 10,042 examples
  - McNemar p-value GMN vs GPT-2
  - Whether GMN scratch beats GPT-2 scratch
  - Whether GMN scratch beats Paper 1 GMN
  - The headline result

Do not proceed to TASK 24R until confirmed.

---

## TASK 24R — Final Paper Tables
**Status: WAITING**

### Instructions

Generate the complete paper for V3.

environment/results/paper_v3.tex:

TABLE 1 — Main result (HellaSwag):
  Model                    | Params | Training data    | HellaSwag
  Random                   | --     | --               | 25.0%
  GPT-2 small (pretrained) | 124M   | 40GB web         | 28.6%
  GMN (Paper 1)            | 0.6M   | borrowed encoder | 31.3%
  GPT-2 (ours, scratch)    | XX.XM  | WikiText-103     | XX.X%
  GMN (ours, scratch)      | XX.XM  | WikiText-103     | XX.X%

TABLE 2 — What each model learned (geometry verification):
  Model           | Continuation rank | Direction consistency
  GPT-2 scratch   | XX.X / 10         | N/A (generative)
  GMN scratch     | XX.X / 10         | XX.XX (same topic)
                  |                   | XX.XX (cross topic)

environment/results/v3_findings.md:

## Finding 1: Training Objective At Equal Scale
[fill from results]
Does geometric next-sentence prediction beat
autoregressive next-token prediction at equal params?

## Finding 2: Geometry Quality
[fill from results]
Does next-sentence training produce
meaningful semantic directions?

## Finding 3: Transfer to HellaSwag
[fill from results]
Does WikiText pretraining transfer to
HellaSwag commonsense reasoning?

## Finding 4: Scaling Comparison
[fill from results]
GMN scratch (XX.XM, WikiText-103) vs
GPT-2 small pretrained (124M, 40GB web text)

---

## HOW TO USE WITH CLINE

Paste at start of each session:

"Read README.md, TASKS.md, TASKS_V2.md,
TASKS_V3.md and TASKS_V3_REVISED.md.

TASKS 1-17 complete (Paper 1, HellaSwag).
TASKS V3 original (ARC) abandoned —
ARC is too small for a pretraining comparison.

V3 REVISED is the new experiment:
  Pretrain both architectures from scratch
  on WikiText-103 (103M tokens).
  Same data. Same params. Different objective.
  Evaluate on HellaSwag.

Execute TASK [N]R from TASKS_V3_REVISED.md only.
Print DONE block. Wait for confirmation."

Replace [N] with current task number (start at 18R).

---

## THE CORE SCIENTIFIC QUESTION

```
If you train a 22M parameter model on 103M tokens
of Wikipedia text using next-sentence contrastive
prediction instead of next-token prediction:

Does the resulting geometry encode meaning
well enough to beat autoregressive models
on downstream reasoning tasks?

If YES: geometric pretraining is a viable
        alternative to autoregressive pretraining.
        
If NO:  next-token prediction captures something
        that next-sentence prediction misses.
        The negative result tells us what that is.

Either way: a clean scientific contribution.
```