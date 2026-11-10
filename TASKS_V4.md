# TASKS V4 — Dynamic Latent Memory: Prediction in Latent Space

## The Core Idea

Standard LLMs predict the next token.
You predict the next LATENT POSITION.
Memory grows with every prediction.
Better memory → better predictions.
The model learns from experience, not just training.

## The Fundamental Claim To Prove

> "Prediction accuracy in latent space improves as
>  dynamic outcome-weighted memory grows.
>  The model gets smarter with every inference."

If true: this is a new learning paradigm.
Not pretraining then freezing.
Continuous learning through geometric navigation.

## What We're Building

```
Input sentence
  ↓ [encoder — already built, VAE head]
Current latent position P_current (256-dim)
  ↓ [query dynamic memory]
Context vector C (256-dim, from neighbors)
  ↓ [enrich position]
P_enriched = combine(P_current, C)
  ↓ [latent predictor — NEW]
P_predicted (256-dim, predicted next position)
  ↓ [compare to actual]
P_actual = encode(next sentence)
  ↓ [memory update — NEW]
Add (P_enriched, P_actual, weight) to memory
Update edge weights by prediction accuracy
  ↓ [decoder — only when needed]
P_final → tokens (used sparingly)
```

Execute one task at a time. Print DONE block. Wait for confirmation.

---

## TASK 27 — Latent Predictor

**Status: CURRENT**

The latent predictor is a small MLP that takes your
current enriched position and predicts where you go next
in latent space.

This is the core of the new architecture.

### Instructions

Write src/predictor/latent_predictor.py:

1. Class LatentPredictor(nn.Module):

   Architecture:
     Input:  512-dim (concat of P_current + context C)
     Hidden: Linear(512, 512) → LayerNorm → GELU
     Hidden: Linear(512, 256) → LayerNorm → GELU
     Output: Linear(256, 256)
     L2 normalize output to unit sphere

   Forward(current_vector, context_vector):
     x = concat([current_vector, context_vector])  # (512,)
     x = through MLP layers
     x = L2_normalize(x)
     return predicted_next_position (256-dim unit vector)

   Loss function — cosine prediction loss:
     loss = 1 - cos(predicted, actual_next)
     (0 = perfect prediction, 2 = worst case)
     
     Also compute: top-1 recall
       Is actual_next the closest vector in memory
       to predicted_next?
       (measures if prediction is directionally correct)

   count_parameters() -> int

2. Write scripts/train_latent_predictor.py:

   Data: WikiText sentence pairs (val_pairs.json)
         Use val set for speed — 7,666 pairs
         Train set (3.57M pairs) for full training later
   
   Use ALREADY TRAINED VAE (checkpoints/vae.pt):
     Encode all sentences with vae.encode_texts()
     L2 normalize → latent vectors
     Store as numpy array (fast lookup)
     Do NOT retrain the VAE
   
   For each sentence pair (S_n, S_n+1):
     P_current = encode(S_n)               # (256,)
     P_actual  = encode(S_n+1)             # (256,)
     
     Context C:
       Query memory store for P_current
       If memory has < 5 entries: C = zeros(256)
       Else: C = fuse_directions(top-3 neighbors)
             (reuse existing fusion code)
     
     P_predicted = predictor(P_current, C)
     loss = 1 - cos(P_predicted, P_actual)
     
     Memory update AFTER prediction:
       prediction_quality = cos(P_predicted, P_actual)
       outcome_weight = max(0.1, prediction_quality)
       add (P_current, P_actual, outcome_weight) to memory
       (memory grows with each example)
   
   Train for 5 passes over val_pairs.json
   (simulates 5 "experiences" of the same data)
   
   After each pass report:
     mean prediction loss (cosine distance)
     mean cos(predicted, actual)
     top-1 recall in memory
     memory size
   
   KEY METRIC: does prediction loss DECREASE
   over the 5 passes as memory grows?
   This is the proof of concept.
   
   Save predictor to checkpoints/latent_predictor.pt
   Save history to experiments/predictor_history.json

3. Baseline (no memory):
   Run the same evaluation with C = zeros(256) always
   (predictor sees current position only, no memory)
   Report prediction loss for this baseline
   
   If with-memory loss < no-memory loss:
     Memory is helping prediction. Core claim validated.

After completion print DONE block with:
  - Predictor parameter count
  - Per-pass prediction loss (5 passes)
  - Baseline (no memory) prediction loss
  - Whether loss decreased over passes
  - Whether memory helps vs no-memory baseline
  - Final memory size

Do not proceed to TASK 28 until confirmed.

---

## TASK 28 — Dynamic Memory Update Rules

**Status: WAITING**

The memory needs smart update rules.
Not everything gets added equally.
Good predictions reinforce good directions.
Bad predictions get lower weight.

### Instructions

Write src/memory/dynamic_store.py:

Class DynamicMemoryStore:

  __init__(max_size=100000, latent_dim=256):
    self.vectors     = []   # list of (256,) arrays
    self.outcomes    = []   # list of floats (outcome weights)
    self.timestamps  = []   # list of ints (when added)
    self.hit_counts  = []   # list of ints (how often retrieved)
    self.faiss_index = None # rebuilt periodically
    self.step        = 0
    self.max_size    = max_size

  add(position_vector, outcome_weight):
    Append to lists
    If len > max_size:
      FORGET: remove lowest outcome_weight * hit_count entries
      (keep what was useful AND accurate)
    Rebuild FAISS index every 1000 additions
    self.step += 1

  query(vector, k=10):
    FAISS kNN search
    Return top-k with:
      vector, outcome_weight, hit_count, timestamp
    Update hit_counts for retrieved entries

  update_outcome(idx, new_outcome):
    When actual outcome arrives (e.g. prediction was right)
    Update outcome_weight for entry at idx
    outcome_weight = 0.9 * old + 0.1 * new  (EMA update)

  forget_policy():
    Called when max_size exceeded
    Score each entry: score = outcome_weight * log(1 + hit_count)
    Remove bottom 10% by score
    This keeps:
      - High outcome weight (worked well)
      - High hit count (frequently useful)
    And forgets:
      - Low weight AND rarely retrieved

  save(path) / load(path)

  properties:
    size -> int
    mean_outcome_weight -> float
    most_useful_k(k=5) -> list of top entries by score

Write scripts/test_dynamic_memory.py:

  Simulate 1000 add operations
  Verify forget policy fires at max_size
  Verify hit counts update on query
  Verify outcome update works
  Print memory statistics over time:
    size | mean_outcome | top_hit_count | forgotten_count
  
  Key test: add 500 good entries (weight=0.9)
            add 500 bad entries (weight=0.1)
            fill to max_size
            verify good entries survive forget

After completion print DONE block with:
  - Memory size at various points
  - Forget policy working (confirmed True/False)
  - Good entries survive over bad entries (True/False)
  - Query time for 10 neighbors at 10K memory size

Do not proceed to TASK 29 until confirmed.

---

## TASK 29 — The Prediction Loop

**Status: WAITING**

Now connect everything into a continuous
prediction loop over a real text corpus.

The loop:
  Encode sentence → predict next → update memory
  → encode next sentence → predict next → update memory
  → repeat

### Instructions

Write scripts/run_prediction_loop.py:

  Setup:
    Load VAE from checkpoints/vae.pt (frozen)
    Load LatentPredictor from checkpoints/latent_predictor.pt
    Initialize DynamicMemoryStore (empty, max_size=50000)
    Load first 10000 pairs from train_pairs.json
    (stream line by line — do NOT load full file)

  For each pair (S_n, S_n+1) in order:

    Step 1: Encode
      P_current = vae_encode(S_n)    # unit vector
      P_actual  = vae_encode(S_n+1)  # unit vector

    Step 2: Query memory
      if memory.size >= 5:
        neighbors = memory.query(P_current, k=10)
        C = fuse_directions(neighbors, top_k=3,
                            use_tangent=True,
                            query_vector=P_current)
      else:
        C = zeros(256)

    Step 3: Predict
      P_predicted = predictor(P_current, C)
      prediction_quality = cos(P_predicted, P_actual)

    Step 4: Update memory
      memory.add(P_current, outcome_weight=prediction_quality)

    Step 5: Log every 500 steps:
      step | pred_quality | memory_size | mean_memory_weight

  After all 10000 pairs:
    Report prediction quality over time:
      First 1000 pairs:  mean prediction quality
      Middle 1000 pairs: mean prediction quality
      Last 1000 pairs:   mean prediction quality
    
    If last > first: memory is helping over time.
    This is the CORE PROOF OF CONCEPT.

  Also run ABLATION:
    Same loop but memory.query() always returns zeros
    (predictor sees no memory context)
    Compare prediction quality at each checkpoint

  Print final comparison:
    Window        | With memory | No memory | Delta
    Steps 0-1000  | XX.XXXX     | XX.XXXX   | XX.XXXX
    Steps 1000-5000 | XX.XXXX   | XX.XXXX   | XX.XXXX
    Steps 5000-10000| XX.XXXX   | XX.XXXX   | XX.XXXX

  Save to experiments/prediction_loop_results.json

After completion print DONE block with:
  - Prediction quality at start vs end
  - Memory vs no-memory comparison table
  - Whether prediction improved over time
  - Whether memory helped significantly
  - Final memory size and mean outcome weight

Do not proceed to TASK 30 until confirmed.

---

## TASK 30 — Latent Space Generation

**Status: WAITING**

Now add the decoder.
Prediction happens in latent space.
Text output is generated ONCE from the final position.
Not token by token — position by position then decode.

### Instructions

The generation process:

  Given: context text (the prompt)

  Step 1: Encode all context sentences
    positions = [vae_encode(s) for s in context_sentences]
    P_current = positions[-1]  # last position

  Step 2: Navigate forward N steps in latent space
    for step in range(N_steps):
      C = memory.query_and_fuse(P_current)
      P_next = predictor(P_current, C)
      memory.add(P_current, outcome_weight=0.5)  # neutral
      P_current = P_next

  Step 3: Decode final position to text
    The final P_current is where you predict
    the conversation should go.
    
    Decode by RETRIEVAL not generation:
      Find top-5 neighbors in memory
      Return the actual text of those neighbors
      Weighted by outcome_weight and similarity
    
    This is the output: retrieved text from similar
    past experiences at the predicted position.
    NOT generated token by token.

Write scripts/demo_generation.py:

  Load: VAE, predictor, memory (from prediction loop)
  
  Demo 1: Given first sentence of a WikiText paragraph
    Predict 3 steps forward
    Retrieve text at final position
    Print: original continuation vs retrieved continuation
    Measure: cos(predicted_final, actual_continuation)
  
  Demo 2: Run 20 examples
    For each measure cos(predicted, actual)
    Compare to baseline (predict P_current as next,
    i.e. "stay in place" baseline)
    
  Print:
    Example | Predictor cos | Stay-in-place cos | Memory size
    ...20 examples...
    Mean predictor cos:       XX.XXXX
    Mean stay-in-place cos:   XX.XXXX
    Delta:                    XX.XXXX

  Save to experiments/generation_demo.json

After completion print DONE block with:
  - Mean prediction cosine similarity
  - Stay-in-place baseline cosine
  - Whether predictor beats baseline
  - Sample generated output (retrieved text)

Do not proceed to TASK 31 until confirmed.

---

## TASK 31 — Memory Growth Experiment

**Status: WAITING**

The key scientific experiment.
Does prediction quality improve as memory grows?
Test at different memory sizes.

### Instructions

Write scripts/memory_growth_experiment.py:

  Use first 50,000 pairs from train_pairs.json (streaming)
  Split into chunks of 1000 pairs each

  For each chunk:
    Run prediction loop on that chunk
    After chunk: evaluate on FIXED 500 val pairs
    Record: prediction quality, memory size

  This gives you a curve:
    memory_size → prediction_quality
    over 50 evaluation points

  Plot the curve:
    X axis: memory size (0 to 50000)
    Y axis: mean prediction quality (cosine similarity)
    
    Save to notebooks/memory_growth_curve.png

  Key measurements:
    At memory size 0:      XX.XXXX (baseline)
    At memory size 1000:   XX.XXXX
    At memory size 5000:   XX.XXXX
    At memory size 10000:  XX.XXXX
    At memory size 50000:  XX.XXXX

  Fit a curve to the data:
    Does it follow a log curve? (log(memory_size))
    Or power law? Or plateau?
    Report best fit.

  Save to experiments/memory_growth.json

After completion print DONE block with:
  - Prediction quality at key memory sizes
  - Improvement from 0 to 50000 memories
  - Curve shape (log/power/plateau)
  - Whether growth is statistically significant

Do not proceed to TASK 32 until confirmed.

---

## TASK 32 — The Paper: Dynamic Latent Memory

**Status: WAITING**

### Instructions

Write paper/v4_paper.md:

# Dynamic Latent Memory Navigation:
# Continuous Learning Through Geometric Prediction

## Abstract
We propose an architecture where prediction happens
in latent space rather than token space. A small
encoder maps text to positions on a semantic manifold.
A latent predictor navigates forward on this manifold,
guided by a dynamic outcome-weighted memory that
grows with every inference. We show that prediction
accuracy improves monotonically as memory grows,
demonstrating that the architecture learns from
experience without retraining. On WikiText-103
sentence continuation, our XX-parameter model
achieves mean cosine similarity of XX vs XX for
a memoryless baseline.

## 1. Introduction
[The bridge metaphor. Structure is learning.
Dropped into geometric space. Navigate by experience.]

## 2. Architecture
[Encoder → Dynamic Memory → Enriched Position
→ Latent Predictor → Memory Update → Decoder]

## 3. Dynamic Memory
[Growth, forget policy, outcome weighting,
comparison to Engram (token identity) and
kNN-LM (static retrieval)]

## 4. Latent Prediction
[Why predict in latent space not token space.
Cosine prediction loss. Continuous vs discrete.]

## 5. Experiments
[Memory growth curve. Ablations.
With vs without memory. Static vs dynamic memory.]

## 6. Related Work
[kNN-LM, RETRO, Engram, DeepSeek CED,
Predictive Coding, World Models]

## 7. Conclusion
[The living geometry. Intelligence as navigation.
Future: scale memory not parameters.]

Generate the LaTeX tables from experiments/
Generate the key figures list

Save paper/v4_paper.md

---

## THE CORE SCIENTIFIC STORY

```
Question:
  Can a model get smarter just by experiencing more,
  without any retraining or parameter updates?

Method:
  Encode experiences as positions in latent space
  Store positions with prediction accuracy as weight
  Predict next position guided by past experiences
  Update memory with every new experience

Answer (to be shown by experiments):
  Yes — prediction quality improves as memory grows
  The improvement follows a log curve
  (each doubling of memory gives diminishing but real gains)
  Memory-guided prediction beats memoryless prediction

Implication:
  Intelligence can scale with EXPERIENCE not PARAMETERS
  The geometry is learned once (encoder training)
  Navigation improves continuously (memory growth)
  No retraining required for new knowledge
  New knowledge = new memory entries
```

---

## HOW TO USE WITH CLINE

Paste at start of each session:

"Read README.md, TASKS.md, TASKS_V2.md,
TASKS_V3_REVISED.md and TASKS_V4.md.

TASKS 1-17: Paper 1 (HellaSwag, complete)
TASKS V3: WikiText pretraining experiments
TASKS V4: Dynamic latent memory (current)

Existing assets available:
  checkpoints/vae.pt              (trained VAE encoder)
  checkpoints/gmn_wiki.pt         (partial)
  src/memory/store.py             (static FAISS store)
  src/sampler/jev.py              (parallel sampler)
  src/fusion/fuse.py              (tangent fusion)
  src/eval/evaluate.py            (geometric scorer)
  environment/data/wikitext/      (WikiText-103 data)

The VAE encoder (checkpoints/vae.pt) is the
geometric foundation. Do not retrain it.
Build the latent predictor ON TOP of it.

Execute TASK [N] from TASKS_V4.md only.
Print DONE block. Wait for confirmation."

Replace [N] with current task number (start at 27).