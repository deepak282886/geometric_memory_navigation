# V3 — From Scratch: GPT-2 Style vs GMN Style

A clean, self-contained controlled experiment. Two architectures are trained
**from scratch** on **identical data** with **identical parameter budgets**:

| | GPT-2 style | GMN style |
|---|---|---|
| Attention | causal (decoder) | bidirectional (encoder) |
| Objective | autoregressive next-token prediction | geometric contrastive |
| Layers / heads / width | 6 / 8 / 256 | 6 / 8 / 256 |
| FFN dim / context / vocab | 1024 / 128 / 8000 | 1024 / 128 / 8000 |
| Parameters | ~22 M | ~22 M |

**Architecture and training objective are the only variables.** Everything else
— tokenizer, data, schedule, seed, optimiser, batch size, gradient clipping — is
byte-identical because both models read the frozen dictionaries in
`environment/config.py`.

> **TASK 19R (WikiText-103) supersedes the sizes in the table above.** The
> ARC-era contract it describes (6 / 8 / 256, FFN 1024, context 128, vocab 8000)
> is kept in this document as the record of TASKs 18-26; the live configs are now
> `config.MODEL_SMALL` (6 × 256, ffn 1024, seq 64, vocab 16000 — the pipeline
> check) and `config.MODEL_MEDIUM` (12 × 512, ffn 2048, seq 64, vocab 16000 — the
> real experiment), and both models take a **config dict** instead of reading
> `config.MODEL`. Measured: SMALL 8,851,456 (GPT-2) / 8,917,248 (GMN),
> MEDIUM 46,054,400 / 46,185,728 — see the Models section below.

## The claim under test

> Geometric contrastive training outperforms autoregressive next-token
> prediction on reasoning tasks at equal parameter count.

Unlike the earlier GMN work (`README.md`, TASKS.md, TASKS_V2.md), nothing here
is borrowed: no frozen `all-MiniLM-L6-v2`, no pretrained tokenizer, no
pretrained weights.

## Layout

```
environment/
├── tokenizer/          bpe.py — BPE from scratch                    (TASK 19)
│                       train_tokenizer.py, bpe_vocab.json
├── models/             transformer.py — SHARED blocks + heads   (TASK 20)
│                       gpt2_style.py / gmn_style.py / test_models.py
├── training/           train_gpt2.py / train_gmn.py                (TASKS 22-23)
├── data/               loader.py (TASK 18)  arc/  conceptnet/
├── eval/               eval_gpt2.py / eval_gmn.py                  (TASK 24)
├── compare.py          head-to-head harness                        (TASK 24)
├── config.py           ALL hyperparameters, one place
└── README_V3.md
```

## Data

### ARC-Challenge — the reasoning task

`environment/data/loader.py` pulls `ARC-Challenge` from the Hub repo
`allenai/ai2_arc` and writes the fixed 4-way schema described in TASKS_V3.md:

```json
{"question": "George wants to warm his hands quickly by rubbing them. ...",
 "choices":  ["dry palms", "wet palms", "palms covered with oil", "palms covered with lotion"],
 "label":    0}
```

Two details worth knowing:

* **Answer keys come in two encodings.** Most rows use `A`-`D`; the legacy
  NYSED rows use `1`-`4`. The answer is located by its position in
  `choices["label"]`, so both resolve to a 0-based index without a special case.
* **A few questions are not 4-way** (ARC has 3-choice and 5-choice rows). They
  are dropped, because this experiment's record schema — and therefore the
  scoring procedure both models share — is a fixed 4-way choice. The loader
  prints exactly how many rows each reason removed, per split.

The label balance is *not* exactly 25%: ARC-Challenge's answer key distribution
is genuinely skewed (roughly 21 / 26 / 26 / 26). Because both models get the
identical file, this is a property of the task, not a confound.

### ConceptNet — the relational signal for GMN

`conceptnet-assertions-5.7.0.csv.gz` (the build TASKS_V3.md names, ~475 MB) is
streamed to a local cache and parsed line by line. Kept triples must:

1. have a relation in `Causes`, `IsA`, `HasProperty`, `PartOf`, `UsedFor`,
   `CapableOf`, `CausesDesire`, `HasPrerequisite`;
2. have both endpoints be English (`/c/en/...`) and reduce to a clean
   natural-language term (letters/digits plus `'`, `-`, `.` — so `/c/en/fire/n/1`
   becomes `fire`, and URI-ish garbage is excluded).

Relation URIs are expanded at camel-case boundaries and lower-cased, which
turns them into readable phrases: `/r/IsA` → `is a`, `/r/HasProperty` →
`has property`, `/r/UsedFor` → `used for`.

Accepted triples are **reservoir sampled** (Algorithm R, seeded from
`TRAIN["seed"]`) down to 100,000. ConceptNet's file order is grouped by
dataset/relation, so taking the first N would bake that skew into the training
signal; the reservoir gives a uniform, deterministic, O(target)-memory sample.
The final file is de-duplicated and sorted by `(subject, relation, object)` so
it is stable and diffable.

TASK 19 turns each triple into the sentence `"subject relation object"`
(e.g. `"fire causes heat"`) for tokenizer training; TASK 23 uses the triples
themselves as the relational (direction-consistency) signal.

## Running TASK 18

From the repository root (the `PATHS` entries are relative to it; scripts
resolve them through `config.path(...)`, so the cwd does not actually matter):

```bash
python environment/data/loader.py                       # ARC + ConceptNet
python environment/data/loader.py --arc-only            # ARC only (seconds)
python environment/data/loader.py --conceptnet-only --conceptnet-target 20000
python environment/data/loader.py --force-download      # ignore the raw cache
python environment/data/loader.py --conceptnet-csv path/to/assertions.csv.gz
```

The 475 MB download is resumable (HTTP `Range`), so an interrupted run
continues instead of starting over.

Artifacts land at:

```
environment/data/arc/train.json           ~1117 records
environment/data/arc/val.json              ~295 records
environment/data/arc/test.json            ~1165 records
environment/data/conceptnet/triples.json  100000 triples
environment/data/conceptnet/conceptnet-assertions-5.7.0.csv.gz   (raw cache)
```

`environment/data/arc/`, `environment/data/conceptnet/`, `checkpoints/` and
`results/` are git-ignored — they are regenerable from `loader.py`.

## Tokenizer (TASK 19)

`environment/tokenizer/bpe.py` (the tokenizer) + `train_tokenizer.py` (the
training/report script). Learned from the TASK 18 corpus **only** — no
pretrained vocabulary and no borrowed merges.

**Corpus** — 112,072 strings: 2,577 ARC questions, 10,308 ARC choices, 99,187
ConceptNet sentences (`"subject relation object"`, e.g. `"fire causes heat"`).

**Preprocessing**, applied identically at train, encode and round-trip time:
lowercase → a space around every punctuation character → collapse whitespace →
strip. Nothing is deleted, so `"The trophy didn't fit"` → `"the trophy didn ' t fit"`.

**Word boundaries** — every word is prefixed with `Ġ` (GPT-2's convention), so a
merge can never fuse the end of one word with the start of the next, and
`decode` restores the spaces by replacing `Ġ`.

**Vocabulary layout** — 8,000 ids, target met exactly:

| ids | contents |
|---|---|
| 0-4 | `[PAD] [UNK] [BOS] [EOS] [MASK]` — reserved, always first |
| 5-74 | the 70 single characters that occur in the corpus |
| 75-7999 | 7,925 merged tokens, in merge order |

Merges therefore start at id 75 and never occupy the reserved ids 0-4.

**Results**

| Metric | Value |
|---|---|
| Vocab size achieved | **8,000** (target 8,000, did not stop early) |
| Merge rules | 7,925 |
| Most frequent merge | `Ġ` + `a`, 98,883 occurrences |
| Lowest merge count | 18 — no merge below 10, so every merge is meaningful |
| Avg tokens / ARC question | 27.8 (median 23, p95 62, max 186) |
| Avg tokens / ARC choice | 6.4 (median 6, p95 13, max 48) |
| Round-trip, ARC val questions | **100%** (295/295, unpadded) |
| Round-trip, ARC val choices | **100%** (1,180/1,180) |
| `[UNK]` tokens produced | 0 |
| Training time | ~7 s |

Domain terms become single tokens: `photosynthesis`, `evaporation`,
`gravitational`, `ecosystem`, `electricity` and `causes` are each 1 token.

**Context budget** — `max_seq_len = 128` fits question + correct choice for
294/295 val records; only the single longest question (186 tokens) is truncated.
`encode()` returns exactly 128 ids (truncating/padding as TASK 19 specifies);
`encode_unpadded()` is the lossless path used by the round-trip test and by the
trainers, which need true sequence lengths.

Saved to `environment/tokenizer/bpe_vocab.json` (258 KB) as exactly
`{"vocab": {...}, "merges": [[a, b], ...]}`, reloadable via `BPETokenizer.load()`.

```bash
python environment/tokenizer/train_tokenizer.py                  # full 8,000
python environment/tokenizer/train_tokenizer.py --vocab-size 4000
python environment/tokenizer/train_tokenizer.py --round-trip 300 --quiet
```

## Models (TASK 20)

`environment/models/` — the shared trunk plus the two task-specific heads.

```
transformer.py      TransformerBase / TransformerBlock / MultiHeadAttention / FeedForward
                    + resolve_model_config (TASK 19R config-dict validation)
gpt2_style.py       GPT2Style  — causal trunk + tied LM head
gmn_style.py        GMNStyle   — bidirectional trunk + [CLS] projection head
                    + optional key_padding_mask on encode()/forward()
test_models.py      47-check verification script (exits non-zero on failure)
test_models_wiki.py 75 checks = the same 47 on MODEL_SMALL + the WikiText tokenizer
                    + 28 TASK 19R checks (padding mask, fixed 256-d head, config
                    plumbing, parameter table, real-sentence forwards, MEDIUM)
model_sizes.py      parameter table for every config, formula vs measured (TASK 19R)
```

Both models are built through
`TransformerBase.from_config(causal=..., config_dict=...)`, so they share one
code path and cannot drift apart. `config_dict=None` falls back to
`config.MODEL`; the WikiText trainers pass `config.MODEL_SMALL` /
`config.MODEL_MEDIUM` (or a name via `config.model_config("medium")`). `causal`
is the **only** architectural switch:

### TASK 19R sizes (measured, `python environment/models/model_sizes.py`)

| config | layers / heads / width | ffn | seq | vocab | GPT-2 params | GMN params |
|---|---|---|---|---|---|---|
| `MODEL_SMALL` | 6 / 8 / 256 | 1024 | 64 | 16000 | **8,851,456** | **8,917,248** |
| `MODEL_MEDIUM` | 9 / 8 / 384 | 1536 | 64 | 16000 | **22,139,520** | **22,238,080** |

MEDIUM's fields were corrected to 9 x 384 / ffn 1536 (DECISION 1). TASK 19R had
specified 12 x 512 / ffn 2048 for "~22M", which measures 46,054,400 — the target
is unreachable that way (the tied embedding is 16000 x 512 = 8.2M and the 12
blocks are 37.8M). 22,139,520 is +0.6% of 22M. Both numbers are asserted by
`test_models_wiki.py` (within 10% of target, formula == measured).

Training runs on **`MODEL_SMALL`** (TASK 20R/21R, DECISION 2): 8.85M is what a
CPU-only, 3.1 GB-RAM machine can actually train, and the claim under test is
about the training objective, not the parameter count.

The GMN head is always `Linear(hidden_dim, 256)`, so the latent space stays
256-d and `MODEL_SMALL` / `MODEL_MEDIUM` remain directly comparable.

| | shared | GPT-2 style | GMN style |
|---|---|---|---|
| Token embedding (8000×256) | ✓ | | |
| Learned positional embedding (128×256) | ✓ | | |
| 6 × TransformerBlock (pre-LN, 8 heads, FFN 1024) | ✓ | | |
| Final LayerNorm | ✓ | | |
| GPT-2 init + residual rescaling | ✓ | | |
| Attention mask | | strictly lower-triangular | dense / bidirectional |
| Head | | `Linear(256, 8000, bias=False)` tied to the embedding | `Linear(256, 256)` on the `[CLS]` position |
| Output | | logits `(B, T, 8000)` | scores `(B, 4)` |

### Parameter counts

| Model | Parameters |
|---|---|
| GPT-2 style | **6,819,840** |
| GMN style | **6,885,632** |
| difference | 65,792 (**0.96%**) |

The difference is the GMN projection head alone (`256×256 + 256 = 65,792`); the
tied LM head contributes nothing extra. **If the LM head were left untied**,
GPT-2 would be 8,867,840 parameters, i.e. 2,048,000 (= `8000 × 256`) larger than
GMN — a 29.7% gap. Tying is what keeps the budgets comparable, which is why
TASK 20 note 4 requires it. Verified by rebuilding with `tie_weights=False` and
diffing.

Per-component (identical for both models unless noted):

```
trunk.token_embedding       2,048,000
trunk.positional_embedding     32,768
trunk.blocks                4,738,560   (6 x 789,760)
trunk.ln_final                    512
projection (GMN only)          65,792
```

### Padding sensitivity — a correctness note for TASKS 22/24

The tokenizer pads every sequence out to `max_seq_len = 128`, but ARC questions
are only ~20-48 tokens, so most positions are `[PAD]`. With **causal** attention
that is harmless: position *t* never sees later positions, so right-padding is
invisible and a padding mask would be a no-op (measured: 2e-6).

With **bidirectional** attention the `[CLS]` token attends to every `[PAD]` too,
which measurably corrupts the sentence vector:

| variant | cosine vs the properly trimmed vector |
|---|---|
| 128-wide batch, no padding mask | **0.538 – 0.732** (worst 0.538) |
| 128-wide batch + `key_padding_mask` | **1.000000** (max abs diff 2e-6) |

So the shared blocks expose an optional `key_padding_mask` that reproduces
trimming exactly (verified), and it is **off by default for both models** — it is
an input mask, not the causal-vs-bidirectional architectural switch. TASK 22/24
must either mask padding for GMN, or collate sequences trimmed to the batch
maximum so that almost no padding exists. Either resolves it; leaving it
unaddressed would handicap the bidirectional model for a non-architectural
reason.

### Verification

`python environment/models/test_models.py` runs 47 checks, all passing:

* causal GPT-2 gives **exactly 0.0** weight to every position > 5, and perturbing
  token 6 leaves the logits at positions ≤ 5 bitwise unchanged;
* bidirectional GMN gives non-zero weight (0.052) to the same position, spreads
  weight over all 16 positions, and perturbing token 6 **does** move the `[CLS]`
  vector;
* the two trunks have identical state-dict keys *and* tensor shapes (100 tensors),
  use the same classes throughout, and **initialise to elementwise identical
  weights from the same seed** — proof that one code path builds both;
* GPT-2 logits `(2, 16, 8000)`, GMN `question_vec` `(2, 256)`, `choice_vecs`
  `(2, 4, 256)`, `scores` `(2, 4)` — all finite, unit-norm, batch-invariant;
* positional embeddings are `nn.Embedding(128, 256)`, trainable, receive
  non-zero gradient, and no sinusoidal buffer exists;
* the LM head weight **is** the token-embedding tensor, and `[PAD]` targets get
  exactly zero loss.

Also integration-tested against the real TASK 18/19 artifacts: real ARC val
records encoded with the trained BPE run through both models and take an
optimiser step with finite losses and non-zero gradients on every parameter.

## Status

| Task | What | Status |
|---|---|---|
| 18 | Environment setup & data | **done** |
| 19 | BPE tokenizer from scratch | **done** |
| 20 | Shared transformer blocks | **done** |
| 21 | Both architectures | waiting |
| 22-24 | Training + evaluation | waiting |
| 25-26 | Navigation + final table | waiting |

### WikiText-103 track (TASKS_V3_REVISED.md) — the live experiment

The ARC track above is abandoned (too small for a pretraining comparison). The
running experiment pretrains both architectures from scratch on WikiText-103 at
equal parameter count and evaluates on HellaSwag.

| Task | What | Status |
|---|---|---|
| 18R | Stop ARC, WikiText-103 sentences/pairs, 16k BPE tokenizer | **done** |
| 19R | WikiText-scale configs + config-dict API + model tests | **done** |
| 20R | Train GPT-2 style on WikiText-103 | waiting |
| 21R | Train GMN style on WikiText-103 | waiting |
| 22R-24R | Geometry verification, HellaSwag, final tables | waiting |

Data (TASK 18R): `environment/data/wikitext/{train,val}_{sentences,pairs}.json`
(3,837,479 / 3,570,028 train, 8,206 / 7,666 val; one element per line, stream it),
tokenizer `environment/tokenizer/bpe_wiki_vocab.json` (vocab 16,000).
