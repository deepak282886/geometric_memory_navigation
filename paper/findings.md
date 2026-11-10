# Key Findings

**Note on evaluation sets.** Findings 1-5 are measured on the honest
5,042-example split (the TASK 1-12 navigation study); findings 6-8 are measured on
the full 10,042-example validation set (the TASK 13-17 structure-induction study).
The original-VAE baseline is therefore 29.7% in findings 1-5 and 31.3% in findings
6-8. Numbers must not be compared across the two groups.

1. A 624K-parameter VAE geometric scorer significantly beats
   GPT-2 small (124M parameters) on
   HellaSwag: 29.7% vs 25.6%
   (McNemar p < 0.0001, 199x parameter
   reduction). On 5,042 held-out examples, context-cosine scoring alone beats
   GPT-2; adding the trained navigation vector does not.

2. Chord navigation significantly degrades accuracy
   (-3.6 points, p < 0.0001)
   because of a geometric artifact: the chord vector (memory - context) sits
   117 degrees from the context on the unit sphere
   (mean cos = -0.4486), so moving the scoring position along it moves it
   *away* from the discriminative signal.

3. Tangent projection (removing the radial component) recovers
   1.2 points
   (27.8% -> 29.0%) but navigation still adds
   no benefit: tangent remains 2.3 points
   below the context-only baseline (p < 0.0001).

4. Using correct-ending vectors as navigation targets produces the strongest
   navigation direction yet measured - the combined-score gap rises
   0.0031 (chord) -> 0.0078 (tangent) -> 0.0159
   (continuation), against 0.0195 for context-only scoring - but the
   navigation term's own discriminative gap is still 6.9x weaker than
   the context term's (0.0039 vs 0.0274).

5. Additive scoring (ctx + lambda * nav) with honest lambda tuning shows lambda is
   not measurable: the tune-set spread across lambda (0.90 points)
   is smaller than the tune/eval swing (1.27 points), so
   navigation's contribution is not reliably estimable. The peak lambda
   (0.50) ranks last among the swept values on the held-out set, and
   navigation vs no-navigation is not significant (p = 0.0616).

6. **End-to-end task training (CLIP-style).** Fine-tuning the VAE head directly on
   the 4-way scoring objective (frozen base encoder, `CrossEntropy(cos(ending, ctx)
   / T, label)`, T=0.10) improved accuracy from 31.3% to
   32.4% (p = 0.0034) - the first significant improvement
   from structure induction, and +1.1 points over
   the reconstruction-trained VAE. Critically, the CLIP-style models have by far
   the **smallest** correct-vs-wrong cosine gaps of any variant yet the **highest**
   accuracies:

   | Model | Val gap | Accuracy |
   |---|---|---|
   | CLIP-style (5 epochs) | 0.0071 | 32.14% |
   | CLIP-style (T=0.07, best observed) | 0.0075 | 32.56% |
   | CLIP-style (T=0.10, pre-registered) | 0.0108 | 32.43% |
   | Original VAE (recon + KL) | 0.0274 | 31.31% |
   | Margin ranking (v3, margin=0.3) | 0.0474 | 31.88% |
   | Margin ranking (v2, margin=0.5) | 0.0530 | 31.37% |
   The relationship is **inverted**: a margin-ranking loss maximised the gap
   (0.047) for no accuracy gain, while CLIP cuts the gap to 0.007
   (with per-example gap std 0.032 rather than 0.241) and still wins.
   Accuracy therefore comes from fine-grained **ordering** of the four candidates, not
   from wide separation - the gap-maximisation hypothesis is falsified.

7. **Overfitting in task fine-tuning.** Val accuracy peaks at epoch
   6 (41.6% on 500 validation examples) and then
   **declines** to 40.2% at epoch 15, while the training
   loss falls monotonically (1.39 -> 1.26) and training
   accuracy keeps rising. A falling training loss was therefore *not* evidence that
   more epochs would help: the optimum is early stopping around epoch 6,
   not maximum training.

8. **Temperature robustness.** CLIP-style training is robust across the useful
   temperature range; all of T=0.03, 0.07 and 0.10 significantly beat the original
   VAE (largest p = 0.0129):

   | Temperature | Accuracy (10,042) | vs original | p vs original |
   |---|---|---|---|
   | 0.03 | 32.23% | +0.93 | 0.0129 |
   | 0.07 | 32.56% | +1.25 | 0.0010 |
   | 0.10 | 32.43% | +1.13 | 0.0034 |
   | 0.20 | 32.03% | +0.72 | 0.0316 |
   The optimum is T=0.07-0.10; T=0.20 is the worst setting at
   32.0%, which is still significant against the original
   (p = 0.0316) but falls *below* the 5-epoch CLIP model, so the method
   is more sensitive to too-high temperature than to too-low.

## Corrections applied to the requested text

Every requested figure was checked against the experiment JSONs; these were changed
to match what the runs actually measured.

* **Chord degradation is -3.6 points, not -1.9.**
  Chord is measured on the 10,042-example validation set, whose no-navigation
  baseline is 31.3%, whereas -1.9 was computed against the
  5,042-example honest baseline of 29.7%. Comparing a
  measurement on one set against a baseline from another is not valid.
* **The chord angle is 116.7 degrees, not ~127 degrees.** The measured
  mean cos(ctx, nav) is -0.4486, whose arccosine is 116.7
  degrees.
* **Tangent and continuation are not "n.s." against no-navigation.** Both are
  significantly *worse*: p < 0.0001 and
  p = 0.0003. They are only non-significant relative to
  GPT-2 small.
* **The "gap_t = +0.0030 vs -0.0085" pair was not reproducible.** The +0.0030 value
  matches the *chord* gap (0.0031), not the tangent or continuation target,
  and no saved diagnostic contains -0.0085. Finding 4 reports the verified gap
  progression instead.
* **The random baseline is reported at chance (25.0%).** The measured value on the
  500-example subset was 25.8%; both appear in Table 1's caption.

All accuracy figures above are traceable to `experiments/results_honest.json`,
`results_full_val.json`, `results_tangent.json`, `results_continuation.json`,
`results_additive.json` and `navigation_diagnosis*.json` (findings 1-5), and to
`experiments/results_structure_comparison.json`, `results_clip.json`,
`results_clip_extended.json` and `structured_geometry_v2.json` (findings 6-8).
Every $p$-value in findings 6-8 is an exact McNemar test recomputed from the stored
per-example predictions on the same 10,042 examples, so all comparisons are
same-set.
