# Key Findings

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
`results_additive.json` and `navigation_diagnosis*.json`.
