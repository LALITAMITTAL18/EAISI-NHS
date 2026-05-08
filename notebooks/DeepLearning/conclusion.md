# Chain of Thought: Knee Osteoarthritis Severity Classification

A complete reasoning trace through this project, from problem framing to current state. The point of writing it this way is that for research, the *reasoning* matters as much as the result — readers and reviewers want to see why each choice was made, what alternatives were rejected, and what each experiment actually proved.

---

## Stage 1: Understanding the problem before touching code

Before writing a single line of code, it is worth being honest about what is actually being solved. The dataset provides knee X-rays labeled with Kellgren-Lawrence grades 0–4. The naive framing is "5-class image classification," but that framing is wrong in three subtle ways, and getting them wrong upfront is what causes most papers in this space to underperform.

**First wrong framing: it is not categorical, it is ordinal.** KL grades have an order. Grade 0 is closer to Grade 1 than to Grade 4. A model that predicts Grade 4 when the truth is Grade 0 should be punished much more than one that predicts Grade 1. Cross-entropy treats every wrong class as equally wrong. This will be a recurring theme — every design decision needs to respect ordinality.

**Second wrong framing: it is not balanced.** Looking at the dataset: roughly 40% Grade 0, 18% Grade 1, 26% Grade 2, 13% Grade 3, 3% Grade 4. The ratio between most-common and least-common is ~13:1. A model that predicts only Grade 0 would achieve 40% accuracy. Accuracy itself is therefore a misleading metric here — both an ordinal-aware metric *and* a class-imbalance-aware training objective are needed.

**Third wrong framing: KL grading is genuinely ambiguous.** Published inter-rater agreement between expert radiologists on this exact task is around κ = 0.7. That means even two trained doctors disagree on roughly 30% of cases. Grade 1 in particular is defined as "doubtful joint narrowing, possible osteophytic lipping" — the word "doubtful" is in the definition. There is a *ceiling* on achievable accuracy that has nothing to do with model capacity. Reaching 85% accuracy on this dataset should raise suspicion of overfitting to label noise rather than be celebrated.

This implies the right metric is **Quadratic Weighted Kappa (QWK)**. It penalizes errors by squared distance from the truth — a 2-grade miss costs 4× a 1-grade miss. It is the standard for ordinal medical grading, including the original Kaggle Diabetic Retinopathy competition. Published QWK on this dataset is typically 0.80–0.86. The realistic target is therefore QWK ≥ 0.85, with QWK as the headline number and accuracy and per-class F1 as supporting evidence.

## Stage 2: Designing the v1 baseline

Before optimizing, a baseline that actually *works* is needed, even if it is not great. The principle: do not try to do everything at once. Get something running end-to-end, measure it carefully, and let the diagnostic signals indicate what to improve next. This is the opposite of throwing every trick at the wall and hoping something sticks — which is what most blog posts on this topic do, and why their conclusions are unreliable.

For v1, the most defensible "boring" choices were made: EfficientNet-B4 pretrained on ImageNet (good capacity-to-parameter ratio for medical images at 380×380), standard cross-entropy with class weights, mild augmentation (horizontal flip — knees are bilaterally symmetric so this is anatomy-safe — small rotations, color jitter), AdamW optimizer, two-phase training (head only first, then unfreeze). The ordinal loss was deliberately *not* used yet, to see how far standard methods get.

**v1 result: 65.46% test accuracy, Grade 1 F1 = 0.27.** The diagnostic verdict was "underfitting" because final train accuracy was below val accuracy. Looking at the per-class F1s, the disaster zone was exactly where expected — Grade 1, the ambiguous one, with F1 of 0.27 (precision 0.31, recall 0.25). The model was so unsure about Grade 1 that it was barely better than random on that class.

Importantly, the train-below-val pattern indicated the augmentation stack was *too aggressive* for this dataset, suppressing the training signal. This was a useful piece of evidence — it meant that adding ordinal loss without dialing down augmentation might over-correct.

## Stage 3: What v1 told us, and the v2 design

The v1 result surfaced four problems to solve, ordered by what should have the biggest effect:

**Problem 1: Cross-entropy ignores ordinality.** This was the biggest one. The fix was not a tweak — it was a fundamental loss replacement. Gaussian-smoothed soft labels were introduced: instead of a one-hot target, the true Grade 2 becomes something like `[0.005, 0.188, 0.613, 0.188, 0.005]`. Combined with KL-divergence loss, this directly incentivizes predictions that are *close* to the truth, which is exactly what QWK rewards.

Alternatives were considered: CORAL/CORN ordinal heads (more principled but more complex), pure regression with rounding (loses softmax interpretability), Earth Mover's Distance loss (similar effect, more compute). Gaussian-smoothed cross-entropy is the simplest and works well enough to start there.

The σ parameter controls how soft the labels are. Higher σ = more spread = stronger ordinal signal but blurrier class boundaries. σ = 0.65 was chosen as a starting point — a tunable knob to revisit later.

**Problem 2: Class imbalance gets two attacks, not one.** Class-weighted loss alone was not enough — Grade 4 has 173 samples and batch size is 16, so most batches would contain *zero* Grade 4 examples. The gradient signal for the rarest class was too noisy. The fix was a `WeightedRandomSampler` that ensures every batch is approximately class-balanced. Combined with mild class weights (square-root-softened to avoid double-penalizing rare classes when the sampler is also doing rebalancing), this should give Grade 4 a real signal.

**Problem 3: Generic preprocessing is not optimal for X-rays.** Medical imaging has its own conventions. CLAHE (Contrast-Limited Adaptive Histogram Equalization) is standard for chest/skeletal X-rays — it equalizes contrast in local 8×8 tiles, sharpening joint-space narrowing and osteophytes that would otherwise be flattened by global lighting. This is one of those "domain knowledge eats general ML knowledge" wins that comes from reading the medical-imaging literature rather than the deep-learning literature.

**Problem 4: Val accuracy was the wrong early-stopping criterion.** If the metric of interest is QWK, model selection should be done by QWK. The v1 best checkpoint was saved at the epoch with highest val accuracy, which might not be the highest-QWK epoch. Switching the early-stopping criterion to QWK was essentially free and aligned model selection with the actual objective.

A few smaller decisions stacked on top: discriminative learning rates (early backbone layers learn slower than late layers, which learn slower than the head), a proper warmup-then-cosine schedule, and TTA (test-time augmentation) actually wired into the final evaluation rather than defined-but-unused like in v1.

**Things deliberately rejected for v2:** MixUp/CutMix (conflicts with ordinal soft labels — would confuse the gradient signal), focal loss (class weights + sampler is already two attacks on imbalance, focal would be a third and risk over-correcting), joint cropping (high-impact but high-effort, deferred for later), RadImageNet weights (environment friction with downloads and version pinning), 5-fold CV (correct but expensive — fast iteration was the priority).

**v2 result: Test Acc 66.73% (TTA), Test QWK 0.8383 (TTA), Grade 1 F1 0.42.** The Grade 1 F1 jumped from 0.27 to 0.42 — a 54% relative improvement on the hardest class. QWK was now firmly in published-paper range. The training diagnostic now said "high variance" — the regime had moved from underfitting to mildly overfitting, which is a more comfortable problem to be in. But an asymmetry was visible: Grade 2 had precision 0.74 but recall 0.49 — the model was being too cautious about predicting Grade 2 and shoving borderline cases into Grade 1.

## Stage 4: Diagnosing the v2 imbalance and designing v3

This is where the reasoning gets interesting. The naive interpretation of v2's diagnostic ("high variance, increase regularization") would suggest more dropout, more weight decay, more augmentation. But the per-class report told a different story.

The Grade 2 → Grade 1 leakage was not *generic* overfitting — it was a specific consequence of σ = 0.65 being too soft. With σ = 0.65, when the true label is Grade 2, the model's target distribution puts ~19% mass on Grade 1. Over thousands of training steps, this systematically pulls the Grade 2/Grade 1 decision boundary toward Grade 2's territory. The fix was not more dropout; it was *sharper* soft labels.

So v3 changed three things specifically:
1. `GAUSSIAN_SIGMA: 0.65 → 0.5` — sharper soft labels, recover Grade 2 recall
2. `WEIGHT_DECAY: 1e-4 → 3e-4` — modest regularization bump for the gap
3. `+ RandomErasing(p=0.25)` — X-ray-friendly augmentation that breaks single-region reliance

The naive recommendations were deliberately rejected: no dropout bump (model is at the right capacity, more dropout would re-trigger underfitting), no patience tightening (early stopping was already catching the right epoch).

**v3 result: Test Acc 66.85% (TTA), Test QWK 0.8376 (TTA), Grade 2 F1 0.61 (up from 0.59).** The σ change did exactly what was predicted — Grade 2 F1 rose by +0.026 — but the price was paid at Grade 4 (-0.028) and Grade 3 (-0.017). Net QWK change: **−0.0007**, which is run-to-run noise. v2 and v3 are statistically indistinguishable.

## Stage 5: What this proves

Two diminishing-returns runs in a row, with regularization moves all pushing in the same direction and the train-val gap *widening*, was a clear signal: **the model has reached a Pareto frontier.** Accuracy can be shifted *between* classes by tuning σ, but all of them cannot be improved at once with hyperparameter changes. This is the dataset's intrinsic noise floor talking, not a model capacity issue.

The way to know this is the right interpretation, rather than just "the tuning was not aggressive enough," is the failure pattern: more weight decay AND more augmentation AND sharper labels all *together* did not reduce the gap. When stacked regularization fails, the apparent overfit is not conventional overfitting — it is the model memorizing noise that does not exist in val/test (most likely the ~173 Grade 4 images that the WeightedRandomSampler shows it dozens of times per epoch).

So the next 0.02 QWK will not come from more sigma/dropout/decay tweaking. It needs a structural change.

## Stage 6: Where the real improvements live

Three remaining levers, in order of expected effect:

**Lever 1: Joint cropping.** A small object detector (YOLOv8-nano works) trained on ~200 hand-labeled bounding boxes — or borrowed from one of the bounding-box variants of this dataset published separately on Kaggle — to crop tightly to the knee joint before everything else. This is the single biggest remaining lever, expected +3 to +5 pp QWK, putting the model in 0.87–0.88 territory. The reason it works is informational: the joint space and osteophytes are what is clinically diagnostic; femur/tibia shaft and image margins are pure noise. Removing them removes noise from the input signal entirely.

**Lever 2: Multi-architecture ensemble.** Different backbones (EfficientNet, ConvNeXt, Swin) make systematically different errors. Averaging their TTA softmax outputs typically adds 1–3 pp. Compute cost is roughly 3× a single model.

**Lever 3: Self-supervised pretraining.** Pretrain the backbone with MAE or DINO on a much larger pool of unlabeled knee X-rays (OAI cohort has ~50K) before fine-tuning on the labeled 9.7K. Biggest possible lift, biggest setup cost. Only worth it if joint cropping and ensembling have already been exhausted.

For a research framing, the order is: do **joint cropping first**, then ensemble within the cropped pipeline, then consider self-supervised pretraining if those do not reach the target.

## Stage 7: What an honest research write-up looks like

The instinct in research is to report only what worked. The more useful framing is to report the **decision tree** — what was tried, what worked, what did not, and why. This is more useful to readers because:

- The σ = 0.65 → 0.5 result is informative even though it did not move QWK. It proves the v2 sigma was too soft and the trade-off it implies (boundary calibration trades off across ordinal positions). A future researcher might use this to motivate per-position adaptive σ.
- The "stacked regularization failed" finding from v3 is a *positive* result — it pins down the source of the train-val gap (most likely sampler-induced memorization of rare-class images) rather than treating it as a generic overfit. This is testable: replace the WeightedRandomSampler with class-weighted loss-only and see if the gap closes.
- The fact that TTA improvement was bigger for v3 (+1.69 pp) than v2 (+0.66 pp) is itself interesting — the regularized v3 is more uncertain on individual views, so averaging benefits more. This suggests TTA value scales with model uncertainty, which has implications for how TTA budget should be allocated.

## Stage 8: Open questions worth investigating

For a serious research extension, these are the questions worth pursuing:

1. **Is sigma the right form of soft label?** Gaussian assumes symmetric uncertainty. Real radiologist disagreement might be asymmetric — e.g., G1 vs G2 might be harder than G2 vs G3. Per-position sigma values could be fit from inter-rater data.

2. **Can the model recover the radiologist's reasoning?** Grad-CAM should attend to joint space, osteophytes, and tibial spines for high-grade predictions. If it attends to image margins, text annotations, or femur shafts, the model is exploiting non-clinical features and the result is on shaky ground.

3. **Does the model agree with humans where humans agree, and disagree where humans disagree?** Computing model confidence vs. inter-rater agreement on individual images would test whether the model's uncertainty is calibrated to the dataset's intrinsic ambiguity. If yes, the apparent ceiling is a *good* result, not a failure.

4. **Does ensemble error correlation match what theory predicts?** v2 and v3 differ only in σ — their errors should be highly correlated on most images and decorrelated only near the Grade 1/2 and Grade 3/4 boundaries. If empirical correlation matches this, it validates the soft-label-as-decision-boundary-shift mental model.

5. **Per-class calibration.** Are predicted probabilities well-calibrated? On rare classes especially, models are often over-confident. ECE (Expected Calibration Error) per class would quantify this.

## Stage 9: Tying it back to the meta-lesson

The thing worth emphasizing for a research narrative: every move was driven by a specific signal in the previous run's output, not by the general advice "improve accuracy." v1's underfitting verdict drove v2's loss replacement. v2's per-class asymmetry drove v3's sigma change. v3's null result drove the conclusion that the next step needs to change *the input*, not the *training*.

That is the core methodology: **let the diagnostics steer, not the literature.** Literature gives you the menu of techniques; diagnostics tell you which dish to order. A blog post that throws CLAHE + ordinal loss + ensemble + joint cropping at the problem in one go and reports a final number is not a research contribution. The contribution is showing which interventions moved which metric, in what order, and why.

The current state — QWK 0.84, Grade 1 F1 0.41, single-model — is a defensible plateau. The next defensible step is joint cropping. Everything else is rearranging deck chairs at the same QWK.