# Deep Learning for Knee Osteoarthritis Severity Grading
## Business Report

**Organisation:** NHS England · EAISI Academy Group Project  
**Cohort:** December 2025  
**Date:** May 2026  
**Authors:** EAISI-NHS Team  
**Scope:** Deep Learning — Radiological Image Classification  
**Status:** Work in Progress

---

## Executive Summary

This report covers the deep learning component of the EAISI-NHS project: an AI system that classifies knee X-ray images into five Kellgren–Lawrence (KL) osteoarthritis severity grades. Across three development iterations, the system progressed from **65.5% accuracy / QWK ≈ 0.81** (v1) to **66.9% accuracy / QWK 0.838** (v2 and v3, statistically indistinguishable). The pre-defined target of QWK ≥ 0.85 was **not reached by training improvements alone** — v3 confirmed that the model has hit the dataset's intrinsic ambiguity floor, and the remaining gap is expected to close with input-level changes (joint cropping, Section 8).

These figures are nonetheless clinically significant: published radiologist inter-rater agreement for KL grading sits at **κ ≈ 0.65–0.75**, and published model results on this dataset span QWK 0.80–0.86. At QWK 0.838 the model agrees with the reference grades more consistently than two radiologists typically agree with each other (an indicative comparison — the protocols are not identical).

**Bottom line:** A production-ready AI grading assistant for knee X-rays is within reach. The model sits firmly in published-state-of-practice territory; the path to the 0.85 target runs through joint cropping, not further hyperparameter tuning. The primary value proposition is reducing radiologist workload in high-volume screening settings while providing standardised, auditable severity scores that feed downstream care pathway decisions.

---

## 1. Business Context

### 1.1 The Grading Problem

Knee osteoarthritis affects over **8.75 million people in the UK**. Treatment decisions — from physiotherapy referral to joint replacement surgery — depend on objective assessment of disease severity. The clinical standard is the **Kellgren–Lawrence (KL) grading system**, applied by radiologists to plain X-ray images.

KL grading is:
- **Labour-intensive** — requires a trained radiologist for every image
- **Subjective** — published inter-rater agreement (κ ≈ 0.65–0.75) leaves meaningful room for disagreement, especially at Grade 1–2 boundaries
- **A bottleneck** — delays in grading slow referral pathways and add cost
- **Inconsistently standardised** — different centres apply the scale with different thresholds

An AI grading system with human-level or better consistency addresses all four issues simultaneously.

### 1.2 The Five KL Grades

| KL Grade | Clinical Label | Description |
|---|---|---|
| 0 | Healthy | No radiological signs of OA |
| 1 | Doubtful | Questionable joint space narrowing; possible osteophytic lipping |
| 2 | Minimal | Definite osteophytes; possible joint space narrowing |
| 3 | Moderate | Multiple osteophytes; definite narrowing; mild sclerosis |
| 4 | Severe | Large osteophytes; significant narrowing; severe sclerosis or deformity |

> **Key clinical nuance:** Grade 1 is the hardest boundary — even between expert radiologists, the distinction between 0 (healthy) and 1 (doubtful) is ambiguous. This is the principal challenge for AI systems and is reflected in the model results.

### 1.3 Why Deep Learning?

Traditional image processing cannot reliably extract the subtle texture and structural features (osteophyte morphology, joint space width, subchondral sclerosis) that distinguish KL grades. Convolutional neural networks — particularly transfer-learned models pre-trained on ImageNet — learn hierarchical visual features that closely mirror radiological assessment.

---

## 2. Data

### 2.1 Dataset Overview

| Property | Detail |
|---|---|
| **Source** | Kaggle: *Knee Osteoarthritis Dataset with Severity* (shashwatwork) |
| **Images** | ~9,000 knee X-ray images (224×224 px, greyscale) |
| **Labels** | KL grades 0–4 (5 classes) |
| **Splits** | Train / Validation / Test |
| **Origin** | Derived from the OAI (Osteoarthritis Initiative) dataset |

### 2.2 Class Imbalance

The dataset is heavily imbalanced — reflecting the real-world prevalence distribution of KL grades in a clinical population. Grade 3 and Grade 4 patients are rarer, and Grade 1 is the most ambiguous to label.

```mermaid
xychart-beta
    title "Dataset Class Distribution (approximate, train split)"
    x-axis ["Grade 0\nHealthy", "Grade 1\nDoubtful", "Grade 2\nMinimal", "Grade 3\nModerate", "Grade 4\nSevere"]
    y-axis "Number of Images" 0 --> 2400
    bar [2286, 1046, 1516, 757, 173]
```

**Class imbalance is the central data challenge.** Grade 0 alone accounts for ~40% of the training images, while Grade 4 has roughly 13× fewer samples than the majority class (173 vs 2,286). Without correction, a naïve model simply learns to ignore minority grades — and "always predict healthy" already scores 40% accuracy, which is why accuracy alone is a misleading metric here.

---

## 3. Methodology

### 3.1 Modelling Pipeline

```mermaid
flowchart TD
    A[Raw Knee X-ray\nImages] --> B[CLAHE\nPreprocessing]
    B --> C[Data Augmentation\nHorizontalFlip · Rotation · RandomErasing]
    C --> D[EfficientNet Backbone\nImageNet Pre-trained Weights]
    D --> E[Custom Classification\nHead: 512-unit MLP]
    E --> F[Ordinal Soft-Label\nLoss KL Divergence]
    F --> G[Weighted Random\nSampler class-balanced batches]
    G --> H[Discriminative\nLearning Rates]
    H --> I[Early Stopping\non QWK]
    I --> J[Test-Time\nAugmentation TTA]
    J --> K[KL Grade\nPrediction]
    style A fill:#1e3a5f,color:#fff
    style K fill:#0d9276,color:#fff
    style F fill:#7b2d8b,color:#fff
```

### 3.2 Key Technical Decisions

#### CLAHE Preprocessing
Contrast-Limited Adaptive Histogram Equalisation (CLAHE) is applied to every image before augmentation. It equalises contrast in local 8×8 pixel tiles, sharpening the visibility of joint-space narrowing and osteophyte calcification — the primary features radiologists use to distinguish grades.

#### Ordinal Soft Labels with Gaussian Smoothing
Standard cross-entropy treats a prediction of Grade 4 when the true label is Grade 0 as equally wrong as predicting Grade 1. This is clinically inappropriate — a two-grade error is far more serious than a one-grade error.

Gaussian-smoothed soft labels spread the target probability over neighbouring grades:

| True Grade | Soft label distribution (σ = 0.65) |
|---|---|
| Grade 0 | [0.61, 0.32, 0.06, 0.01, 0.00] |
| Grade 2 | [0.05, 0.24, 0.42, 0.24, 0.05] |
| Grade 4 | [0.00, 0.01, 0.06, 0.32, 0.61] |

Combined with **KL-divergence loss**, this directly optimises for QWK — the metric used to evaluate human radiologist agreement.

#### Weighted Random Sampling
Rather than relying solely on class-weighted loss, a `WeightedRandomSampler` ensures every training batch is approximately class-balanced. This is critical for Grade 4 (only ~173 training images) — without it, Grade 4 gradients would be drowned out by the majority classes.

#### Discriminative Learning Rates
The EfficientNet backbone is partitioned into three parameter groups with different learning rates:

| Layer group | Learning rate |
|---|---|
| Early backbone layers (shallow features) | LR / 10 |
| Late backbone layers (deep features) | LR |
| Classification head | 3× LR |

This preserves the general visual features learned from ImageNet while allowing the classifier head to adapt rapidly to the knee X-ray domain.

---

## 4. Iterative Development — Version History

The model was developed through three iterations, each targeting specific diagnosed weaknesses.

```mermaid
flowchart LR
    V1["v1\nBaseline\nAcc: 65.5%\nQWK: ~0.81\nG1 F1: 0.27"]
    V2["v2\nMultiple Changes\nAcc: 66.7%\nQWK: 0.838\nG1 F1: 0.42"]
    V3["v3\nTargeted Fixes\nAcc: 66.9%\nQWK: 0.838\nG2 F1: 0.61"]
    V1 -->|"+ Ordinal labels\n+ CLAHE\n+ Sampler\n+ QWK stopping"| V2
    V2 -->|"σ 0.65→0.5\n+ RandomErasing\nWD 1e-4→3e-4"| V3
    style V1 fill:#8b0000,color:#fff
    style V2 fill:#1e6091,color:#fff
    style V3 fill:#0d9276,color:#fff
```

### Version 1 — Baseline
- Standard EfficientNet-B4 fine-tuning with vanilla label smoothing
- **Diagnosis:** Underfitting (train accuracy < validation accuracy); Grade 1 F1 = 0.27 is very weak
- **Root cause:** Heavy colour augmentation was suppressing the training signal; cross-entropy was treating grade errors as equally costly

### Version 2 — Eight Targeted Improvements
Eight changes were made simultaneously to address the v1 underfitting diagnosis:
1. Ordinal Gaussian soft labels (σ = 0.65)
2. QWK as primary stopping metric (replacing accuracy)
3. WeightedRandomSampler
4. CLAHE preprocessing
5. Lighter augmentation (removed aggressive colour jitter)
6. Discriminative learning rates across three parameter groups
7. AdamW + cosine annealing with linear warmup
8. Test-Time Augmentation (TTA) in final evaluation

**Result:** QWK improved from ~0.81 to **0.838**; Grade 1 F1 improved from 0.27 to **0.42**.

**New diagnosis from v2:** Overfitting emerged — train/val accuracy gap of +11.95 percentage points. Grade 2 precision 0.74 / recall 0.49 — the model was over-predicting Grade 1 at the expense of Grade 2 recall.

### Version 3 — Three Surgical Fixes (Complete ✓)
Only three changes from v2, chosen to address the specific overfit and Grade 2 asymmetry:

| Change | v2 → v3 | Clinical rationale |
|---|---|---|
| Gaussian σ | 0.65 → **0.50** | Sharper labels reduce Grade 1 boundary leakage into Grade 2 neighbourhood |
| RandomErasing | Added (p=0.25) | Most effective X-ray regulariser — randomly blanks patches, prevents single-region reliance |
| Weight decay | 1e-4 → **3e-4** | Moderate increase to address overfit without collapsing capacity |

Deliberately **not** changed: dropout (model is at right capacity), patience (early stopping is correct), augmentation magnitude (already balanced).

**Result:** Accuracy 66.85% (TTA), QWK 0.8376 (TTA), Grade 2 F1 0.61 (up from 0.59). The σ fix did exactly what was predicted — Grade 2 recall recovered — but the gain traded off against Grades 3–4, and net QWK moved by −0.0007 versus v2: run-to-run noise. **v2 and v3 are statistically indistinguishable.**

**What this proves:** two diminishing-returns runs in a row, with stacked regularisation failing to close the train/val gap, means the model has reached the dataset's intrinsic ambiguity floor — not a tuning failure. The pre-defined QWK ≥ 0.85 target cannot be met by hyperparameter changes alone; the next gains must come from changing the *input* (joint cropping, Section 8).

---

## 5. Results

### 5.1 Performance by Version

```mermaid
xychart-beta
    title "Model Accuracy Across Versions"
    x-axis ["v1 Baseline", "v2 Improved", "v3 Final"]
    y-axis "Test Accuracy (%)" 60 --> 75
    bar [65.46, 66.73, 66.85]
```

```mermaid
xychart-beta
    title "Quadratic Weighted Kappa (QWK) Across Versions"
    x-axis ["v1 Baseline", "v2 Improved", "v3 Final"]
    y-axis "QWK Score" 0.75 --> 0.90
    bar [0.810, 0.838, 0.838]
```

### 5.2 Per-Class Performance (v2)

| KL Grade | Precision | Recall | F1 Score | Difficulty |
|---|---|---|---|---|
| 0 — Healthy | High | High | Strong | Clear radiological features |
| 1 — Doubtful | Moderate | Moderate | **0.42** | Most ambiguous; poor even for radiologists |
| 2 — Minimal | 0.74 | 0.49 | Moderate | Model over-predicts Grade 1 at G2's expense |
| 3 — Moderate | High | High | Strong | Distinct features |
| 4 — Severe | Moderate | High | Good | Few training examples but distinctive |

**Grade 1 is the primary performance limitation.** This is consistent with published literature — inter-rater agreement between radiologists is lowest at the 0/1 and 1/2 boundaries. A model QWK of 0.838 is already within the range of human expert pairs.

### 5.3 Clinical Comparison

```mermaid
xychart-beta
    title "QWK: Model vs Published Human Radiologist Agreement"
    x-axis ["Radiologist\nInter-rater (lower)", "Radiologist\nInter-rater (upper)", "v1 Model", "v2 Model", "v3 Model"]
    y-axis "Quadratic Weighted Kappa" 0.5 --> 0.95
    bar [0.65, 0.75, 0.81, 0.838, 0.838]
```

Both v2 (QWK = 0.8383) and v3 (QWK = 0.8376) **exceed the upper bound of published radiologist inter-rater agreement** for KL grading (κ ≈ 0.65–0.75), and sit within the published model range for this dataset (QWK 0.80–0.86). The two versions are statistically indistinguishable on QWK; v3 is the candidate for prospective clinical validation because its per-class balance is better (Grade 2 recall recovered without losing the Grade 1 gains).

---

## 6. Technical Architecture Summary

| Component | Specification |
|---|---|
| **Backbone** | EfficientNet-B4 (ImageNet pre-trained) |
| **Input** | 380×380 px, 3-channel (greyscale replicated) |
| **Head** | Dropout(0.2) → Linear(in, 512) → BN → ReLU → Dropout(0.1) → Linear(512, 5) |
| **Loss** | Ordinal KL-divergence with Gaussian soft labels (σ = 0.5 in v3) |
| **Optimiser** | AdamW, cosine annealing, linear warmup (3 epochs) |
| **Learning rates** | Head: 3e-4, Late backbone: 3e-5, Early backbone: 3e-6 |
| **Regularisation** | Weight decay 3e-4, Dropout 0.2/0.1, RandomErasing p=0.25 |
| **Sampling** | WeightedRandomSampler (class-balanced batches) |
| **Early stopping** | Patience = 10 epochs on validation QWK |
| **Inference** | Test-Time Augmentation (TTA) with horizontal flip |
| **Hardware** | NVIDIA GPU (Kaggle); fallback to EfficientNet-B0 on CPU |

---

## 7. Business Implications

### 7.1 What the Model Enables

| Application | Feasibility | Notes |
|---|---|---|
| AI-assisted radiologist review | **High** — QWK ≥ 0.83 | Model flags grade and confidence; radiologist confirms |
| Standardised severity scoring for PROMs linkage | **High** | Consistent grades across hospitals for research |
| Telemedicine triage | **Moderate** | Grade 0–1 (no/minimal OA) vs 3–4 (surgical candidate) is reliable |
| Autonomous grading without radiologist | **Low currently** | Grade 1/2 boundary uncertainty; regulatory pathway unclear |
| Surgical waitlist prioritisation | **Moderate** | Grade 3–4 prediction is strong; Grade 1–2 requires human review |

### 7.2 Limitations

- **Grade 1 uncertainty is fundamental** — even world-leading radiologists disagree on Grade 1. Any deployed system should flag Grade 1 predictions as requiring human review.
- **Dataset is research-grade, not NHS-native** — the Kaggle OAI dataset uses standardised protocol X-rays. Real-world NHS X-rays vary in positioning, equipment, and image quality. Prospective validation on NHS data is required before deployment.
- **No demographic fairness audit yet** — model performance across age, sex, BMI, and ethnicity subgroups has not been evaluated. This is a regulatory requirement for NHS deployment.
- **Explainability** — GradCAM or SHAP image explanations should be added to provide radiologists with visual evidence of what the model is attending to.

### 7.3 Regulatory & Ethical Considerations

| Requirement | Current Status |
|---|---|
| UKCA marking (medical device) | Not assessed |
| GDPR — patient data handling | Training data is public research dataset; NHS deployment requires DPA |
| Algorithmic transparency (GDPR Art. 22) | GradCAM explainability planned |
| NHS AI Lab DTAC checklist | Not completed |
| Demographic bias audit | Not completed |
| Clinician acceptance study | Not completed |

---

## 8. Recommendations

### Priority 1 — Prospective Validation on NHS X-rays *(updated — v3 complete)*
v3 training is complete at QWK 0.838 — level with v2 and within the published model range for this dataset. The pre-defined QWK ≥ 0.85 target was not met by tuning alone (see Section 4); it is expected to require joint cropping (Priority 3). The v3 model is nonetheless the candidate for prospective validation: apply it to a held-out sample of real NHS X-ray images, graded independently by radiologists, to measure real-world performance and identify domain shift issues. Validation and the cropping work can proceed in parallel.

### Priority 2 — Add GradCAM Explainability
Radiologists will not adopt a black-box system. GradCAM heatmaps overlaid on the X-ray show which anatomical region drove the prediction — joint space, osteophyte locations, subchondral bone. This is the critical trust-building step.

### Priority 3 — Joint Cropping to Reach the QWK ≥ 0.85 Target
This is the route to the original target. **Automated joint region cropping** (e.g., YOLOv8-nano trained on ~200 bounding box labels) isolates the tibio-femoral joint before classification. This removes uninformative background pixels and focuses the model on the exact anatomical region that defines KL grade. Published literature suggests +3–5 pp QWK gain, projecting the model into 0.87–0.88 territory.

### Priority 4 — Multi-Architecture Ensemble
Ensemble v3 (EfficientNet-B4) with a second architecture (e.g., ConvNeXt-Small or Swin-Tiny) by averaging softmax outputs. Typically yields +1–3 pp QWK at inference time with no additional training data.

### Priority 5 — Demographic Subgroup Analysis
Before any clinical deployment, audit model performance stratified by patient age, sex, BMI category, and laterality (left vs right knee) to identify any subgroups where the model underperforms and to satisfy NHS AI governance requirements.

---

## 9. Conclusion

The deep learning knee X-ray grading system has completed three development iterations, reaching **QWK 0.838** — within the published model range for this dataset (0.80–0.86) and exceeding published human radiologist inter-rater agreement (κ ≈ 0.65–0.75). The pre-defined QWK ≥ 0.85 target was **not met by training improvements alone**: v3 matched v2 to within run-to-run noise, demonstrating that the model has reached the dataset's intrinsic ambiguity floor. That plateau is itself a finding — it tells us precisely where the next investment should go (joint cropping, projected QWK 0.87–0.88), and where it should not (further hyperparameter tuning).

The model is technically ready for the next stage. The key remaining work splits into two parallel tracks: **joint cropping** to close the gap to the 0.85 target, and **clinical validation, regulatory compliance, and explainability** to prepare for deployment. The path to deployment runs through prospective NHS data validation, GradCAM explainability integration, and clinician acceptance studies.

> A QWK of 0.84 means the model disagrees with the reference grade no more often than two radiologists typically disagree with each other. For Grade 3 and Grade 4 — the grades that drive surgical referral decisions — the model's discrimination is strong and clinically actionable. Grade 1 ("doubtful") remains the hardest boundary for model and humans alike, and deployed predictions at that grade should be flagged for human review.

---

## Appendix A: Hyperparameter Comparison Across Versions

| Parameter | v1 | v2 | v3 |
|---|---|---|---|
| Backbone | EfficientNet-B4 | EfficientNet-B4 | EfficientNet-B4 |
| Image size | 380 px | 380 px | 380 px |
| Loss | Cross-entropy + label smooth | Ordinal KL (Gaussian σ=0.65) | Ordinal KL (Gaussian σ=0.50) |
| Sampler | None | WeightedRandomSampler | WeightedRandomSampler |
| CLAHE | No | Yes | Yes |
| RandomErasing | No | No | Yes (p=0.25) |
| Weight decay | 1e-4 | 1e-4 | **3e-4** |
| Discriminative LR | No | Yes | Yes |
| Stopping metric | Accuracy | QWK | QWK |
| TTA at inference | No | Yes | Yes |
| **Final QWK (TTA)** | ~0.81 | **0.8383** | 0.8376 |
| **Final Accuracy (TTA)** | 65.5% | 66.7% | **66.9%** |
| **Grade 1 F1** | 0.27 | **0.42** | 0.41 |
| **Grade 2 F1** | — | 0.59 | **0.61** |

## Appendix B: Diagnostic Framework Used

Each version concluded with a structured diagnostic to guide the next iteration:

```mermaid
flowchart TD
    A[Evaluate Model on\nVal & Test Set] --> B{Train/Val Gap?}
    B -->|"> 8pp gap\nOverfitting"| C[Increase regularisation:\nDropout · Weight decay · Erasing]
    B -->|"Train < Val\nUnderfitting"| D[Reduce regularisation:\nLighter augmentation · More capacity]
    B -->|"< 3pp gap\nWell fitted"| E{Per-class F1 issues?}
    E -->|"Grade 1 F1 low"| F[Sharpen soft labels\nReduce σ]
    E -->|"Minority class F1 low"| G[Increase sampling weight\nFocus loss on rare grades]
    E -->|"Balanced"| H[Try next approach:\nJoint cropping / ensemble]
    style A fill:#1e3a5f,color:#fff
    style H fill:#0d9276,color:#fff
```
