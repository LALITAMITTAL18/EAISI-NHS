# NHS PROMs Knee Replacement — Modelling Conclusions

> **Dataset:** NHS PROMs Knee Replacement (2016–2019), ~87k patients  
> **Features:** 12 Oxford Knee Score (OKS) pre-op questions, EQ-5D dimensions, demographics  
> **Primary target:** `health_gain = post-op OKS score − pre-op OKS score` (range −37 to +47, mean 16.8, std 9.7)

---

## What We Tried

### 3.1 — Classification: Predict Non-Benefiters

**Goal:** Binary classification — will this patient *not* benefit (`health_gain < 7`)?

**Strategy:** Treat it as a high-stakes screening problem. Optimise for **Precision ≥ 80%** on the minority class (non-benefiters, ~15%) using Bayesian HPO (Optuna, 40 trials) to maximise PR-AUC, followed by threshold tuning.

**Models:** Logistic Regression, Random Forest, HistGradientBoosting, XGBoost, CatBoost  
**Datasets:** Pipeline1, 2.1-Manual

**Results:**

| Model (2.1-Manual) | ROC-AUC | PR-AUC | Tuned Precision | Tuned Recall |
|---|---|---|---|---|
| Logistic Regression | 0.720 | 0.337 | 80%+ | <1% |
| Random Forest | 0.718 | 0.348 | 80.0% | 4.6% |
| HistGBM | 0.720 | 0.352 | 80.2% | **4.9%** |
| XGBoost | 0.722 | 0.356 | 80.2% | 4.4% |
| CatBoost | 0.724 | 0.355 | 80.2% | 4.3% |

**Best:** 2.1-Manual × HistGBM — 80.2% precision, 4.9% recall at high threshold (~0.87).

**Key finding:** ROC-AUC ≈ 0.72 shows the problem is learnable to a degree. However, the severe class imbalance (85/15) forces a brutal precision–recall trade-off. At 80% precision, models identify only ~5% of true non-benefiters — the vast majority are missed. The model is useful for a very high-confidence filter but cannot serve as a broad screening tool.

---

### 3.2 — Optimised Regression: Bayesian HPO Across All Models

**Goal:** Regress `health_gain` directly. Replace fixed hyperparameters with Optuna Bayesian HPO (40 trials per model). Compare across 6 data pipeline variants.

**Strategy:** GroupKFold by Year (prevents data leakage), recency weighting (oldest year = 0.7x), then test-set evaluation on the held-out year.

**Models:** Ridge, Lasso, Random Forest, HistGBM, XGBoost, EBM (Explainable Boosting Machine), CatBoost, MLP  
**Datasets:** 2.1-Manual, 2.2-UMAP-MICE, 2.4-FeatEng, Pipeline1, Pipeline2 (MedianMode), Pipeline3 (MICE)

**Results (Pipeline1 — best dataset):**

| Model | Test RMSE | Test R² |
|---|---|---|
| Ridge / Lasso | 8.10 | 0.285 |
| Random Forest | 8.10 | 0.286 |
| HistGBM | 8.05 | 0.295 |
| XGBoost | 8.03 | 0.298 |
| EBM | 8.02 | 0.299 |
| **CatBoost** | **8.019** | **0.300** |
| MLP | 8.13 | 0.279 |

**Key finding:** All models — from simple linear to deep trees to neural — converge to RMSE ≈ 8.0, R² ≈ 0.30. This is the first hard evidence of an **information ceiling**. CatBoost on Pipeline1 becomes the definitive baseline.

---

### 3.3 — Stacking Ensemble

**Goal:** Can a diversity-aware stacking ensemble outperform any single model?

**Strategy:** Load residuals from all 3.2 models. Greedily select 3 base learners by lowest residual correlation (diversity), then stack with a Ridge meta-learner using 5-fold OOF predictions.

**Results:**

| Dataset | Best Individual RMSE | Stacking RMSE | Gain |
|---|---|---|---|
| 2.1-Manual | 8.105 (CatBoost) | 8.094 | +0.011 |
| 2.4-FeatEng | 8.223 (CatBoost) | 8.106 | +0.117 |

**Key finding:** Stacking yields at most marginal gains. Even the best stacking result (8.094) sits above the Pipeline1 × CatBoost baseline (8.019). Dataset quality matters more than ensemble strategy.

---

### 3.4 — Feature & Target Engineering

**Goal:** Attack the ceiling at the feature/target level, not the model level.

**Baseline:** CatBoost on Pipeline1, Test RMSE = **8.0186**, R² = **0.2952**

**Techniques and outcomes:**

| Technique | Strategy | RMSE | Δ |
|---|---|---|---|
| Yeo-Johnson target transform | Normalise `health_gain` distribution before fitting | 8.032 | −0.013 |
| Regression-to-mean (RTM) features | Add `pre_op_total`, `pre_op_deviation`, `potential_gain` (OKS ceiling effect) | 8.035 | −0.016 |
| Polynomial interactions | Top-5 SHAP features → 10 pairwise interaction columns | 8.041 | −0.022 |
| Target encoding | OOF cross-validated target encoding for categoricals | 8.039 | −0.020 |
| Subgroup models (severity quartiles) | Separate CatBoost per pre-op severity group | 8.055 | −0.037 |
| MLP + QuantileTransformer | 80 Optuna trials, quantile-normalised inputs | 8.144 | −0.125 |
| **Optuna weighted blend** | Blend weights across all experiment OOF predictions | **8.024** | **−0.005** |

**Key finding:** Not one technique improves RMSE by more than 0.02 points. The best result (weighted blend, 8.024) is only 0.005 better than baseline. All roads lead to R² ≈ 0.30.

---

### 3.5 — Reframe: Predict Post-Op OKS Score Directly

**Goal:** Instead of predicting the *change*, predict the *absolute post-op score*, then derive `health_gain = predicted_post_op − pre_op_total` at evaluation time.

**Strategy:** Fresh data pipeline from `2.0-preprocessing.parquet` (raw). Drop all post-op question columns except the target. Listwise deletion (drop any row with a null). Year-based split. CatBoost with 40 Optuna trials.

**Results:**

| Target | RMSE | R² |
|---|---|---|
| Post-op score (raw) | ~6.0 | — |
| Health gain (derived) | **8.159** | 0.302 |
| Baseline (3.2 direct) | 8.019 | 0.295 |

**Key finding:** The reframe does not help. Mathematically, subtracting the pre-op total (a known constant per patient) from the predicted post-op score preserves prediction error exactly. The two framings are algebraically equivalent when pre-op features are available, which they are.

---

### 3.6 — Multi-Task Learning: Joint Regression + Classification

**Goal:** Train a single two-head PyTorch MLP that simultaneously predicts `health_gain` (regression) and `NO_Benefit` probability (binary classification, `health_gain < 7`). Hypothesis: the shared trunk allows the classification signal to regularise the regression head and vice versa.

**Strategy:** Same data pipeline as 3.5 (`2.0-preprocessing.parquet`, listwise deletion, year-based split). Shared trunk (Linear → BatchNorm → ReLU → Dropout, 2–4 layers), branching into a regression head (MSE) and a classification head (BCE). Combined loss: `alpha × MSE + (1−alpha) × BCE`. Optuna over 25 trials (TPE sampler), with 15% chronological hold-out for HPO validation.

**Results:**

| Task | Metric | Multi-Task MLP | Single-Task Baseline | Delta |
|---|---|---|---|---|
| Regression | RMSE | 8.1907 | 8.0186 (CatBoost 3.2) | −0.172 *(worse)* |
| Regression | R² | 0.2968 | 0.2952 | +0.002 *(negligible)* |
| Classification | PR-AUC | 0.3427 | 0.355 (CatBoost 3.1) | −0.012 *(worse)* |
| Classification | ROC-AUC | 0.7129 | 0.724 (CatBoost 3.1) | −0.011 *(worse)* |

Residuals: mean = 0.24, std = 8.19.

**Key finding:** Multi-task learning delivers no improvement on either task. The regression RMSE is 0.17 points *worse* than the CatBoost baseline, and the classification PR-AUC is 0.012 points *worse* than the dedicated classifier. The shared trunk does not successfully transfer signal between tasks — if anything, the two objectives interfere rather than regularise each other. MLP architectures, even with joint training, consistently underperform gradient-boosted trees on this tabular dataset.

**Impact: None.** Both heads are outperformed by their dedicated single-task baselines.

---

### 3.7 — Ordinal Regression: Binned Health Gain (CatBoost MultiClass)

**Goal:** Bin `health_gain` into four clinically meaningful categories and train a CatBoost multi-class classifier. Hypothesis: ordinal classification may be easier to learn than exact regression, and bins map directly to clinical decisions.

**Bins:**

| Class | Label | Condition | Train % | Bin Mean (train) |
|---|---|---|---|---|
| 0 | Worsened | `health_gain < 0` | ~5% | −5.0 |
| 1 | Minimal benefit | `0 ≤ health_gain < 7` | ~12% | 3.5 |
| 2 | Moderate benefit | `7 ≤ health_gain < 15` | ~33% | 10.9 |
| 3 | Strong benefit | `health_gain ≥ 15` | ~50% | 23.1 |

**Strategy:** Same data pipeline as 3.5/3.6. Optuna 40 trials minimising ordinal MAE in 5-fold CV. Final model evaluated by: (a) ordinal metrics in bin space, and (b) a derived RMSE by mapping predicted class → training-set bin mean, for a fair comparison against the regression baseline.

**Results:**

| Metric | Value | Baseline | Delta |
|---|---|---|---|
| Exact accuracy | 67.0% | — | — |
| Adjacent (±1 bin) accuracy | **89.5%** | — | — |
| Ordinal MAE (bin space) | 0.460 | — | — |
| Derived health_gain RMSE | 9.8655 | 8.0186 | **−1.847 *(worse)*** |
| Derived health_gain R² | −0.020 | 0.295 | **−0.315 *(worse)*** |

The confusion matrix shows the model is heavily biased toward predicting "Strong benefit" (class 3), which dominates the training set (~50%). The Worsened and Minimal classes are almost entirely absorbed into Moderate or Strong predictions.

**Key finding:** Ordinal regression is substantially worse than direct regression for RMSE comparison. The derived R² of −0.020 means the binned model performs *below a naïve mean predictor* when evaluated on the continuous `health_gain` scale. The adjacent accuracy of 89.5% looks reasonable on the surface, but it is largely driven by the class imbalance — the model almost always predicts "Strong" or "Moderate", which are adjacent to each other and together account for ~83% of the data.

**Impact: None — significantly worse.** Ordinal binning loses within-bin precision and cannot compete with continuous regression for RMSE. The approach is useful only if the clinical decision genuinely requires a 4-category output and RMSE is not the success criterion.

---

## Why We Cannot Improve Further

The RMSE ≈ 8.0 / R² ≈ 0.30 ceiling is **not a modelling problem — it is a data problem.**

The pre-operative PROMs questionnaire data carries **approximately 30% of the signal** needed to predict individual health gain. The remaining 70% of variance in outcomes is driven by factors that are **not present in the dataset at all**:

- Surgical factors (surgeon skill, implant type, fixation method, theatre time)
- Patient comorbidities (BMI, diabetes, depression, cardiovascular disease)
- Rehabilitation adherence (physiotherapy compliance, activity levels post-op)
- Social factors (support at home, return-to-work pressure)
- Pre-op patient expectations and psychological readiness
- Random/irreducible biological variability in healing response

No amount of feature engineering, normalisation, or model sophistication can recover signal that was never collected.

---

## Regression vs Classification — Which Is Better and When?

### Use Regression When:
- You need to **quantify how much** benefit a patient will receive
- You want to **rank patients** by expected gain for resource prioritisation
- The downstream decision depends on the **magnitude** of improvement
- Reporting to clinicians who need an interpretable predicted score

**This dataset:** Regression is the primary task. `health_gain` is a continuous, clinically meaningful quantity. **Best: CatBoost on Pipeline1, RMSE = 8.019, R² = 0.300.**

### Use Classification When:
- You need a **yes/no clinical decision** (e.g., "flag for further review")
- The consequence of false positives and false negatives is asymmetric
- You want to **screen** for high-risk patients before surgery
- You can tolerate low recall if precision must be very high

**This dataset:** Classification is a useful secondary task for identifying likely non-benefiters. **Best: HistGBM (2.1-Manual), PR-AUC = 0.352, ROC-AUC = 0.720, Precision ≥ 80% at recall ~5%.** The very low recall limits clinical utility to a narrow "high-confidence screening" role only.

### Summary:
| Question | Task | Winner |
|---|---|---|
| "How much will this patient improve?" | Regression | CatBoost, RMSE 8.02 |
| "Will this patient NOT benefit?" | Classification | HistGBM, PR-AUC 0.352 |
| "Which model family is best overall?" | Both | CatBoost > XGBoost > HistGBM > linear > MLP |
| "Is MLP competitive with tree models?" | Both | No — consistently worst |

---

## Next Steps to Improve Further

These are approaches we have **not yet tried** and that could genuinely break the 8.0 ceiling:

### 1. Link with Hospital Episode Statistics (HES) — Highest Impact
NHS PROMs can be linked by NHS number to HES data, which contains:
- **BMI, ASA grade, length of stay** — strong outcome predictors in literature
- **Comorbidities** (ICD-10 codes): diabetes, depression, heart disease, obesity
- **Surgical data**: surgeon GMC number (proxy for skill/volume), implant code, operation time
- **Provider-level variables**: hospital volume, hospital type (teaching vs district general)

This is the single most impactful step. Linked PROMs+HES datasets are available from NHS Digital / NHSE to approved researchers.

### 2. Add Pre-Op Mental Health / Expectation Features
Literature consistently shows **pre-operative anxiety and depression** (e.g., PHQ-9, GAD-7) and **patient expectations** are among the strongest predictors of PROMs outcomes. These are not in the standard questionnaire but are collected in some centres.

### 3. Use Longer Follow-Up Data (1-year OKS)
The current data is 6-month post-op. The **1-year OKS** has lower measurement noise and more stable outcomes. NHS PROMs also collects annual data — using it would reduce irreducible variance.

### 4. Quantile Regression / Prediction Intervals
Rather than predicting a point estimate, fit **quantile regression** (e.g., LightGBM with `objective='quantile'`) to produce a range: *"this patient is likely to gain between 10 and 22 points."* This is more honest and clinically useful than a single number that carries RMSE ≈ 8.

### 5. ~~Multi-Task Learning (joint regression + classification)~~ ✓ Tried — No impact (notebook 3.6)
A two-head PyTorch MLP jointly predicting `health_gain` (regression) and `NO_Benefit` (classification) was implemented and evaluated. Both heads were outperformed by their dedicated single-task baselines. RMSE: 8.191 (vs 8.019 baseline). PR-AUC: 0.343 (vs 0.355 baseline). The shared trunk did not transfer signal beneficially — tasks interfered rather than regularised each other.

### 6. ~~Ordinal Regression on Binned Health Gain~~ ✓ Tried — Significantly worse (notebook 3.7)
CatBoost MultiClass across 4 clinical bins (Worsened / Minimal / Moderate / Strong) was implemented and evaluated. Derived RMSE: 9.866 (vs 8.019 baseline, **1.85 points worse**). R²: −0.020 (below a naïve mean predictor). The approach loses within-bin precision and is dominated by class imbalance. Useful only if the decision genuinely requires 4-category output rather than a continuous score.

### 7. External Validation Across Hospitals
The current pipeline pools all providers. Training a **mixed-effects model** (or including provider as a random effect via LightGBM with provider embeddings) may capture hospital-level outcome variation that inflates residual variance when ignored.

---

## Final Verdict

> The NHS PROMs knee replacement dataset, using pre-operative questionnaire data alone, has an **irreducible prediction floor of approximately RMSE ≈ 8.0 OKS points and R² ≈ 0.30**.  
> This is not a model failure — it is an honest reflection of what pre-operative self-reported data can and cannot tell us about surgical outcomes.  
> Six distinct modelling strategies were exhausted: optimised regression (3.2), stacking ensembles (3.3), feature/target engineering (3.4), target reframing (3.5), multi-task learning (3.6), and ordinal classification (3.7). None broke the ceiling. In fact, ordinal regression (3.7) and multi-task MLP (3.6) both *worsened* results compared to the baseline.  
> To meaningfully improve prediction, the path forward is **richer data (HES linkage, comorbidities, surgical variables)**, not more sophisticated modelling on the current feature set.
