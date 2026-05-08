# ML Pipeline — Stage-by-Stage Plan

## Overview

This document describes each stage of the generalised ML pipeline website, the
techniques available at each stage, the design decisions made and suggestions for
improvement. The pipeline is modelled on the NHS PROMs Knee Replacement
experimental notebooks but is entirely domain-agnostic — it accepts any tabular
dataset in CSV, Excel, Parquet or JSON format.

---

## Project Management

**Purpose:** Allow multiple independent ML experiments to coexist on the same
machine. Each project is isolated with its own datasets, models, session state,
and reports.

**Implementation:**
- Projects stored in `outputs/<slug>/` (e.g. `outputs/nhs_knee_replacement/`)
- Project index at `outputs/projects.json` (list of `{name, slug, description, created_at}`)
- Per-project session state at `outputs/<slug>/session/session_state.json`
- Per-project artifact dirs: `datasets/`, `models/`, `reports/`

**Home page (`app.py`):**
- If no active project → show project selector (tabs: "Open existing" / "Create new")
- Create project form: name (required) + description (optional)
- Project cards show name, description, created date with "Open" / "Delete" buttons
- Delete requires confirmation to prevent accidental loss
- Once a project is selected, it is stored in `st.session_state["_active_project"]`
- Session state is loaded from the project's own JSON on first access

**Sidebar (`shared/nav.py`):**
- Active project name shown below the pipeline title
- "← Switch project" button clears the active project and navigates to home
- Sidebar also enforces the project gate — any page loaded without an active
  project is immediately redirected to `app.py`

**Key functions (`shared/state.py`):**
- `list_projects()` — read `outputs/projects.json`
- `create_project(name, description)` — mkdir + append to index
- `delete_project(slug)` — rmtree + remove from index
- `get_active_project() / set_active_project()` — `st.session_state["_active_project"]`
- `project_datasets_dir() / project_models_dir() / project_reports_dir()` — resolve
  artifact dirs relative to the active project

**Known fix applied:** `shared/state.py` previously contained duplicate `SessionState`
class and persistence function definitions (a second copy using an undefined `STATE_PATH`
constant). The duplicate block was removed; the correct implementations using
`_active_state_path()` are the sole definitions.

---

## Stage 1 — Upload Data

**Purpose:** Ingest raw data from any supported format, apply sentinel value
replacement, define or derive the outcome variable, and perform memory optimisation.

**Supported formats:** CSV (any separator), Excel (.xlsx / .xls), Parquet, JSON.

**Key operations:**
- Auto-detect file format from extension
- Replace user-defined sentinel values (e.g. `*`, `""`, `9`, `999`) with `NaN`
  across all columns — numeric sentinels are cast to the column dtype before comparison
- Downcast integer columns (Int64 → UInt8/16) and float columns (Float64 → Float32)
- Cast low-cardinality string columns to `Categorical`
- Validate UploadConfig (Pydantic v2)
- Save optimised DataFrame to `outputs/<slug>/datasets/1_raw.parquet` (gzip)
- Save session state to disk (`outputs/<slug>/session/session_state.json`)

**Outcome Variable Builder:**

Two modes:
1. **Use existing column** — standard selectbox; same column is used as target.
2. **Derive from formula** — user names a new column and provides a pandas
   `eval()` expression referencing existing columns (e.g. `post_op - pre_op`).
   A live preview shows the distribution of the derived column. The formula and
   column name are stored in `upload_cfg["derived_outcome_formula"]` and
   `upload_cfg["derived_outcome_column"]` so the transformation is reproducible.

**User decisions:**
- Target column
- Task type: Regression / Classification / Ordinal Classification
- Sentinel values (editable list)
- CSV separator

**Outputs:** `1_raw.parquet`, updated `session_state.json`

**Improvement suggestions:**
- Auto-detect sentinel values by inspecting unique value distributions
- Support multi-sheet Excel files (sheet selector dropdown)

---

## Stage 2 — Data Exploration (EDA)

**Purpose:** Understand data distributions, correlations and subgroup patterns
before any cleaning or modelling decisions are made.

**Visualisations (all follow Storytelling with Data rules — gray for context,
accent color for the message):**
- Per-column histograms with KDE overlay; target column highlighted
- Q-Q normality plot for numeric columns
- Pearson / Spearman correlation heatmap; values near ±1 shown darkest
- Feature–target correlation bar chart; highest-correlation feature highlighted
- Subgroup mean deviation bar; groups with above-average deviation highlighted
- Summary statistics table (mean, std, skewness, kurtosis, missing counts)

**Key metrics reported:** skewness, kurtosis, % missing, unique count, top category.

**Target column handling:**
The configured target column (e.g. `health_gain`) may not exist in the raw uploaded
data — it is often a derived column created in a later stage (e.g. post-op minus
pre-op score). The explore stage handles this gracefully:
- Column distribution selectbox defaults to the first column if the target is absent
- Correlation matrix highlights the target only if it is present and numeric
- Feature–target tab shows a selectbox of all numeric columns so the user can pick
  a proxy to explore correlations; an info banner explains why the configured target
  is not available
- Subgroup analysis checks `target in df.columns` before accessing `df[target]`

**Outputs:** No files written (exploratory only). Stage marked complete on review.

**Improvement suggestions:**
- Add interactive pairplot for user-selected feature subset
- Add class distribution overview for classification targets

---

## Stage 3 — Outlier Detection

**Purpose:** Identify and handle extreme values using a dual-flag approach that
reduces false positives compared to single-method detection.

**Methods:**
- **IQR rule:** `Q1 − factor × IQR` / `Q3 + factor × IQR` (default factor = 1.5)
- **Z-score rule:** `|z| > threshold` (default threshold = 3.0)
- **Dual-flag (recommended):** A row is confirmed as an outlier only if flagged by
  BOTH rules simultaneously — same logic as the 1.2-Outlier-Detection notebook

**Per-column actions:**
- `keep` — log the outlier but leave the value unchanged
- `remove` — drop the row from the dataset
- `winsorize` — clip the value to the bound (preserves row count)
- `exempt` — skip this column entirely (e.g. VAS columns where 9 is a valid score)

**Visualisations:**
- Scatter of all values per column; flagged points in accent red, normal in gray
- Horizontal bar of dual-flagged counts per column; highest column highlighted

**Outputs:** `2_outliers.parquet`, updated `session_state.json`

**Known fix applied:** Pages 4, 5 and 6 were loading parquet files using
`_BASE / prev_path` (resolved relative to `MLPortal/`) instead of
`project_datasets_dir() / prev_path`. All three pages now use the correct
project-scoped path.

---

## Stage 4 — Missing Data Analysis & Imputation

**Purpose:** Quantify missingness, visualise patterns, and configure per-column
imputation strategies. Imputers are **fit on training data only** (applied in Stage 6).

**Analysis:**
- Per-column missingness bar chart; columns > threshold highlighted
- Co-missingness heatmap (which columns tend to be missing together)
- **UMAP missingness embedding (opt-in):** binary indicator matrix (1 = missing) →
  2D UMAP reduction → rows cluster by similar missing-data patterns. Requires
  `umap-learn` (optional dependency — install with `pip install umap-learn`)

**Imputation strategies per column:**
| Strategy | When to use |
|---|---|
| `median` | Numeric columns — robust to outliers |
| `mean` | Numeric columns with roughly normal distribution |
| `most_frequent` | Binary / categorical columns |
| `constant` | Domain-specific fill (e.g. 9 = "Unknown") |
| `knn` | When similar rows exist; k is configurable |
| `mice` | Multivariate imputation for correlated numeric columns |
| `drop_rows` | When missingness is informative and rows are expendable |
| `add_indicator` | Add a binary `col_missing` flag, impute separately |

**Key invariant:** `fit_imputer()` and `apply_imputer()` are separate pure
functions. `fit_imputer()` never receives test data.

**Outputs:** Imputation config stored in `session_state.json`; imputation
applied during Stage 6 after the train/test split.

---

## Stage 5 — Feature Engineering

**Purpose:** Create new informative features from existing columns.

**Operations available:**

| Operation | Description |
|---|---|
| Derived features | User-typed formula using pandas `eval()` (e.g. `48 - pre_op_score`) |
| Polynomial / interaction | `PolynomialFeatures(degree=2, interaction_only=True)` on selected columns |
| Binning | Equal-width, equal-frequency, or custom breakpoints → ordered categorical |
| Transforms | Log1p, sqrt, Yeo-Johnson, or quantile (for skewed numeric columns) |
| Aggregations | Group-level mean/std/median by a categorical column (computed on train only) |
| Target encoding | Smoothed target encoding with OOF cross-validation (no leakage) |
| Drop columns | Remove ID, admin, or post-outcome-leakage columns |

**Leakage prevention:** Aggregation statistics and target encoding mappings are
always computed on training data and merged into test data.

**Visualisations:**
- Distribution before/after transform (overlaid histograms)
- Feature–target correlation bar after engineering

**Outputs:** Config stored in `session_state.json`; applied during Stage 6.

---

## Stage 6 — Data Preparation & Dataset Variants

**Purpose:** Create one or more named **dataset variants** — each is a fully
prepared, leakage-free train/test pair. Different variants reflect different data
preparation choices and are stored independently within the project. Stage 7
picks which variant to train on.

### Dataset variant concept

A dataset variant is a named pair `(train.parquet, test.parquet)` stored in
`outputs/<slug>/datasets/<variant_slug>_train.parquet`. Multiple variants coexist
within one project so you can compare the effect of different preparation choices
on model performance without creating separate projects.

Each variant records:
- **name** and **description** (user-supplied)
- **pipeline_method** — short string summarising the preparation choices
- **n_train, n_test, n_features** — shape metadata
- **created_at** — ISO timestamp

The active variant is stored in `session_state.active_dataset` (slug) and its
paths are mirrored to `train_data_path` / `test_data_path` for downstream stages.

### Creating a variant — three routes

#### Route A — Pipeline (default)
Data flows from Stages 1–5. The user configures:
- **Imputation strategy** (use Stage 4 config / median+mode with flags / MICE / drop rows)
- **Feature selection** (include or exclude specific columns from the dataset)
- **Split method** (random or time-based) and split ratio
- **Scaler** and **encoder**
- Optional **Outcome Threshold** to derive a binary label

All operations are applied in leakage-free order: split first, then feature
engineering and imputation fitted on train only.

#### Route B — Upload pre-split files (bypass)
The user uploads their own train and test files (CSV, Excel, Parquet, JSON).
Stages 1–5 are skipped. On save, all prior stages are marked complete.

### Already-prepared variant flow

When an active variant with valid `train_data_path` / `test_data_path` already exists
(e.g. bootstrapped from notebook interim files), Stage 6 detects this and:
- Marks preparation complete automatically
- Shows a green success banner with train/test row counts
- Presents a **"Continue to Modelling →"** button — no split or imputation needed
- A secondary **"➕ Create a different variant"** button reveals the full creation form

This prevents re-splitting already-prepared data and avoids accidental data leakage
when data was prepared by a separate, controlled notebook pipeline.

**Known fix applied:** Stage 6 previously contained a full duplicate copy of the
both bypass and pipeline mode implementations, causing `StreamlitDuplicateElementId`
errors from duplicate radio button keys. The duplicate block was removed.

### Pre-loaded variants in demo projects

**NHS Knee Replacement** (from `notebooks/Experiment/Knee/data/interim/`):

| Variant | Source file | Train rows | Test rows | Features | Target | Notes |
|---|---|---|---|---|---|---|
| Manual Imputation | `2.1-train/test` | 74,400 | 36,312 | 47 | `health_gain` | Hand-crafted, domain-guided imputation |
| Median / Mode + Flags | `2.2-train/test` | 111,388 | 27,848 | 40 | `health_gain` | Median/mode + `*_missing` indicator columns |
| MICE Imputation | `2.4-train/test` | 93,601 | 45,635 | 57 | `health_gain` | sklearn `IterativeImputer`, same missingness flags |
| Reduced Features | `2.6-train/test` | 74,400 | 36,312 | 47 | — | Rare comorbidities and redundant cols dropped |
| Reduced Features — Classification | `2.7-cls-train/test` | 36,957 | 9,239 | 44 | `NO_Benefit` | Binary outcome; reduced feature set |
| Reduced Features — Regression | `2.7-reg-train/test` | 36,957 | 9,239 | 44 | `health_gain` | Regression; reduced feature set |

**NHS Hip Replacement** (from `notebooks/Experiment/Hip/data/interim/`):

| Variant | Source file | Train rows | Test rows | Features | Target | Notes |
|---|---|---|---|---|---|---|
| Manual Imputation | `2.1-train/test` | 68,169 | 33,365 | 49 | `health_gain` | Hand-crafted, domain-guided imputation |

**Data sync note:** All project dataset files are copies of the notebook interim
files. The source of truth is `notebooks/Experiment/{Hip,Knee}/data/interim/`.
If notebooks regenerate those files, re-run the sync script to update the project
`datasets/` folder and session state row counts.

### Imputation strategies available in UI

| Strategy | Description |
|---|---|
| Use config from Stage 4 | Apply the per-column spec defined in the Missing Data stage |
| Median / mode | Numeric → median, categorical → most frequent. Adds `*_missing` flag columns |
| MICE | `sklearn.IterativeImputer` (BayesianRidge, 10 iterations). Adds `*_missing` flag columns |
| Drop rows | Remove any row with at least one missing value (may lose large fractions of data) |

**Scaler options:** StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, none.
**Encoder options:** OneHotEncoder, OrdinalEncoder.

**State schema changes:**
- `SessionState.datasets: list[dict]` — list of all registered variant metadata
- `SessionState.active_dataset: str | None` — slug of the currently active variant
- `state.py` exports `register_dataset()`, `set_active_dataset()`, `list_project_datasets()`
- Artifact files are named `<variant_slug>_train.parquet` (not `5_train.parquet`)

---

## Stage 7 — Model Configuration & Training

**Purpose:** Select a dataset variant, choose models from the registry, configure
Optuna HPO and run training.

**Dataset variant selector** (new, at top of page):
A selectbox shows all available variants with their shape metadata (rows, features,
method). Changing the selection calls `set_active_dataset()` which updates
`train_data_path` / `test_data_path` in state. Model results are cached to files
named `<variant_slug>_results_cache.joblib` so different variants never overwrite
each other.

**Model registry design:**
- All models defined in `website/data/models.json` — no hardcoded Python dicts
- Users add custom models via `website/data/custom_models.json`
- Schema documented in `website/data/models_schema.json` for editor autocomplete
- `ModelRegistry.from_json()` loads both files at startup
- `ModelSpec.check_availability()` verifies the constructor is importable; unavailable
  models are shown as disabled in the UI

**Built-in models:**

*Regression:* Ridge, Lasso, ElasticNet, RandomForest, HistGradientBoosting,
XGBoost, CatBoost, LightGBM, ExplainableBoostingMachine (EBM), MLP, SVR

*Classification:* LogisticRegression, RandomForest, HistGradientBoosting,
XGBoost, CatBoost, LightGBM, MLP, SVC

*Ordinal:* CatBoost (MultiClass loss), RandomForest, HistGBM, LightGBM, MLP

**Optuna HPO:**
- Sampler: TPE (default), CMA-ES, Random
- Pruner: MedianPruner (n_startup_trials=5)
- CV: StratifiedKFold for classification/ordinal; KFold for regression
- Objective metric: user-selected (RMSE, F2, ROC-AUC, etc.)
- Search spaces defined per-model in `models.json`

**Known fix applied (`stages/modelling/trainer.py`):** `_suggest_params()` had a
broken operator-precedence bug in the `step` extraction for `int` params — the
ternary expression always resolved to `None` for object-based specs, passing
`None` to `optuna.suggest_int()` which raised `TypeError: '<=' not supported
between NoneType and int`. Fixed to correctly fall back to `1` when step is unset.

**Fine-tuning:**
After an initial Optuna run, `finetune_model()` narrows the search space to ±20%
(configurable via `finetune_window` in JSON) of the best found parameters and
re-runs Optuna. Warm-start hints are passed via `study.enqueue_trial()`.

**Visualisations:**
- Optuna trial history scatter; best trial highlighted
- CV score bar across all trained models; best highlighted

**Outputs:** `outputs/models/{ModelName}.joblib` per model, `results_cache.joblib`,
`pipeline_config.json` (full config for reproducibility)

**Incremental model persistence:** Each model's `.joblib` file is written to
`outputs/<project>/models/` immediately after that model finishes training (not
after all models complete). This means partial results survive if training is
interrupted — already-trained models do not need to be re-run. The results cache
and pipeline config JSON are still written once at the end of the full run.

---

## Stage 8 — Model Comparison

**Purpose:** Evaluate all trained models on the held-out test set and compare
across multiple metrics, calibration and equity dimensions.

**Metrics:**

*Regression:* RMSE, MAE, R², Outcome-Threshold Precision/Recall/F2

*Classification:* Accuracy, Precision, Recall, F1, F2 (β=2), ROC-AUC, PR-AUC

*Ordinal:* Exact accuracy, Adjacent accuracy (±1 bin), Ordinal MAE

**Comparison views:**
- Per-metric horizontal bar chart; best model highlighted, all others gray
- Bland-Altman limits of agreement (bias ± 1.96 SD) — regression
- Calibration by decile (mean predicted vs mean actual; bubble size ∝ n) — regression
- ROC curve overlay; best model highlighted, others gray
- Equity / subgroup bar: per-group metric, groups with large gaps highlighted
- **Threshold explorer** (classification/ordinal only) — interactive 5-panel chart

**Threshold explorer (classification/ordinal tasks only):**

An interactive 5-panel figure driven by a Streamlit slider for the decision
threshold `k` (0.01 – 0.99). All panels update together on each slider move.

| Panel | Content |
|---|---|
| Left (full height) | KDE density curves for Case vs Non-case; jittered scatter coloured by TP / FP / TN / FN; vertical threshold line; TP / FN / FP / TN box annotations |
| Middle top | TP and FN counts as a function of threshold k; current k marked |
| Middle bottom | FP and TN counts as a function of threshold k; current k marked |
| Right top | ROC curve; current k highlighted; Recall / FPR / AUC readout |
| Right bottom | Precision-Recall curve; current k highlighted; Precision / Recall / AUC readout |

Implemented in `stages/comparison/plots.py` as `threshold_explorer_plot()`.
Uses `scipy.stats.gaussian_kde` for density estimation and
`sklearn.metrics.roc_curve` / `precision_recall_curve` for the curves.

**Best model selection:** Lowest Test RMSE (regression), highest F2 (classification),
lowest Ordinal MAE (ordinal).

**Outputs:** `comparison_table` persisted in `session_state.json`; best model name
stored for downstream stages.

---

## Stage 9 — Model Explanation

**Purpose:** Understand feature importance and individual predictions using
model-agnostic and model-specific explanation techniques.

**Techniques:**

| Technique | Method | Scope |
|---|---|---|
| SHAP summary bar | TreeExplainer / LinearExplainer / KernelExplainer (fallback) | Global |
| SHAP waterfall | Per-row SHAP decomposition | Local (single row) |
| Native importance | `feature_importances_` or `coef_` | Global |
| Permutation importance | Model-agnostic, n_repeats=10 | Global |
| Partial Dependence Plot (PDP) | Average prediction across a grid of one feature | Global |
| Live prediction explorer | Form → instant prediction for any row | Local |

**SHAP computation:**
- TreeExplainer for tree-based models (fast)
- LinearExplainer for linear models
- KernelExplainer fallback for any model (slow; 50-sample background)
- Sample capped at `max_shap_samples` (default 500) for speed

**Visualisations:**
- SHAP bar: all features gray except the top feature (highlighted)
- Waterfall: positive SHAP contributions in accent red, negative in secondary blue
- PDP: clean line chart with accent color

---

## Stage 10 — Conclusions & Export

**Purpose:** Consolidate findings, record analyst notes and export reproducible reports.

**Outputs:**
- Model card: best model name, task, hyperparameters, all metrics
- Performance matrix heatmap: all models × all metrics; best per column highlighted
- HTML report: self-contained with summary, comparison table, key findings
- CSV comparison table: flat export for Excel / further analysis
- `pipeline_config.json`: full serialised pipeline configuration (already saved in Stage 7)

**Analyst notes:** Free-text markdown area auto-saved to `session_state.json` as you type.

---

## Visualization Rules (Storytelling with Data)

All charts in the pipeline follow these rules, enforced via `shared/viz.py`:

1. **Non-message data → gray** (`#BFBFBF`)
2. **The message → single accent** (`#E63946` red)
3. **Secondary comparison → secondary** (`#457B9D` blue)
4. **No gridlines, no chart junk, minimal axis decoration**
5. **Every chart has an annotation** stating the takeaway message in plain language
6. `StoryPalette` is the single source of truth — never hardcode colors in page code

---

## Session Persistence

Every user action triggers `save_state()` which writes `SessionState` to
`outputs/session/session_state.json`. On app restart, `load_state()` restores the
full pipeline state including:
- Which stages are complete
- All configuration objects (as JSON dicts)
- Paths to all artifact files on disk

Stage artifacts (`parquet`, `joblib`) remain on disk independently. The session
state only stores their paths.

---

## Adding New Models

1. Open `website/data/custom_models.json`
2. Add a JSON object matching the schema in `website/data/models_schema.json`
3. Restart the app — the new model appears in Stage 7's model selector

No Python code changes required. The `ModelRegistry.from_json()` class method
handles dynamic import of any `constructor_path`.

---

## Further Considerations

1. **Ordinal task type:** Included as a first-class task (CatBoost MultiClass +
   ordinal-MAE scorer). Configured via the task selector in Stage 1.

2. **UMAP missingness visualisation:** Opt-in in Stage 4. A `try/except ImportError`
   block shows a `pip install umap-learn` message if the package is not installed.
   It is not in `requirements.txt` as a hard dependency.

3. **Custom model JSON schema:** `website/data/models_schema.json` ships with the
   app so any JSON editor (VS Code, PyCharm) provides field autocomplete and
   type validation for `custom_models.json`.

4. **Large datasets:** MICE and UMAP can be slow on >500k rows. Consider adding a
   row-sample cap option for exploratory stages.

5. **Stacking / ensemble:** `StackingRegressor` and `StackingClassifier` can be
   added as custom models in `custom_models.json` once base models are trained.
   A dedicated stacking builder (greedy diversity selection by residual correlation)
   can be added as a future feature in Stage 7.
