# NHS PROMs ML Pipeline

Production-ready machine learning pipeline for predicting **post-operative health gain** after Hip and Knee replacement surgery, built on NHS Patient Reported Outcome Measures (PROMs) data.

> **Who is this for?**
> - **Data engineers / ML engineers** — running training and retraining jobs
> - **Clinicians / data analysts** — generating predictions for individual patients
> - **Developers** — extending the pipeline with new models or procedures

## What it Does

Given a patient's pre-operative profile (demographics, Oxford score dimensions, EQ-5D, comorbidities), the pipeline predicts their expected post-operative **health gain** — the change in their Oxford Knee/Hip Score after surgery.

A **clinical benefit classification** is derived by comparing the predicted gain against the procedure-specific Minimum Clinically Important Difference (MCID):

| Procedure | Score | MCID |
|-----------|-------|------|
| Knee | Oxford Knee Score (OKS, 0–48) | **5 pts** |
| Hip | Oxford Hip Score (OHS, 0–48) | **6 pts** |

If `predicted_health_gain ≥ MCID` → patient is predicted to **benefit** from surgery.

---

## Project Structure

```
nhs_proms_pipeline/
├── pyproject.toml                    # Package metadata and dependency list
├── requirements.txt                  # Pip-installable dependencies
├── .env.example                      # Template — copy to .env and fill in paths
├── README.md                         # This file
│
├── nhs_proms_pipeline/               # Main Python package
│   ├── config/
│   │   └── settings.py               # Pydantic settings (all config via .env)
│   │
│   ├── schemas/
│   │   ├── patient.py                # PatientRecord — validated doctor input
│   │   └── results.py                # ModelRunResult, BestModelInfo, PredictionResult
│   │
│   ├── data/
│   │   ├── collector.py              # Step 1 — CSV → Parquet, memory optimisation
│   │   ├── preprocessor.py           # Step 2 — value recoding, label encoding
│   │   └── preparator.py             # Step 3 — listwise deletion, derived scores,
│   │                                 #           train/test split, imputation, outcome
│   │
│   ├── features/
│   │   └── eq5d.py                   # EQ-5D-3L index calculator (UK value set)
│   │
│   ├── modelling/
│   │   ├── registry.py               # Model registry — add new models here
│   │   ├── trainer.py                # RepeatedKFold CV + full-train fit
│   │   └── evaluator.py              # Bland-Altman, MCID confusion matrix, calibration
│   │
│   ├── inference/
│   │   └── predictor.py              # Load best model + predict for a single patient
│   │
│   ├── utils/
│   │   ├── io.py                     # Parquet + joblib read/write helpers
│   │   ├── logging.py                # Centralised logging setup
│   │   └── memory.py                 # dtype downcasting, memory optimisation
│   │
│   ├── pipeline.py                   # Orchestrators: TrainingPipeline, InferencePipeline
│   │
│   └── cli/
│       ├── train.py                  # CLI: nhs-train
│       └── predict.py                # CLI: nhs-predict
│
└── tests/
    ├── test_features.py              # EQ-5D unit tests
    ├── test_schemas.py               # Pydantic schema validation tests
    └── test_evaluator.py             # Clinical evaluation metric tests
```

---

## Data Flow — Where Every File Is Saved

Understanding where each artefact lives is essential for operations and debugging.

```
nhs_proms_pipeline/
│
├── data/
│   └── interim/                          ← all intermediate Parquet files
│       ├── 1.1-Reduced.parquet           ← Step 1: raw CSV after dtype optimisation
│       ├── 2.0-preprocessing.parquet     ← Step 2: value recoding + label encoding
│       ├── 2.1-train.parquet             ← Step 3: training set (80 %)
│       └── 2.1-test.parquet              ← Step 3: test / hold-out set (20 %)
│
├── models/
│   ├── all_results_cache.joblib          ← metric cache for every (model × CV run)
│   ├── best_pipeline.joblib              ← sklearn Pipeline for the WINNER model
│   └── best_model_meta.joblib            ← BestModelInfo (name, RMSE, R², F₂, path)
│
└── reports/
    ├── bland_altman_best.png             ← Bland-Altman limits of agreement chart
    └── calibration_best.png              ← Calibration by predicted score decile chart
```

> **Paths are fully configurable in `.env`.**
> The defaults above assume you run commands from inside the `nhs_proms_pipeline/` folder.
> Adjust `INTERIM_DATA_DIR`, `MODELS_DIR`, and `REPORTS_DIR` if running from elsewhere.

---

## Step-by-Step Setup

### Prerequisites

- Python **3.10 or later** (3.11 recommended)
- ~500 MB disk space for data, models, and reports
- Access to the NHS PROMs raw CSV dataset

---

### Step 1 — Open the project folder

All commands below are run from inside `nhs_proms_pipeline/`.

```bash
cd nhs_proms_pipeline
```

---

### Step 2 — Create a virtual environment

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
python -m venv .venv
.venv\Scripts\activate.bat

# Linux / macOS
python -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` at the start of your prompt when the environment is active.

---

### Step 3 — Install the package

```bash
pip install -e .
```

`-e` (editable mode) means code changes take effect immediately without reinstalling.

To also install developer tools (test runner, linting):

```bash
pip install -e ".[dev]"
```

Verify the CLI commands are available:

```bash
nhs-train --help
nhs-predict --help
```

---

### Step 4 — Configure settings

```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

Open `.env` in any text editor and update at minimum:

```ini
# ── REQUIRED ──────────────────────────────────────────────────────────────────
# Folder containing the raw NHS PROMs CSV files (relative or absolute path)
RAW_DATA_DIR=../data/external

# Which procedure to run: KNEE or HIP
PROCEDURE_TYPE=KNEE

# ── OPTIONAL (these defaults work for most setups) ─────────────────────────────
INTERIM_DATA_DIR=./data/interim    # where intermediate Parquet files are written
MODELS_DIR=./models                # where trained models are saved
REPORTS_DIR=./reports              # where evaluation charts (PNG) are saved
RANDOM_SEED=42
TRAIN_TEST_RATIO=0.8               # 80 % train, 20 % test
CV_SPLITS=5
CV_REPEATS=3
LOG_LEVEL=INFO
```

---

### Step 5 — Place the raw data files

Copy your NHS PROMs CSV file into the folder specified by `RAW_DATA_DIR`.

The file name must exactly match (including capitalisation and spaces):

| Procedure | Required file name |
|-----------|-------------------|
| Knee | `Knee Replacement Provider.csv` |
| Hip | `Hip Replacement Provider.csv` |

> **Where to get the data:**
> NHS Digital PROMs publication:
> https://digital.nhs.uk/data-and-information/data-collections-and-data-sets/data-collections/patient-reported-outcome-measures-proms
>
> The pipeline has been tested on data packs from 2016–17, 2017–18, and 2018–19.
> All three annual files can be placed in `RAW_DATA_DIR` together; the pipeline
> reads only the file for the configured procedure.

---

### Step 6 — Verify the setup (optional but recommended)

```bash
pytest
```

Expected result: **28 passed** in roughly 10 seconds.  
No network access or data files are needed to run the unit tests.

---

## Step-by-Step: Training the Pipeline

### Quickstart — full training run

```bash
# Knee (default)
nhs-train

# Hip
nhs-train --procedure HIP
```

Training a full Knee dataset (~130,000 rows, 6 models with repeated cross-validation) typically takes **10–30 minutes** depending on hardware.

---

### What each training step does

#### Step 1 — Data Collection

**Reads:** `RAW_DATA_DIR/Knee Replacement Provider.csv`

**Actions:**
- Replaces the NHS suppression marker `'*'` with `NaN`
- Downcasts `int64` → smallest unsigned integer type (e.g. `uint8`)
- Downcasts `float64` → `float32`
- Converts text columns to `category` dtype

**Why:** The raw CSV is ~85 MB. After optimisation it becomes ~16 MB as a compressed Parquet — a ~80 % reduction that makes all subsequent steps significantly faster.

**Writes:** `INTERIM_DATA_DIR/1.1-Reduced.parquet`

---

#### Step 2 — Pre-processing

**Reads:** `INTERIM_DATA_DIR/1.1-Reduced.parquet`

**Actions:**
- Recodes binary comorbidity columns: `9 → 0` (where `9` means "not applicable" = condition absent)
- Recodes `Pre-Op Q Assisted By` and `Post-Op Q Assisted By`: `9 → 0`, any other value → `1`
- Replaces sentinel values `9`, `999`, `'*'`, `''` with proper `null`
- Label-encodes `Year`, `Age Band`, `Procedure`, `Provider Code` as integers
- Drops `Predicted` columns and the administrative `CSVYear` column

**Writes:** `INTERIM_DATA_DIR/2.0-preprocessing.parquet`

---

#### Step 3 — Data Preparation

**Reads:** `INTERIM_DATA_DIR/2.0-preprocessing.parquet`

This is the most complex step. It implements the CRISP-DM methodology with strict data leakage prevention.

**Sub-steps ran BEFORE the train/test split:**

| Sub-step | Rows removed | Why |
|----------|-------------|-----|
| Listwise delete Oxford score dimensions | ~0.5 % | Missing pre-op dimension = cannot train primary predictor |
| Calculate pre-op score from dimensions | 0 | Fills gaps deterministically — no leakage |
| Calculate EQ-5D index from profile | 0 | UK value set formula — no leakage |
| Listwise delete EQ-5D dimensions | ~3–4 % | Cannot impute (clinically invalid values) |
| Listwise delete Symptom Period | ~0.9 % | Strong predictor — imputing would add noise |
| Remove Post-Op question columns | — | Would cause direct target leakage |
| Listwise delete Age Band / Gender | ~6.6 % | Critical casemix variables — imputing adds bias |
| Listwise delete Post-Op Score | ~0.7 % | Outcome variable cannot be imputed |

**Train/test split: 80 % / 20 %, seed = 42**  
The split happens HERE — after all listwise deletions, before any imputation.  
This is the key leakage-prevention boundary.

**Sub-steps ran AFTER the split (imputation fit on training set only):**

| Column | Strategy | Value |
|--------|----------|-------|
| `Pre-Op Q Assisted` | Mode of training set | Typically 2 (No) |
| `Pre-Op Q Previous Surgery` | Mode of training set | Typically 2 (No) |
| `Pre-Op Q Living Arrangements` | Constant | 9 (Unknown) |
| `Pre-Op Q Disability` | Constant | 9 (Not Disclosed) |

**Outcome variable created:**

```
health_gain = Post-Op Q Score − Pre-Op Q Score
```

The `Post-Op Q Score` column is then dropped to prevent target leakage.

**Writes:**
- `INTERIM_DATA_DIR/2.1-train.parquet`  
- `INTERIM_DATA_DIR/2.1-test.parquet`

---

#### Step 4 — Model Training

**Reads:** `INTERIM_DATA_DIR/2.1-train.parquet` and `2.1-test.parquet`

For **each** enabled model, the pipeline:

1. Builds a preprocessing transformer:
   - Numeric columns → median imputation → `StandardScaler`
   - Categorical columns → constant-fill → `OneHotEncoder`
2. Runs `RepeatedKFold(n_splits=5, n_repeats=3)` cross-validation on the training set
3. Records CV MAE, CV RMSE, CV R²
4. Fits once on the **full** training set
5. Evaluates on the held-out test set (Test MAE, Test RMSE, Test R²)

**Model comparison and winner selection:**  
The model with the lowest **Test RMSE** is selected as the winner.

**Clinical evaluation of the winner:**
- Bland-Altman limits of agreement vs MCID threshold
- MCID-based confusion matrix (F₂ score prioritised — reduces denied-surgery errors)
- Calibration by predicted score decile

**Writes:**
- `MODELS_DIR/all_results_cache.joblib` — updated after every model (allows resuming)
- `MODELS_DIR/best_pipeline.joblib` — the fitted sklearn Pipeline for the winner
- `MODELS_DIR/best_model_meta.joblib` — model name, Test RMSE, Test R², MCID F₂, path
- `REPORTS_DIR/bland_altman_best.png`
- `REPORTS_DIR/calibration_best.png`

---

### Models trained by default

| Model | Algorithm | Key hyperparameters |
|-------|-----------|---------------------|
| `LinearRegression` | OLS regression | None |
| `Ridge` | L2-regularised regression | α = 1.0 |
| `Lasso` | L1-regularised regression | α = 0.01 |
| `RandomForest` | Ensemble of decision trees | 200 trees, all CPU cores |
| `GradientBoosting` | Histogram gradient boosting | 300 iterations |
| `XGBoost` | Extreme gradient boosting | 300 rounds, lr = 0.05, subsample = 0.8 |

---

### Resuming a failed or partial training run

Each step writes its output before the next step begins. If training crashes, skip the completed steps:

```bash
# Only data prep and modelling failed — re-run from there
nhs-train --skip-collection --skip-preprocessing

# Only modelling failed — re-run that alone
nhs-train --skip-collection --skip-preprocessing --skip-preparation

# List all available flags
nhs-train --help
```

---

### Retraining when new NHS PROMs data is released

When a new annual data pack is available:

1. Copy the new CSV into `RAW_DATA_DIR` (replace the old file, or append data first using external tooling)
2. Run the full pipeline — all intermediate files and models are automatically overwritten with the results from the new data

```bash
nhs-train --procedure KNEE
```

The best model is automatically re-selected and `best_pipeline.joblib` is replaced. Predictions made after retraining will use the new model.

---

## Step-by-Step: Predicting for a Patient

Training must complete before prediction. If you see a `FileNotFoundError` about `best_model_meta.joblib`, run `nhs-train` first.

---

### Option A — JSON file (recommended for production / auditing)

**1. Create `patient.json`** using the field reference below:

```json
{
  "age_band": "65 to 69",
  "gender": 2.0,
  "pre_op_q_1": 3, "pre_op_q_2": 3, "pre_op_q_3": 3, "pre_op_q_4": 3,
  "pre_op_q_5": 3, "pre_op_q_6": 3, "pre_op_q_7": 3, "pre_op_q_8": 3,
  "pre_op_q_9": 3, "pre_op_q_10": 3, "pre_op_q_11": 3, "pre_op_q_12": 3,
  "eq5d": {
    "mobility": 2,
    "self_care": 1,
    "activity": 2,
    "discomfort": 3,
    "anxiety": 1
  },
  "symptom_period": 2,
  "comorbidities": {
    "arthritis": 0, "cancer": 0, "circulation": 0, "depression": 0,
    "diabetes": 1, "heart_disease": 0, "high_bp": 1, "kidney_disease": 0,
    "liver_disease": 0, "lung_disease": 0, "nervous_system": 0, "stroke": 0
  }
}
```

**2. Run prediction:**

```bash
nhs-predict --input patient.json
```

**3. Save the result JSON for auditing:**

```bash
nhs-predict --input patient.json --output result.json
```

---

### Option B — Interactive prompt

```bash
nhs-predict
```

The CLI will prompt for each field one at a time with allowed values shown inline.

---

### Option C — Print a sample JSON

```bash
nhs-predict --show-sample
```

Prints a fully populated example you can copy-paste and edit.

---

### Reading the prediction output

```
┌─────────────────────────────┬──────────────────────────────────────┐
│ Procedure                   │ KNEE                                 │
│ Predicted Health Gain       │ +9.4 pts                             │
│ Expected Benefit            │ YES — Expected to benefit            │
│ MCID Threshold              │ 5 pts                                │
│ Model                       │ GradientBoosting                     │
└─────────────────────────────┴──────────────────────────────────────┘

Clinical Note:
  Model F₂ (clinical recall) = 0.832.
  Predicted health gain: 9.4 pts (MCID threshold: 5 pts).
  This prediction is generated by a regression model trained on
  NHS PROMs data and should be used alongside clinical judgement.
```

| Output field | Meaning |
|-------------|---------|
| `Predicted Health Gain` | Expected change in Oxford score (Post-Op − Pre-Op). Higher = better outcome. |
| `Expected Benefit` | `YES` if predicted gain ≥ MCID. `NO` otherwise. |
| `MCID Threshold` | Minimum gain considered clinically worthwhile for this procedure. |
| `Model` | Name of the winning model used for this prediction. |
| `F₂` | How often the model correctly identifies patients who benefit (higher = fewer missed cases). |

---

## Patient Input Field Reference

### Demographics

| Field | Type | Valid values |
|-------|------|-------------|
| `age_band` | string | `"Under 45"` · `"45 to 49"` · `"50 to 54"` · `"55 to 59"` · `"60 to 64"` · `"65 to 69"` · `"70 to 74"` · `"75 to 79"` · `"80 to 84"` · `"85 to 89"` · `"90 and over"` |
| `gender` | float | `1.0` = Male · `2.0` = Female |

### Oxford Knee/Hip Score — 12 Pre-Op Dimensions (`pre_op_q_1` … `pre_op_q_12`)

Each question scores from **1** (best — no difficulty) to **5** (worst — severe difficulty or unable to do).

The total pre-op score (range 12–60; **lower = better** function) is computed automatically.

### EQ-5D Dimensions (`eq5d` object)

| Field | Dimension | Valid values |
|-------|-----------|-------------|
| `mobility` | Mobility | `1` No problems · `2` Moderate · `3` Extreme |
| `self_care` | Self-care | `1` No problems · `2` Moderate · `3` Extreme |
| `activity` | Usual activities | `1` No problems · `2` Moderate · `3` Extreme |
| `discomfort` | Pain / discomfort | `1` No problems · `2` Moderate · `3` Extreme |
| `anxiety` | Anxiety / depression | `1` No problems · `2` Moderate · `3` Extreme |

### Clinical History

| Field | Valid values | Default | Notes |
|-------|-------------|---------|-------|
| `symptom_period` | `1`–`4` | **Required** | `1`=<1yr · `2`=1–5yrs · `3`=6–10yrs · `4`=>10yrs |
| `previous_surgery` | `1` or `2` | `2` | `1`=Yes · `2`=No |
| `assisted` | `1`, `2`, or omit | `null` → imputed | Whether questionnaire needed assistance |
| `living_arrangements` | integer | `9` | `9` = Unknown / Not Disclosed |
| `disability` | integer | `9` | `9` = Not Disclosed |

### Comorbidities (`comorbidities` object)

All fields default to `0` (condition absent). Set to `1` if the patient has the condition.

| Field | Condition |
|-------|-----------|
| `arthritis` | Arthritis |
| `cancer` | Cancer |
| `circulation` | Circulation problems |
| `depression` | Depression |
| `diabetes` | Diabetes |
| `heart_disease` | Heart disease |
| `high_bp` | High blood pressure |
| `kidney_disease` | Kidney disease |
| `liver_disease` | Liver disease |
| `lung_disease` | Lung disease |
| `nervous_system` | Nervous system condition |
| `stroke` | Stroke |

---

## Using the Python API

### Training from code

```python
from nhs_proms_pipeline.config import PipelineSettings, ProcedureType
from nhs_proms_pipeline.pipeline import TrainingPipeline

settings = PipelineSettings(procedure_type=ProcedureType.KNEE)
pipeline = TrainingPipeline(settings)

# Full run
best = pipeline.run()

# Skip steps whose output files already exist on disk
best = pipeline.run(skip_collection=True, skip_preprocessing=True)

print(f"Best model : {best.model_name}")
print(f"Test RMSE  : {best.test_rmse:.4f}")
print(f"Test R²    : {best.test_r2:.4f}")
print(f"MCID F₂    : {best.mcid_f2:.4f}")
print(f"Saved at   : {best.pipeline_path}")
```

### Prediction from code

```python
from nhs_proms_pipeline.config import PipelineSettings, ProcedureType
from nhs_proms_pipeline.pipeline import InferencePipeline
from nhs_proms_pipeline.schemas.patient import (
    ComorbidityProfile, PatientRecord, PreOpEQ5D,
)

settings = PipelineSettings(procedure_type=ProcedureType.KNEE)

# Create once and reuse — the model is loaded from disk on __init__
pipe = InferencePipeline(settings)

patient = PatientRecord(
    age_band="65 to 69",
    gender=2.0,
    pre_op_q_1=3, pre_op_q_2=3, pre_op_q_3=3, pre_op_q_4=3,
    pre_op_q_5=3, pre_op_q_6=3, pre_op_q_7=3, pre_op_q_8=3,
    pre_op_q_9=3, pre_op_q_10=3, pre_op_q_11=3, pre_op_q_12=3,
    eq5d=PreOpEQ5D(mobility=2, self_care=1, activity=2, discomfort=3, anxiety=1),
    symptom_period=2,
    comorbidities=ComorbidityProfile(diabetes=1, high_bp=1),
)

result = pipe.predict(patient)

print(result.predicted_health_gain)      # e.g. 9.4
print(result.predicted_benefit)          # True or False
print(result.confidence_note)            # plain-English note for clinician
print(result.model_dump_json(indent=2))  # full JSON for audit trail / EHR logging
```

---

## Extending the Pipeline

### Adding a New Model

1. Open `nhs_proms_pipeline/modelling/registry.py`
2. Add one entry to `_BASE_MODELS`:

```python
from sklearn.svm import SVR

_BASE_MODELS["SVR"] = SVR(kernel="rbf", C=1.0, epsilon=0.1)
```

3. Optionally restrict which models run via `.env`:

```ini
ENABLED_MODELS=["Ridge","GradientBoosting","SVR"]
```

No other changes needed.

---

### Adding a New Procedure (e.g. Shoulder)

1. Add a value to `ProcedureType` in `config/settings.py`:

```python
class ProcedureType(str, Enum):
    KNEE = "KNEE"
    HIP = "HIP"
    SHOULDER = "SHOULDER"
```

2. Add its column mapping to `ProcedureConfig._CONFIGS` in the same file:

```python
ProcedureType.SHOULDER: {
    "raw_csv_name": "Shoulder Replacement Provider.csv",
    "pre_op_score_col": "Shoulder Replacement Pre-Op Q Score",
    "post_op_score_col": "Shoulder Replacement Post-Op Q Score",
    "pre_op_q_prefix": "Shoulder Replacement Pre-Op Q",
    "mcid_threshold": 10.0,
    "keep_post_op_cols": [...],
    "predicted_col_substrings": [...],
},
```

3. No other changes needed.

---

## Running Tests

```bash
# All tests
pytest

# Verbose
pytest -v

# Specific file
pytest tests/test_features.py

# With coverage report
pytest --cov=nhs_proms_pipeline --cov-report=term-missing
```

Expected: **28 passed** in ~10 seconds with no data files required.

---

## Troubleshooting

### `FileNotFoundError: Raw CSV not found`

The raw CSV is missing or misnamed.

- Check `RAW_DATA_DIR` in your `.env` points to the correct folder.
- Verify the filename matches **exactly** (including capitalisation and spaces):
  - `Knee Replacement Provider.csv`
  - `Hip Replacement Provider.csv`

### `FileNotFoundError: Joblib file not found: best_model_meta.joblib`

You are trying to predict before training has completed.

```bash
nhs-train       # run training first
nhs-predict --input patient.json
```

### `FileNotFoundError: Parquet file not found: 1.1-Reduced.parquet`

The data collection step has not run yet.

```bash
nhs-train                          # run everything
nhs-train --skip-preprocessing     # run collection + the rest
```

### `ValidationError` on patient input

Pydantic rejected one or more field values. The error message shows exactly which field failed and what is expected.

| Common mistake | Fix |
|----------------|-----|
| `"age_band": "60-64"` | Must use `"60 to 64"` (with spaces around "to") |
| `"gender": 0` | Must be `1.0` (Male) or `2.0` (Female) |
| `"pre_op_q_1": 6` | Oxford dimensions must be **1–5** |
| `"eq5d": {"mobility": 4}` | EQ-5D dimensions must be **1, 2, or 3** |
| `"symptom_period": 0` | Must be **1–4** |

### Training is very slow

- Reduce CV repeats for development: `CV_REPEATS=1` in `.env`
- Restrict to faster models: `ENABLED_MODELS=["Ridge","GradientBoosting"]`
- `RandomForest` and `XGBoost` already use all CPU cores (`n_jobs=-1`) — ensure no other CPU-heavy processes are running.

### `ModuleNotFoundError: No module named 'nhs_proms_pipeline'`

The package is not installed in the currently active virtual environment.

```bash
# Activate the venv first
.venv\Scripts\Activate.ps1     # Windows PowerShell
source .venv/bin/activate      # Linux / macOS

# Then install
pip install -e .
```

---

## Environment Variables — Full Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `PROCEDURE_TYPE` | `KNEE` | `KNEE` or `HIP` |
| `RAW_DATA_DIR` | `../data/external` | Folder containing raw NHS PROMs CSV files |
| `INTERIM_DATA_DIR` | `./data/interim` | Folder where intermediate Parquet files are written |
| `MODELS_DIR` | `./models` | Folder where trained model artefacts are saved |
| `REPORTS_DIR` | `./reports` | Folder where evaluation charts (PNG) are saved |
| `RANDOM_SEED` | `42` | Seed for shuffle and cross-validation (reproducibility) |
| `TRAIN_TEST_RATIO` | `0.8` | Fraction of cleaned dataset used for training (0.0–1.0) |
| `CV_SPLITS` | `5` | Number of KFold splits |
| `CV_REPEATS` | `3` | Number of times KFold is repeated |
| `LOG_LEVEL` | `INFO` | `DEBUG` · `INFO` · `WARNING` · `ERROR` |

---

## Clinical Evaluation Reports

### `reports/bland_altman_best.png`

A Bland-Altman plot shows the agreement between the model's predictions and actual outcomes across the held-out test set.

- **Red solid line** — mean bias (systematic over- or under-prediction)
- **Orange dashed lines** — 95 % limits of agreement (bias ± 1.96 × SD)
- **Green dotted lines** — ±MCID threshold

**Interpretation:** If the orange lines fall entirely within the green lines, 95 % of the model's individual predictions are precise enough to be clinically meaningful for a single patient. If the orange lines exceed the green lines, population-level predictions are still valid but single-patient predictions carry more uncertainty.

### `reports/calibration_best.png`

Compares mean predicted vs mean actual `health_gain` across 10 equal-frequency bins (deciles) of the predicted score.

- Points close to the diagonal = well-calibrated model
- Points consistently above the diagonal = model over-predicts for that score range
- Points consistently below the diagonal = model under-predicts for that score range

---

## Clinical Context and Disclaimer

### Outcome Variable

$$\text{health\_gain} = \text{Post-Op Score} - \text{Pre-Op Score}$$

A positive value means the patient's condition improved after surgery.

### Minimum Clinically Important Difference (MCID)

| Procedure | MCID | Source |
|-----------|------|--------|
| Knee (OKS) | **5 pts** | Beard et al. (2015), *Osteoarthritis and Cartilage* |
| Hip (OHS) | **6 pts** | Murray et al. (2007), *Journal of Bone and Joint Surgery* |

### Why F₂ Score for Clinical Evaluation

Standard accuracy metrics weight False Positives and False Negatives equally.  
Clinically, a **False Negative** (predicting no benefit for a patient who would genuinely improve) is worse than a **False Positive** (recommending surgery for someone with marginal benefit).

**F₂** weights recall twice as heavily as precision, making it the primary metric for model selection in this context.

$$F_2 = 5 \times \frac{\text{Precision} \times \text{Recall}}{4 \times \text{Precision} + \text{Recall}}$$

---

> **Disclaimer:** This tool is intended to **support** clinical decision-making, not replace it.
> Predictions must always be used alongside the clinical judgement of qualified medical professionals.
> Model performance should be reviewed regularly as patient populations and treatment protocols evolve.
