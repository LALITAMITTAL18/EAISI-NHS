# ML Pipeline — General-Purpose Streamlit App

A 10-stage, domain-agnostic machine learning pipeline for any tabular dataset.
Upload your data, explore it, clean it, engineer features, train models, compare
results and explain predictions — all in one interactive app.

---

## Quick start

Operating System,Shell / Terminal,Activation Command
macOS / Linux,Bash / Zsh (Default),source .venv/bin/activate
,Fish,source .venv/bin/activate.fish
,Csh / Tcsh,source .venv/bin/activate.csh

Windows,Command Prompt (cmd),.venv\Scripts\activate.bat
,PowerShell,.venv\Scripts\activate.ps1
,Git Bash,source .venv/Scripts/activate

### 1. Install dependencies

```bash
cd MLPortal
pip install -r requirements.txt
```

Optional packages (install if you need them):

```bash
pip install umap-learn   # UMAP missingness visualisation in Stage 4
pip install shap         # SHAP explanations in Stage 9
```

### 2. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

---

## Typical workflow

### Path A — Full pipeline (raw data → trained model)

Follow the stages in order using the sidebar:

| Stage | Page | What you do |
|---|---|---|
| 1 | Upload Data | Drop a CSV / Excel / Parquet / JSON file; select target column and task type |
| 2 | Explore | Review distributions, correlations and subgroup patterns |
| 3 | Outlier Detection | Choose how to handle outliers per column (keep / remove / winsorize) |
| 4 | Missing Data | Configure imputation strategy per column |
| 5 | Feature Engineering | Create derived features, transforms, binning |
| 6 | Preparation & Split | Set train/test ratio, scaler, encoder, Outcome Threshold |
| 7 | Modelling | Select models from the registry; run Optuna HPO |
| 8 | Comparison | Compare all models on the test set; explore the classification threshold |
| 9 | Explanation | SHAP importance, PDP, live prediction explorer |
| 10 | Conclusions | Model card, notes, export CSV / HTML report |

### Path B — Bring your own pre-split data

If your data is already cleaned and split into train / test files:

1. Navigate directly to **Stage 6 — Preparation & Split**
2. Select **"Bypass (upload my own train / test files)"**
3. Upload your train file and test file (CSV / Excel / Parquet / JSON)
4. Set the target column and task type
5. Click **Save pre-split data & Continue →**
6. Proceed from **Stage 7 → Modelling** onwards

Stages 1–5 are marked complete automatically and the sidebar updates accordingly.

---

## Session persistence

Your progress is saved automatically after every action to:

```
MLPortal/outputs/session/session_state.json
```

When you restart the app, it restores your last session — including which stages
are complete, all configuration choices, and the paths to all saved datasets and
models.

To start a fresh session, click **Reset session** in the sidebar.

---

## Adding custom models

Edit `MLPortal/data/custom_models.json` and add a model object. No Python changes
are needed — just restart the app and the new model appears in Stage 7.

Example entry:

```json
[
  {
    "name": "MyExtraTreesRegressor",
    "display_name": "My Extra Trees",
    "task": ["regression"],
    "constructor_path": "sklearn.ensemble.ExtraTreesRegressor",
    "default_params": {"n_estimators": 300, "random_state": 42},
    "optuna_space": {
      "n_estimators": {"type": "int", "low": 100, "high": 700},
      "max_depth":    {"type": "int", "low": 3,   "high": 20}
    },
    "finetune_window": 0.2,
    "requires_scaling": false,
    "supports_warmstart": false,
    "tags": ["tree", "bagging"]
  }
]
```

The schema for all fields is documented in `MLPortal/data/models_schema.json`.

---

## Project structure

```
MLPortal/
├── app.py                    ← entry point (run this)
├── requirements.txt
├── README.md                 ← this file
├── documents/
│   └── pipeline_plan.md      ← detailed plan for each stage
├── data/
│   ├── models.json           ← 22 built-in model specs (no Python hardcoding)
│   ├── custom_models.json    ← add your own models here
│   └── models_schema.json    ← JSON Schema for editor autocomplete
├── shared/
│   ├── viz.py                ← StoryPalette colour system
│   ├── state.py              ← SessionState Pydantic model + persistence
│   ├── io.py                 ← save/load parquet, joblib, json
│   └── nav.py                ← sidebar renderer
├── stages/
│   ├── upload/               ← loader, dtype optimisation
│   ├── explore/              ← summary stats, correlations, Q-Q
│   ├── outliers/             ← IQR + Z-score dual-flag
│   ├── missing/              ← missingness analysis, imputation
│   ├── features/             ← derived features, transforms, encoding
│   ├── preparation/          ← split, scaler, outcome threshold
│   ├── modelling/            ← ModelRegistry, Optuna, fine-tuning
│   ├── comparison/           ← metrics, Bland-Altman, threshold explorer
│   ├── explanation/          ← SHAP, PDP, permutation importance
│   └── conclusions/          ← model card, export HTML/CSV
├── pages/
│   ├── 1_upload.py
│   ├── 2_explore.py
│   ├── 3_outliers.py
│   ├── 4_missing.py
│   ├── 5_features.py
│   ├── 6_preparation.py      ← pipeline mode or bypass mode
│   ├── 7_modelling.py
│   ├── 8_comparison.py       ← includes threshold explorer
│   ├── 9_explanation.py
│   └── 10_conclusions.py
└── outputs/                  ← auto-created; all artifacts saved here
    ├── datasets/             ← parquet files per stage
    ├── models/               ← joblib pipelines + pipeline_config.json
    ├── reports/              ← exported HTML / CSV
    └── session/              ← session_state.json
```

---

## Supported data formats

| Format | Extensions |
|---|---|
| CSV | `.csv` — any separator |
| Excel | `.xlsx`, `.xls` |
| Parquet | `.parquet` |
| JSON | `.json` — records or columns orientation |

---

## Supported tasks

| Task | Description |
|---|---|
| **Regression** | Continuous numeric target |
| **Classification** | Binary or multi-class label |
| **Ordinal Classification** | Ordered categories (e.g. pain scale 1–5) |

---

## Visualisation style

All charts follow **Storytelling with Data** principles:

- Non-message data → light gray (`#BFBFBF`)
- Key message → accent red (`#E63946`)
- Secondary comparison → blue (`#457B9D`)
- No gridlines; every chart carries a plain-language annotation

Colors are enforced globally via `shared/viz.py` — `StoryPalette` is the single
source of truth.
