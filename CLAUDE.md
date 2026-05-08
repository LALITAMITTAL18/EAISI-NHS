# EAISI-NHS Project Guidelines

## Code Style & Formatting
- Python: PEP 8 (black formatter, 88 char line length)
- Use type hints for function signatures
- Imports: stdlib → third-party → local (organized, one per line for clarity)
- No commented-out code; use git history instead

## Notebooks
- Name by sequence: `1-exploratory.ipynb`, `2-preprocessing.ipynb`, etc.
- Clear cell structure: imports → config → functions → execution
- Remove outputs before committing; document key findings in markdown
- One logical analysis per notebook

## ML Conventions
- Store trained models in `models/` directory
- Data: raw in `data/raw/`, processed in `data/processed/`
- Experiments tracked in `notebooks/Experiment/` with domain subfolders
- Document model versions, hyperparameters, and evaluation metrics
- Use logging instead of print() in production code

## Git Workflow
- Feature branches for new analyses/models
- Commit messages: concise, imperative ("Add feature" not "Added")
- Include rationale in PR descriptions
- Clean up merged branches

## Reproducibility
- Document dependencies in `requirements.txt`
- Set random seeds for ML experiments
- Include train/test split strategy
- Record preprocessing steps and data assumptions
