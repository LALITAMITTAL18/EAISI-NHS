"""Model registry loader and Optuna-based trainer/optimizer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import CmaEsSampler, RandomSampler, TPESampler
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from stages.modelling.models import (
    ModelRegistry,
    ModelSpec,
    OptunaConfig,
    TrainConfig,
    TrainResult,
)
from stages.preparation.encoder import build_column_transformer
from shared.io import save_joblib

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Registry loader ───────────────────────────────────────────────────────────


def load_registry(base_dir: Path) -> ModelRegistry:
    """Load built-in models.json and user custom_models.json into one registry."""
    data_dir = base_dir / "data"
    return ModelRegistry.from_json(
        data_dir / "models.json",
        data_dir / "custom_models.json",
    )


# ── Pipeline builder ──────────────────────────────────────────────────────────


def build_sklearn_pipeline(
    spec: ModelSpec,
    numeric_cols: list[str],
    categorical_cols: list[str],
    model_params: dict[str, Any],
    prep_config: Any,
) -> Pipeline:
    """Build a full sklearn Pipeline: ColumnTransformer → model."""
    from stages.preparation.models import PrepConfig

    preprocessor = build_column_transformer(
        numeric_cols, categorical_cols, prep_config
    )
    model = spec.build(**model_params)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


# ── Optuna helpers ────────────────────────────────────────────────────────────


def _suggest_params(
    trial: optuna.Trial,
    optuna_space: dict,
) -> dict[str, Any]:
    """Convert OptunaParamSpec dicts into actual Optuna trial suggestions."""
    params: dict[str, Any] = {}
    for name, spec in optuna_space.items():
        ptype = spec.type if hasattr(spec, "type") else spec["type"]
        if ptype == "int":
            low = int(spec.low if hasattr(spec, "low") else spec["low"])
            high = int(spec.high if hasattr(spec, "high") else spec["high"])
            step = int(spec.get("step") if isinstance(spec, dict) else getattr(spec, "step", None) or 1)
            params[name] = trial.suggest_int(name, low, high, step=step)
        elif ptype == "float":
            low = float(spec.low if hasattr(spec, "low") else spec["low"])
            high = float(spec.high if hasattr(spec, "high") else spec["high"])
            log = bool(spec.log if hasattr(spec, "log") else spec.get("log", False))
            params[name] = trial.suggest_float(name, low, high, log=log)
        elif ptype == "categorical":
            choices = spec.choices if hasattr(spec, "choices") else spec["choices"]
            params[name] = trial.suggest_categorical(name, choices)
    return params


def _get_sampler(sampler_name: str, seed: int) -> optuna.samplers.BaseSampler:
    if sampler_name == "cmaes":
        return CmaEsSampler(seed=seed)
    if sampler_name == "random":
        return RandomSampler(seed=seed)
    return TPESampler(seed=seed)


def _get_cv(task: str, folds: int, seed: int):
    if task in ("classification", "ordinal"):
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return KFold(n_splits=folds, shuffle=True, random_state=seed)


# ── Training ──────────────────────────────────────────────────────────────────


def run_optuna(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: OptunaConfig,
    task: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    prep_config: Any,
    init_params: dict[str, Any] | None = None,
) -> TrainResult:
    """Run Optuna HPO for a single model and return a TrainResult."""
    cv = _get_cv(task, config.cv_folds, config.random_seed)
    sampler = _get_sampler(config.sampler, config.random_seed)

    def objective(trial: optuna.Trial) -> float:
        suggested = _suggest_params(trial, spec.optuna_space)
        params = {**spec.default_params, **suggested}
        pipeline = build_sklearn_pipeline(
            spec, numeric_cols, categorical_cols, params, prep_config
        )
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=config.metric,
            n_jobs=-1,
            error_score="raise",
        )
        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )

    # Warm-start hint: enqueue the best known params first
    if init_params:
        filtered = {k: v for k, v in init_params.items() if k in spec.optuna_space}
        if filtered:
            study.enqueue_trial(filtered)

    study.optimize(
        objective,
        n_trials=config.n_trials,
        timeout=config.timeout_seconds,
        show_progress_bar=False,
    )

    best_params = {**spec.default_params, **study.best_params}
    best_pipeline = build_sklearn_pipeline(
        spec, numeric_cols, categorical_cols, best_params, prep_config
    )
    best_pipeline.fit(X_train, y_train)

    history = [
        {"trial": t.number, "value": t.value, "params": t.params}
        for t in study.trials
        if t.value is not None
    ]

    return TrainResult(
        model_name=spec.name,
        task=task,
        best_params=best_params,
        optuna_history=history,
        cv_score=study.best_value,
        pipeline=best_pipeline,
    )


def train_single(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: OptunaConfig,
    task: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    prep_config: Any,
) -> TrainResult:
    """Train a single model with Optuna HPO."""
    return run_optuna(
        spec=spec,
        X_train=X_train,
        y_train=y_train,
        config=config,
        task=task,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        prep_config=prep_config,
    )


def finetune_model(
    result: TrainResult,
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: OptunaConfig,
    task: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    prep_config: Any,
) -> TrainResult:
    """Fine-tune by narrowing the search space around the best known params."""
    narrowed_space = spec.narrow_space(result.best_params)
    finetuned_spec = spec.model_copy(update={"optuna_space": narrowed_space})
    return run_optuna(
        spec=finetuned_spec,
        X_train=X_train,
        y_train=y_train,
        config=config,
        task=task,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        prep_config=prep_config,
        init_params=result.best_params,
    )


def train_all(
    selected_names: list[str],
    registry: ModelRegistry,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: TrainConfig,
    numeric_cols: list[str],
    categorical_cols: list[str],
    prep_config: Any,
    progress_callback=None,
    output_dir: Path | None = None,
) -> list[TrainResult]:
    """Train all selected models sequentially; call progress_callback(name) after each.

    If *output_dir* is provided each model is persisted to disk immediately after
    it finishes so partial results survive if training is interrupted.
    """
    results: list[TrainResult] = []
    for name in selected_names:
        spec = registry.get(name)
        result = train_single(
            spec=spec,
            X_train=X_train,
            y_train=y_train,
            config=config.optuna,
            task=config.task,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            prep_config=prep_config,
        )
        results.append(result)
        # Save immediately so the model is on disk even if later models fail
        if output_dir is not None and result.pipeline is not None:
            path = output_dir / f"{result.model_name}.joblib"
            save_joblib(result.pipeline, path)
            result.pipeline_path = str(path)
        if progress_callback:
            progress_callback(name)
    return results


def persist_results(
    results: list[TrainResult],
    output_dir: Path,
) -> dict[str, str]:
    """Save each model's pipeline to disk; return {model_name: path}."""
    paths: dict[str, str] = {}
    for result in results:
        if result.pipeline is not None:
            path = output_dir / f"{result.model_name}.joblib"
            save_joblib(result.pipeline, path)
            result.pipeline_path = str(path)
            paths[result.model_name] = str(path)
    return paths
