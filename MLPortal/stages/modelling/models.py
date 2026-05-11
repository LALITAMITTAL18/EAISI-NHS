"""Pydantic models for the Modelling stage — registry, configs and results."""

from __future__ import annotations

import importlib
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ── Optuna param spec ─────────────────────────────────────────────────────────


class OptunaParamSpec(BaseModel):
    """Specification for a single hyperparameter in Optuna's search space."""

    type: Literal["int", "float", "categorical"]
    low: float | None = None
    high: float | None = None
    log: bool = False
    step: int | None = None
    choices: list[Any] | None = None

    @model_validator(mode="after")
    def _validate(self) -> "OptunaParamSpec":
        if self.type in ("int", "float"):
            if self.low is None or self.high is None:
                raise ValueError("int/float params require 'low' and 'high'")
        if self.type == "categorical" and not self.choices:
            raise ValueError("categorical params require 'choices'")
        return self


# ── Model spec ────────────────────────────────────────────────────────────────


class ModelSpec(BaseModel):
    """Full specification for a single ML model loaded from models.json."""

    name: str
    display_name: str
    task: list[Literal["regression", "classification", "ordinal"]]
    constructor_path: str
    default_params: dict[str, Any] = Field(default_factory=dict)
    optuna_space: dict[str, OptunaParamSpec] = Field(default_factory=dict)
    finetune_window: float = 0.2
    requires_scaling: bool = False
    supports_warmstart: bool = False
    tags: list[str] = Field(default_factory=list)
    available: bool = True  # set to False if the package cannot be imported

    def check_availability(self) -> bool:
        """Return True if the model's constructor can be imported."""
        try:
            module_path, class_name = self.constructor_path.rsplit(".", 1)
            mod = importlib.import_module(module_path)
            getattr(mod, class_name)
            return True
        except (ImportError, AttributeError):
            return False

    def build(self, **override_params: Any) -> Any:
        """Instantiate the model with merged default + override params."""
        module_path, class_name = self.constructor_path.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        params = {**self.default_params, **override_params}
        return cls(**params)

    def narrow_space(self, best_params: dict[str, Any]) -> dict[str, OptunaParamSpec]:
        """Return a narrowed Optuna search space around *best_params*."""
        narrowed: dict[str, OptunaParamSpec] = {}
        for param, spec in self.optuna_space.items():
            if param not in best_params or spec.type == "categorical":
                narrowed[param] = spec
                continue
            best_val = best_params[param]
            low, high = spec.low, spec.high
            if spec.log and best_val > 0 and low and high:
                log_best = math.log(best_val)
                log_range = math.log(high / low) * self.finetune_window
                new_low = math.exp(max(math.log(low), log_best - log_range))
                new_high = math.exp(min(math.log(high), log_best + log_range))
            else:
                rng = (high - low) * self.finetune_window  # type: ignore[operator]
                new_low = max(low, best_val - rng)  # type: ignore[operator]
                new_high = min(high, best_val + rng)  # type: ignore[operator]
            if spec.type == "int":
                new_low, new_high = int(new_low), max(int(new_high), int(new_low) + 1)
            narrowed[param] = spec.model_copy(update={"low": new_low, "high": new_high})
        return narrowed


# ── Registry ──────────────────────────────────────────────────────────────────


class ModelRegistry:
    """In-memory registry of all available ModelSpecs."""

    def __init__(self) -> None:
        self._registry: dict[str, ModelSpec] = {}

    def register(self, spec: ModelSpec) -> None:
        """Add or replace a model spec."""
        spec = spec.model_copy(update={"available": spec.check_availability()})
        self._registry[spec.name] = spec

    def get(self, name: str) -> ModelSpec:
        if name not in self._registry:
            raise KeyError(f"Model '{name}' not found in registry.")
        return self._registry[name]

    def list_for_task(
        self, task: Literal["regression", "classification", "ordinal"]
    ) -> list[ModelSpec]:
        """Return all available specs that support *task*."""
        return [
            s for s in self._registry.values()
            if task in s.task and s.available
        ]

    def all_specs(self) -> list[ModelSpec]:
        return list(self._registry.values())

    @classmethod
    def from_json(cls, *paths: Any) -> "ModelRegistry":
        """Load specs from one or more JSON files (later files override earlier ones)."""
        import json
        from pathlib import Path

        registry = cls()
        for path in paths:
            p = Path(path)
            if not p.exists():
                continue
            raw: list[dict[str, Any]] = json.loads(p.read_text(encoding="utf-8"))
            for item in raw:
                # Convert optuna_space dicts to OptunaParamSpec
                space = {
                    k: OptunaParamSpec(**v)
                    for k, v in item.get("optuna_space", {}).items()
                }
                spec = ModelSpec(**{**item, "optuna_space": space})
                registry.register(spec)
        return registry


# ── Training config & result ──────────────────────────────────────────────────


class OptunaConfig(BaseModel):
    """Optuna hyperparameter optimisation settings."""

    n_trials: int = 40
    cv_folds: int = 5
    timeout_seconds: int | None = None
    metric: str = "neg_root_mean_squared_error"
    sampler: Literal["tpe", "cmaes", "random"] = "tpe"
    random_seed: int = 42


class TrainConfig(BaseModel):
    """Full modelling configuration."""

    task: Literal["regression", "classification", "ordinal"]
    target_column: str
    selected_models: list[str]
    optuna: OptunaConfig = Field(default_factory=OptunaConfig)
    outcome_threshold: float | None = None
    class_imbalance: Literal["balanced", "smote", "none"] = "balanced"
    random_seed: int = 42


class TrainResult(BaseModel):
    """Serializable result from training and evaluating a single model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str
    task: str
    best_params: dict[str, Any]
    optuna_history: list[dict[str, Any]]
    # Metrics (populated after evaluation)
    cv_score: float | None = None
    test_metrics: dict[str, float] = Field(default_factory=dict)
    # Paths to persisted artifacts
    pipeline_path: str | None = None
    study_path: str | None = None
    # Pipeline object (not JSON-serialized — kept in memory only)
    pipeline: Any | None = Field(default=None, exclude=True)
