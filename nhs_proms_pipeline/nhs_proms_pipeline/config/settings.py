"""Configuration and settings for the NHS PROMs ML pipeline.

Uses pydantic-settings so every value can be overridden via environment
variables or a .env file without touching source code.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProcedureType(str, Enum):
    """Supported surgical procedure types."""

    KNEE = "KNEE"
    HIP = "HIP"


class LogLevel(str, Enum):
    """Allowed logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class ProcedureConfig:
    """Static mapping of procedure-specific column names and thresholds.

    All column name constants live here so they are never hard-coded in
    business logic modules.
    """

    _CONFIGS: dict[ProcedureType, dict] = {
        ProcedureType.KNEE: {
            "raw_csv_name": "Knee Replacement Provider.csv",
            "score_prefix": "Knee Replacement",
            "pre_op_score_col": "Knee Replacement Pre-Op Q Score",
            "post_op_score_col": "Knee Replacement Post-Op Q Score",
            "pre_op_q_prefix": "Knee Replacement Pre-Op Q",
            "mcid_threshold": 5.0,
            "keep_post_op_cols": [
                "Post-Op Q EQ5D Index Profile",
                "Post-Op Q EQ5D Index",
                "Post-Op Q EQ VAS",
                "Knee Replacement Post-Op Q Score",
            ],
            "predicted_col_substrings": [
                "Knee Replacement EQ 5D Index Post-Op Q Predicted",
                "Knee Replacement EQ VAS_Post-Op Q Predicted",
                "Knee Replacement OKS Post-Op Q Predicted",
            ],
        },
        ProcedureType.HIP: {
            "raw_csv_name": "Hip Replacement Provider.csv",
            "score_prefix": "Hip Replacement",
            "pre_op_score_col": "Hip Replacement Pre-Op Q Score",
            "post_op_score_col": "Hip Replacement Post-Op Q Score",
            "pre_op_q_prefix": "Hip Replacement Pre-Op Q",
            "mcid_threshold": 6.0,
            "keep_post_op_cols": [
                "Post-Op Q EQ5D Index Profile",
                "Post-Op Q EQ5D Index",
                "Post-Op Q EQ VAS",
                "Hip Replacement Post-Op Q Score",
            ],
            "predicted_col_substrings": [
                "Hip Replacement EQ5D Index Post-Op Q Predicted",
                "Hip Replacement EQ VAS_Post-Op Q Predicted",
                "Hip Replacement OHS Post-Op Q Predicted",
            ],
        },
    }

    def __init__(self, procedure: ProcedureType) -> None:
        cfg = self._CONFIGS[procedure]
        self.raw_csv_name: str = cfg["raw_csv_name"]
        self.score_prefix: str = cfg["score_prefix"]
        self.pre_op_score_col: str = cfg["pre_op_score_col"]
        self.post_op_score_col: str = cfg["post_op_score_col"]
        self.pre_op_q_prefix: str = cfg["pre_op_q_prefix"]
        self.mcid_threshold: float = cfg["mcid_threshold"]
        self.keep_post_op_cols: list[str] = cfg["keep_post_op_cols"]
        self.predicted_col_substrings: list[str] = cfg["predicted_col_substrings"]


# ── Indicator columns (same for both procedures) ─────────────────────────────
COMORBIDITY_INDICATOR_COLS: list[str] = [
    "Arthritis",
    "Cancer",
    "Circulation",
    "Depression",
    "Diabetes",
    "Heart Disease",
    "High Bp",
    "Kidney Disease",
    "Liver Disease",
    "Lung Disease",
    "Nervous System",
    "Stroke",
]

# ── EQ-5D columns (same for both procedures) ─────────────────────────────────
EQ5D_PRE_OP_COLS: list[str] = [
    "Pre-Op Q Mobility",
    "Pre-Op Q Self-Care",
    "Pre-Op Q Activity",
    "Pre-Op Q Discomfort",
    "Pre-Op Q Anxiety",
]

# Columns to be mode-imputed (post train/test split)
MODE_IMPUTE_COLS: list[str] = [
    "Pre-Op Q Assisted",
    "Pre-Op Q Previous Surgery",
]

# Columns to fill with a constant sentinel (post train/test split)
CONSTANT_FILL_COLS: dict[str, int] = {
    "Pre-Op Q Living Arrangements": 9,  # 9 = Unknown
    "Pre-Op Q Disability": 9,  # 9 = Not Disclosed
}

# Columns that must not leak post-op info into features
VAS_DROP_COLS: list[str] = [
    "Pre-Op Q EQ VAS",
    "Post-Op Q EQ VAS",
    "Post-Op Q EQ5D Index",
]

# Categorical columns to label-encode (ordinal or nominal)
COLS_TO_LABEL_ENCODE: list[str] = [
    "Procedure",
    "Year",
    "Age Band",
    "Provider Code",
]

# Numeric "missing" sentinels that should become null
NUMERIC_MISSING_SENTINELS: list[int] = [9, 999]

# String "missing" sentinels that should become null
STRING_MISSING_SENTINELS: list[str] = ["*", ""]


class PipelineSettings(BaseSettings):
    """Global pipeline settings.

    All values configurable via environment variables or .env file.
    Environment variable names match field names (case-insensitive).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Procedure ─────────────────────────────────────────────────────────────
    procedure_type: ProcedureType = Field(
        default=ProcedureType.KNEE,
        description="Surgical procedure type (KNEE or HIP).",
    )

    # ── Data paths ─────────────────────────────────────────────────────────────
    raw_data_dir: Path = Field(
        default=Path("../data/external"),
        description="Directory containing raw NHS PROMs CSV files.",
    )
    interim_data_dir: Path = Field(
        default=Path("./data/interim"),
        description="Directory for intermediate parquet files.",
    )
    models_dir: Path = Field(
        default=Path("./models"),
        description="Directory where trained models are persisted.",
    )
    reports_dir: Path = Field(
        default=Path("./reports"),
        description="Directory for evaluation reports and charts.",
    )

    # ── Reproducibility ────────────────────────────────────────────────────────
    random_seed: int = Field(default=42, ge=0, description="Random seed for reproducibility.")

    # ── Train / test split ─────────────────────────────────────────────────────
    train_test_ratio: float = Field(
        default=0.8,
        gt=0.0,
        lt=1.0,
        description="Fraction of data used for training.",
    )

    # ── Cross-validation ────────────────────────────────────────────────────────
    cv_splits: int = Field(default=5, ge=2, description="Number of KFold splits.")
    cv_repeats: int = Field(default=3, ge=1, description="Number of KFold repeats.")

    # ── Logging ────────────────────────────────────────────────────────────────
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging verbosity.")

    # ── Models to train ────────────────────────────────────────────────────────
    enabled_models: list[
        Literal[
            "LinearRegression",
            "Ridge",
            "Lasso",
            "RandomForest",
            "GradientBoosting",
            "XGBoost",
        ]
    ] = Field(
        default=[
            "LinearRegression",
            "Ridge",
            "Lasso",
            "RandomForest",
            "GradientBoosting",
            "XGBoost",
        ],
        description="Models to include in the training run.",
    )

    @field_validator("raw_data_dir", "interim_data_dir", "models_dir", "reports_dir", mode="before")
    @classmethod
    def _coerce_path(cls, v: object) -> Path:
        return Path(str(v))

    @model_validator(mode="after")
    def _create_directories(self) -> "PipelineSettings":
        """Ensure output directories exist at settings load time."""
        for directory in (self.interim_data_dir, self.models_dir, self.reports_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    def get_procedure_config(self) -> ProcedureConfig:
        """Return the procedure-specific column name mapping."""
        return ProcedureConfig(self.procedure_type)

    def interim_path(self, filename: str) -> Path:
        """Resolve a filename relative to the interim data directory."""
        return self.interim_data_dir / filename

    def model_path(self, filename: str) -> Path:
        """Resolve a filename relative to the models directory."""
        return self.models_dir / filename

    def raw_csv_path(self) -> Path:
        """Resolve the raw input CSV for the configured procedure."""
        proc_cfg = self.get_procedure_config()
        return self.raw_data_dir / proc_cfg.raw_csv_name
