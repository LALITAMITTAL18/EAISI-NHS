"""Pydantic models for the Clinical Insight stage."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AzureMLConfig(BaseModel):
    subscription_id: str
    resource_group: str
    workspace_name: str
    tenant_id: str = "17b35a1d-057c-4ac5-a15a-08758f7a7064"


class EndpointInfo(BaseModel):
    name: str
    scoring_uri: str | None = None
    provisioning_state: str = "Unknown"
    traffic: dict[str, int] = Field(default_factory=dict)
    deployment_names: list[str] = Field(default_factory=list)


class WorkspaceModel(BaseModel):
    name: str
    version: str
    description: str = ""
    tags: dict[str, str] = Field(default_factory=dict)
    type: str = ""


class ShapResult(BaseModel):
    model_name: str
    feature_names: list[str]
    mean_abs_shap: list[float]
    shap_values: list[list[float]] = Field(default_factory=list)
    sample_features: list[dict[str, Any]] = Field(default_factory=list)
    base_value: float = 0.0


class TabularPredictionResult(BaseModel):
    prediction: float
    model_name: str
    endpoint_used: str
    feature_values: dict[str, Any] = Field(default_factory=dict)


class XrayPredictionResult(BaseModel):
    kl_grade: int
    confidence: float
    class_probabilities: list[float]
    class_names: list[str] = Field(default_factory=lambda: ["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4"])
    endpoint_used: str = ""


class ClinicalInsightResult(BaseModel):
    rf_result: TabularPredictionResult | None = None
    xray_result: XrayPredictionResult | None = None
    shap_result: ShapResult | None = None
    llm_advice: str | None = None


KL_DESCRIPTIONS = {
    0: "Normal — No radiographic features of OA",
    1: "Doubtful — Doubtful joint space narrowing; possible osteophytic lipping",
    2: "Mild — Definite osteophytes; possible joint space narrowing",
    3: "Moderate — Multiple osteophytes; definite joint space narrowing; some sclerosis",
    4: "Severe — Large osteophytes; marked joint space narrowing; severe sclerosis; definite deformity",
}

KL_SURGERY_GUIDANCE = {
    0: "Conservative management typically appropriate",
    1: "Conservative management; monitor progression",
    2: "Consider conservative management; evaluate symptoms carefully",
    3: "Surgical intervention often indicated; weigh against functional outcomes",
    4: "Surgery strongly indicated when functional outcomes are favourable",
}
