"""Pydantic request and response models for the public API."""

from __future__ import annotations

import math
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from football_intelligence.features.shots import PITCH_LENGTH, PITCH_WIDTH


class HealthResponse(BaseModel):
    status: Literal["ok"]


class DatasetSummary(BaseModel):
    matches: int
    players: int
    shots: int
    goals: int
    goal_rate: float


class ModelMetrics(BaseModel):
    model_config = ConfigDict(extra="ignore")

    n: int
    positives: int
    prevalence: float
    roc_auc: float
    pr_auc: float
    log_loss: float
    log_loss_skill: float
    brier_score: float
    brier_skill: float
    mean_prediction: float
    calibration_in_the_large: float


class ModelInfo(BaseModel):
    name: str
    task: str
    features: list[str]
    training_samples: int
    test_samples: int
    validation_strategy: str
    metrics: ModelMetrics


class ModelListResponse(BaseModel):
    models: list[ModelInfo]


class ShotPredictionRequest(BaseModel):
    """Minimum shot description needed to score an xG model.

    Coordinates are StatsBomb yards, attacking toward ``x = 120``. Distance and
    angle are derived from ``x`` and ``y``; they are not accepted as inputs, so a
    caller cannot supply geometry that contradicts the location.
    """

    model: str | None = Field(
        default=None,
        description="Artifact name. Omitted: logistic_baseline, else the first available model.",
    )
    x: float = Field(..., ge=0.0, le=PITCH_LENGTH, description="Shot x in yards, 0-120.")
    y: float = Field(..., ge=0.0, le=PITCH_WIDTH, description="Shot y in yards, 0-80.")
    body_part: str | None = None
    shot_type: str | None = None
    technique: str | None = None
    under_pressure: bool = False
    first_time: bool = False

    @field_validator("x", "y")
    @classmethod
    def coordinates_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("must be a finite number")
        return value


class DerivedShotFeatures(BaseModel):
    x: float
    y: float
    shot_distance: float
    shot_angle: float
    shot_angle_degrees: float
    body_part: str | None = None
    shot_type: str | None = None
    technique: str | None = None
    under_pressure: bool
    first_time: bool


class ShotPredictionResponse(BaseModel):
    model: str
    predicted_xg: float
    features: DerivedShotFeatures


class NumericSummary(BaseModel):
    name: str
    n: int
    mean: float
    median: float
    std: float
    q1: float
    q3: float
    iqr: float
    skewness: float


class ConversionRow(BaseModel):
    group: str
    shots: int
    goals: int
    conversion_rate: float


class StatisticalFinding(BaseModel):
    name: str
    question: str
    test_name: str
    statistic: float
    p_value: float
    n_total: int
    n_a: int | None = None
    n_b: int | None = None
    estimate: float | None = None
    estimate_name: str | None = None
    confidence_interval: tuple[float, float] | None = None
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class StatisticsSummary(BaseModel):
    shots: int
    goals: int
    goal_rate: float
    shot_distance: NumericSummary
    shot_angle_degrees: NumericSummary
    conversion_by_body_part: list[ConversionRow]
    findings: list[StatisticalFinding]


class MatchSummary(BaseModel):
    match_id: int
    match_date: date
    competition_name: str
    season_name: str
    stage: str | None
    home_team: str
    away_team: str
    home_score: int
    away_score: int


class MatchListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    matches: list[MatchSummary]


class MatchShot(BaseModel):
    shot_id: str
    period: int
    minute: int
    team: str
    player: str
    x: float
    y: float
    shot_distance: float
    shot_angle: float
    goal: bool
    outcome: str | None = None
    statsbomb_xg: float | None = None
    predicted_xg: float | None = None


class MatchShotsResponse(BaseModel):
    match_id: int
    match_date: date
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    model: str | None
    shots: list[MatchShot]
