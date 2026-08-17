"""Minimal FastAPI application around processed football data and xG artifacts.

The process does not train. Models are loaded from ``artifacts/``; processed
tables are read from Parquet via DuckDB. Missing data or artifacts produce 503
responses rather than implicit retraining.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Annotated

import duckdb
import numpy as np
import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from sklearn.pipeline import Pipeline

from football_intelligence import __version__
from football_intelligence.api.schemas import (
    ConversionRow,
    DatasetSummary,
    DerivedShotFeatures,
    HealthResponse,
    MatchListResponse,
    MatchShot,
    MatchShotsResponse,
    MatchSummary,
    ModelInfo,
    ModelListResponse,
    ModelMetrics,
    NumericSummary,
    ShotPredictionRequest,
    ShotPredictionResponse,
    StatisticalFinding,
    StatisticsSummary,
)
from football_intelligence.data import queries, storage
from football_intelligence.data.storage import StorageError
from football_intelligence.features import shots as shot_features
from football_intelligence.models import artifacts
from football_intelligence.models.artifacts import ArtifactError
from football_intelligence.models.logistic import feature_columns, predict_probability
from football_intelligence.statistics.diagnostics import SampleDiagnostics, describe_sample
from football_intelligence.statistics.tests import (
    StatisticalTestError,
    chi_square_independence_test,
    welch_t_test,
)

logger = logging.getLogger(__name__)

LOCAL_FRONTEND_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)

DEFAULT_MODEL_NAME = "logistic_baseline"
MAX_PAGE_SIZE = 100


def cors_origins() -> list[str]:
    """Local Vite origins plus any extra hosts from ``FOOTBALL_CORS_ORIGINS``."""
    origins = list(LOCAL_FRONTEND_ORIGINS)
    extra = os.environ.get("FOOTBALL_CORS_ORIGINS", "")
    for item in extra.split(","):
        origin = item.strip()
        if origin and origin not in origins:
            origins.append(origin)
    return origins


@dataclass
class AppContext:
    processed_root: Path
    artifacts_root: Path
    pipelines: dict[str, Pipeline] = field(default_factory=dict)


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


def create_app(
    *,
    processed_root: Path | None = None,
    artifacts_root: Path | None = None,
) -> FastAPI:
    """Build the API. Paths default to the project's local data layout."""
    context = AppContext(
        processed_root=processed_root
        if processed_root is not None
        else _env_path("FOOTBALL_PROCESSED_ROOT", storage.DEFAULT_PROCESSED_ROOT),
        artifacts_root=artifacts_root
        if artifacts_root is not None
        else _env_path("FOOTBALL_ARTIFACTS_ROOT", artifacts.DEFAULT_ARTIFACTS_ROOT),
    )
    application = FastAPI(
        title="Football Intelligence Lab",
        version=__version__,
        description="Analytical API over StatsBomb-derived tables and persisted xG models.",
    )
    application.state.context = context
    application.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(_build_router())

    available = artifacts.list_artifacts(context.artifacts_root)
    logger.info(
        "API ready (processed=%s, artifacts=%s, models=%s). No training on startup.",
        context.processed_root,
        context.artifacts_root,
        available or "none",
    )
    return application


def _context(request: Request) -> AppContext:
    loaded = getattr(request.app.state, "context", None)
    if not isinstance(loaded, AppContext):
        raise RuntimeError("Application context is missing.")
    return loaded


def _build_router() -> APIRouter:
    router = APIRouter()

    @router.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @router.get("/api/summary", response_model=DatasetSummary)
    def dataset_summary(request: Request) -> DatasetSummary:
        return _dataset_summary(_context(request))

    @router.get("/api/models", response_model=ModelListResponse)
    def list_models(request: Request) -> ModelListResponse:
        return _list_models(_context(request))

    @router.post("/api/predict/shot", response_model=ShotPredictionResponse)
    def predict_shot(payload: ShotPredictionRequest, request: Request) -> ShotPredictionResponse:
        return _predict_shot(_context(request), payload)

    @router.get("/api/statistics/summary", response_model=StatisticsSummary)
    def statistics_summary(request: Request) -> StatisticsSummary:
        return _statistics_summary(_context(request))

    @router.get("/api/matches", response_model=MatchListResponse)
    def list_matches(
        request: Request,
        limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 25,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> MatchListResponse:
        return _list_matches(_context(request), limit=limit, offset=offset)

    @router.get("/api/matches/{match_id}/shots", response_model=MatchShotsResponse)
    def match_shots(
        match_id: int,
        request: Request,
        model: str | None = None,
    ) -> MatchShotsResponse:
        return _match_shots(_context(request), match_id, model_name=model)

    return router


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #


def _connect(context: AppContext) -> duckdb.DuckDBPyConnection:
    try:
        return queries.connect(context.processed_root)
    except StorageError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


def _shot_frame(context: AppContext) -> pd.DataFrame:
    """Canonical shot dataset, built in memory if the parquet file is absent."""
    dataset_path = context.processed_root / shot_features.DEFAULT_DATASET_PATH.name
    try:
        if dataset_path.exists():
            return shot_features.read_shot_dataset(dataset_path)
        shots = storage.read_table("shots", context.processed_root)
        matches = storage.read_table("matches", context.processed_root)
    except (StorageError, shot_features.ShotFeatureError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return shot_features.build_shot_features(shots, matches)


def _require_pipeline(context: AppContext, name: str | None) -> tuple[str, Pipeline]:
    available = artifacts.list_artifacts(context.artifacts_root)
    if not available:
        raise HTTPException(
            status_code=503,
            detail="No trained model artifacts found. Train with `make train` first.",
        )
    chosen = name or (DEFAULT_MODEL_NAME if DEFAULT_MODEL_NAME in available else available[0])
    if chosen not in available:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model {chosen!r}. Available: {available}.",
        )
    if chosen not in context.pipelines:
        try:
            context.pipelines[chosen] = artifacts.load_pipeline(chosen, context.artifacts_root)
        except ArtifactError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        logger.info("loaded model artifact %s", chosen)
    return chosen, context.pipelines[chosen]


# --------------------------------------------------------------------------- #
# Endpoint implementations
# --------------------------------------------------------------------------- #


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return int(value)
    raise TypeError(f"Expected a number, got {type(value).__name__}")


def _as_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, np.integer | np.floating):
        return float(value)
    raise TypeError(f"Expected a number, got {type(value).__name__}")


def _as_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    converted = pd.to_datetime(str(value))
    return converted.to_pydatetime().date()


def _dataset_summary(context: AppContext) -> DatasetSummary:
    with _connect(context) as connection:
        row = connection.execute(
            """
            SELECT
                (SELECT count(*) FROM matches) AS matches,
                (SELECT count(*) FROM players) AS players,
                count(*) AS shots,
                count(*) FILTER (WHERE is_goal) AS goals,
                count(*) FILTER (WHERE is_goal)::DOUBLE
                    / NULLIF(count(*), 0) AS goal_rate
            FROM shots
            WHERE NOT is_shootout
            """
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=503, detail="Could not summarise the processed tables.")
    matches, players, shots, goals, goal_rate = row
    return DatasetSummary(
        matches=int(matches),
        players=int(players),
        shots=int(shots),
        goals=int(goals),
        goal_rate=float(goal_rate) if goal_rate is not None else 0.0,
    )


def _list_models(context: AppContext) -> ModelListResponse:
    models: list[ModelInfo] = []
    for name in artifacts.list_artifacts(context.artifacts_root):
        try:
            info = artifacts.describe_artifact(name, context.artifacts_root)
        except ArtifactError as error:
            logger.warning("skipping unreadable artifact %s: %s", name, error)
            continue
        try:
            metrics = ModelMetrics.model_validate(info.metrics)
        except ValidationError as error:
            logger.warning("skipping artifact %s with invalid metrics: %s", name, error)
            continue
        models.append(
            ModelInfo(
                name=info.name,
                task=info.task,
                features=list(info.features),
                training_samples=info.training_samples,
                test_samples=info.test_samples,
                validation_strategy=info.validation_strategy,
                metrics=metrics,
            )
        )
    return ModelListResponse(models=models)


def _predict_shot(context: AppContext, payload: ShotPredictionRequest) -> ShotPredictionResponse:
    on_pitch = bool(
        np.asarray(shot_features.is_valid_location(payload.x, payload.y)).reshape(-1)[0]
    )
    if not on_pitch:
        raise HTTPException(
            status_code=422,
            detail=f"Shot coordinates ({payload.x}, {payload.y}) are outside the 120x80 pitch.",
        )

    name, pipeline = _require_pipeline(context, payload.model)
    distance = float(np.asarray(shot_features.shot_distance(payload.x, payload.y)).reshape(-1)[0])
    angle = float(np.asarray(shot_features.shot_angle(payload.x, payload.y)).reshape(-1)[0])
    row: dict[str, object] = {
        "x": payload.x,
        "y": payload.y,
        "shot_distance": distance,
        "shot_angle": angle,
        "body_part": payload.body_part,
        "shot_type": payload.shot_type,
        "technique": payload.technique,
        "under_pressure": payload.under_pressure,
        "first_time": payload.first_time,
    }
    required = feature_columns(pipeline)
    missing = [
        column
        for column in required
        if column not in ("shot_distance", "shot_angle") and row.get(column) is None
    ]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Model {name!r} requires {missing}; they were not supplied.",
        )

    frame = pd.DataFrame([row])
    probability = float(predict_probability(pipeline, frame)[0])
    return ShotPredictionResponse(
        model=name,
        predicted_xg=probability,
        features=DerivedShotFeatures(
            x=payload.x,
            y=payload.y,
            shot_distance=distance,
            shot_angle=angle,
            shot_angle_degrees=float(np.degrees(angle)),
            body_part=payload.body_part,
            shot_type=payload.shot_type,
            technique=payload.technique,
            under_pressure=payload.under_pressure,
            first_time=payload.first_time,
        ),
    )


def _numeric_summary(diagnostics: SampleDiagnostics) -> NumericSummary:
    return NumericSummary(
        name=diagnostics.name,
        n=diagnostics.n_observed,
        mean=diagnostics.mean,
        median=diagnostics.median,
        std=diagnostics.std,
        q1=diagnostics.q1,
        q3=diagnostics.q3,
        iqr=diagnostics.iqr,
        skewness=diagnostics.skewness,
    )


def _statistics_summary(context: AppContext) -> StatisticsSummary:
    dataset = _shot_frame(context)
    goals = int(dataset["goal"].sum())
    distance = describe_sample(dataset["shot_distance"], name="shot_distance (yards)")
    angle = describe_sample(np.degrees(dataset["shot_angle"]), name="shot_angle (degrees)")

    body_part = (
        dataset.groupby("body_part", observed=True)
        .agg(shots=("goal", "size"), goals=("goal", "sum"))
        .reset_index()
        .sort_values("shots", ascending=False)
    )
    conversion = [
        ConversionRow(
            group=str(record["body_part"]),
            shots=_as_int(record["shots"]),
            goals=_as_int(record["goals"]),
            conversion_rate=(
                _as_float(record["goals"]) / _as_int(record["shots"])
                if _as_int(record["shots"])
                else 0.0
            ),
        )
        for record in body_part.to_dict(orient="records")
    ]

    findings: list[StatisticalFinding] = []
    open_play = dataset[dataset["shot_type"] == "Open Play"]
    team_counts = open_play["team"].value_counts()
    if len(team_counts) >= 2 and int(team_counts.iloc[0]) >= 2 and int(team_counts.iloc[1]) >= 2:
        team_a, team_b = (str(name) for name in team_counts.index[:2])
        try:
            result = welch_t_test(
                open_play.loc[open_play["team"] == team_a, "shot_distance"],
                open_play.loc[open_play["team"] == team_b, "shot_distance"],
                label_a=team_a,
                label_b=team_b,
            )
        except StatisticalTestError as error:
            logger.info("skipping team distance comparison: %s", error)
        else:
            findings.append(
                StatisticalFinding(
                    name="open_play_shot_distance_by_team",
                    question=(
                        f"Do {team_a} and {team_b} (the two teams with the most open-play "
                        "shots) shoot from different distances on average?"
                    ),
                    test_name=result.test_name,
                    statistic=result.statistic,
                    p_value=result.p_value,
                    n_total=result.n_total,
                    n_a=result.n_a,
                    n_b=result.n_b,
                    estimate=result.estimate,
                    estimate_name=result.estimate_name,
                    confidence_interval=result.confidence_interval,
                    warnings=list(result.warnings),
                    notes=list(result.notes),
                )
            )

    counted = dataset["body_part"].value_counts()
    kept = [str(part) for part in counted[counted >= 5].index]
    if len(kept) >= 2:
        subset = dataset[dataset["body_part"].isin(kept)]
        table = pd.crosstab(subset["body_part"], subset["goal"])
        try:
            result = chi_square_independence_test(table)
        except StatisticalTestError as error:
            logger.info("skipping body-part conversion test: %s", error)
        else:
            findings.append(
                StatisticalFinding(
                    name="goal_by_body_part",
                    question="Is scoring associated with the body part used?",
                    test_name=result.test_name,
                    statistic=result.statistic,
                    p_value=result.p_value,
                    n_total=result.n_total,
                    warnings=list(result.warnings),
                    notes=list(result.notes),
                )
            )

    return StatisticsSummary(
        shots=len(dataset),
        goals=goals,
        goal_rate=float(dataset["goal"].mean()) if len(dataset) else 0.0,
        shot_distance=_numeric_summary(distance),
        shot_angle_degrees=_numeric_summary(angle),
        conversion_by_body_part=conversion,
        findings=findings,
    )


def _list_matches(context: AppContext, *, limit: int, offset: int) -> MatchListResponse:
    with _connect(context) as connection:
        total_row = connection.execute("SELECT count(*) FROM matches").fetchone()
        total = int(total_row[0]) if total_row is not None else 0
        frame = connection.execute(
            """
            SELECT
                match_id,
                match_date,
                competition_name,
                season_name,
                stage,
                home_team,
                away_team,
                home_score,
                away_score
            FROM matches
            ORDER BY match_date, match_id
            LIMIT ? OFFSET ?
            """,
            [limit, offset],
        ).df()
    matches = [
        MatchSummary(
            match_id=_as_int(record["match_id"]),
            match_date=_as_date(record["match_date"]),
            competition_name=str(record["competition_name"]),
            season_name=str(record["season_name"]),
            stage=None if pd.isna(record["stage"]) else str(record["stage"]),
            home_team=str(record["home_team"]),
            away_team=str(record["away_team"]),
            home_score=_as_int(record["home_score"]),
            away_score=_as_int(record["away_score"]),
        )
        for record in frame.to_dict(orient="records")
    ]
    return MatchListResponse(total=total, limit=limit, offset=offset, matches=matches)


def _match_row(context: AppContext, match_id: int) -> MatchSummary:
    with _connect(context) as connection:
        frame = connection.execute(
            """
            SELECT
                match_id,
                match_date,
                competition_name,
                season_name,
                stage,
                home_team,
                away_team,
                home_score,
                away_score
            FROM matches
            WHERE match_id = ?
            """,
            [match_id],
        ).df()
    if frame.empty:
        raise HTTPException(status_code=404, detail=f"Match {match_id} was not found.")
    row = frame.iloc[0]
    return MatchSummary(
        match_id=_as_int(row["match_id"]),
        match_date=_as_date(row["match_date"]),
        competition_name=str(row["competition_name"]),
        season_name=str(row["season_name"]),
        stage=None if pd.isna(row["stage"]) else str(row["stage"]),
        home_team=str(row["home_team"]),
        away_team=str(row["away_team"]),
        home_score=_as_int(row["home_score"]),
        away_score=_as_int(row["away_score"]),
    )


def _match_shots(
    context: AppContext, match_id: int, *, model_name: str | None
) -> MatchShotsResponse:
    match = _match_row(context, match_id)
    dataset = _shot_frame(context)
    shots = dataset[dataset["match_id"] == match_id].copy()

    predicted: np.ndarray | None = None
    chosen: str | None = None
    available = artifacts.list_artifacts(context.artifacts_root)
    if available:
        chosen, pipeline = _require_pipeline(context, model_name)
        if not shots.empty:
            predicted = predict_probability(pipeline, shots)
    elif model_name:
        raise HTTPException(
            status_code=503,
            detail="No trained model artifacts found. Train with `make train` first.",
        )

    outcome_by_id: dict[str, str] = {}
    try:
        raw_shots = storage.read_table("shots", context.processed_root)
    except StorageError:
        raw_shots = None
    if raw_shots is not None and "outcome" in raw_shots.columns:
        subset = raw_shots[raw_shots["match_id"] == match_id]
        outcome_by_id = {
            str(event_id): str(outcome)
            for event_id, outcome in zip(subset["event_id"], subset["outcome"], strict=True)
            if pd.notna(outcome)
        }

    rows: list[MatchShot] = []
    for index, (_, shot) in enumerate(shots.iterrows()):
        statsbomb = shot.get("statsbomb_xg")
        rows.append(
            MatchShot(
                shot_id=str(shot["shot_id"]),
                period=int(shot["period"]),
                minute=int(shot["minute"]),
                team=str(shot["team"]),
                player=str(shot["player"]),
                x=float(shot["x"]),
                y=float(shot["y"]),
                shot_distance=float(shot["shot_distance"]),
                shot_angle=float(shot["shot_angle"]),
                goal=bool(shot["goal"]),
                outcome=outcome_by_id.get(str(shot["shot_id"])),
                statsbomb_xg=None if pd.isna(statsbomb) else float(statsbomb),
                predicted_xg=None if predicted is None else float(predicted[index]),
            )
        )
    return MatchShotsResponse(
        match_id=match.match_id,
        match_date=match.match_date,
        home_team=match.home_team,
        away_team=match.away_team,
        home_score=match.home_score,
        away_score=match.away_score,
        model=chosen,
        shots=rows,
    )


app = create_app()
