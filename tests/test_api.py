"""API behaviour tests. All data and models are built from fixtures, offline."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from football_intelligence.api.app import create_app
from football_intelligence.data import statsbomb, storage
from football_intelligence.evaluation.metrics import evaluate_probabilities
from football_intelligence.evaluation.validation import (
    assert_no_group_leakage,
    group_train_test_split,
)
from football_intelligence.features import shots as shot_features
from football_intelligence.models import artifacts
from football_intelligence.models.logistic import (
    TARGET,
    build_baseline_model,
    build_contextual_model,
    feature_columns,
    fit,
    predict_probability,
)

SEED = 20260817


def persist_logistic_models(frame: pd.DataFrame, artifacts_root: Path) -> list[str]:
    """Fit the two logistic models on a match-grouped split and write artifacts."""
    split = group_train_test_split(frame, test_size=0.25, random_state=SEED)
    assert_no_group_leakage(split)
    saved: list[str] = []
    for name, builder in (
        ("logistic_baseline", build_baseline_model),
        ("logistic_contextual", build_contextual_model),
    ):
        pipeline = fit(builder(random_state=SEED), split.train)
        metrics = evaluate_probabilities(
            split.test[TARGET].to_numpy(), predict_probability(pipeline, split.test)
        )
        artifacts.save_artifact(
            name,
            pipeline,
            metrics,
            {
                "features": feature_columns(pipeline),
                "training_samples": len(split.train),
                "test_samples": len(split.test),
                "validation_strategy": "group_by_match",
            },
            artifacts_root,
        )
        saved.append(name)
    return saved


def _synthetic_shots(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    distance = rng.uniform(2.0, 40.0, size=n)
    angle = np.arctan2(8.0 * np.maximum(1.0, 40.0 - 0.3 * distance), distance**2)
    logit = 1.4 - 0.20 * distance + 1.8 * angle
    probability = 1.0 / (1.0 + np.exp(-logit))
    return pd.DataFrame(
        {
            "match_id": np.repeat(np.arange(n // 20), 20),
            "shot_distance": distance,
            "shot_angle": angle,
            "body_part": rng.choice(["Right Foot", "Left Foot", "Head"], size=n),
            "shot_type": rng.choice(
                ["Open Play", "Free Kick", "Penalty"], size=n, p=[0.94, 0.04, 0.02]
            ),
            "technique": rng.choice(["Normal", "Volley", "Backheel"], size=n, p=[0.9, 0.09, 0.01]),
            "under_pressure": rng.random(n) < 0.22,
            "first_time": rng.random(n) < 0.26,
            "goal": rng.random(n) < probability,
        }
    )


@pytest.fixture
def processed_root(raw_root: Path, tmp_path: Path) -> Path:
    root = tmp_path / "processed"
    storage.build_processed_tables([statsbomb.CompetitionSeason(900, 1)], raw_root, root)
    shot_features.build_shot_dataset(root, root / "shot_dataset.parquet")
    return root


@pytest.fixture
def artifacts_root(tmp_path: Path) -> Path:
    root = tmp_path / "artifacts"
    persist_logistic_models(_synthetic_shots(), root)
    return root


@pytest.fixture
def client(processed_root: Path, artifacts_root: Path) -> Iterator[TestClient]:
    application = create_app(processed_root=processed_root, artifacts_root=artifacts_root)
    with TestClient(application) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_cors_allows_the_local_vite_origin(client: TestClient) -> None:
    response = client.get("/health", headers={"Origin": "http://localhost:5173"})

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_origins_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, processed_root: Path, artifacts_root: Path
) -> None:
    monkeypatch.setenv("FOOTBALL_CORS_ORIGINS", "http://localhost:9999")
    application = create_app(processed_root=processed_root, artifacts_root=artifacts_root)

    response = TestClient(application).get("/health", headers={"Origin": "http://localhost:9999"})

    assert response.headers["access-control-allow-origin"] == "http://localhost:9999"


def test_summary_uses_real_fixture_counts(client: TestClient) -> None:
    response = client.get("/api/summary")

    assert response.status_code == 200
    payload = response.json()
    # In-play shots only: 3 shots, 2 goals. Two matches, four unique players.
    assert payload["matches"] == 2
    assert payload["players"] == 4
    assert payload["shots"] == 3
    assert payload["goals"] == 2
    assert payload["goal_rate"] == pytest.approx(2 / 3)


def test_models_lists_persisted_artifacts_with_metrics(client: TestClient) -> None:
    response = client.get("/api/models")

    assert response.status_code == 200
    names = {model["name"] for model in response.json()["models"]}
    assert names == {"logistic_baseline", "logistic_contextual"}
    for model in response.json()["models"]:
        metrics = model["metrics"]
        assert 0.0 <= metrics["roc_auc"] <= 1.0
        assert 0.0 <= metrics["brier_score"] <= 1.0
        assert metrics["n"] > 0
        assert "shot_distance" in model["features"]
        assert "shot_angle" in model["features"]
        assert model["validation_strategy"] == "group_by_match"


def test_startup_does_not_train_or_write_artifacts(processed_root: Path, tmp_path: Path) -> None:
    empty = tmp_path / "empty-artifacts"
    empty.mkdir()

    create_app(processed_root=processed_root, artifacts_root=empty)

    assert artifacts.list_artifacts(empty) == []
    assert list(empty.iterdir()) == []


def test_predict_returns_xg_and_derived_geometry(client: TestClient) -> None:
    response = client.post("/api/predict/shot", json={"x": 108.0, "y": 40.0})

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "logistic_baseline"
    assert 0.0 <= payload["predicted_xg"] <= 1.0
    features = payload["features"]
    assert features["x"] == 108.0
    assert features["y"] == 40.0
    assert features["shot_distance"] == pytest.approx(12.0)
    assert features["shot_angle"] > 0
    assert features["shot_angle_degrees"] == pytest.approx(np.degrees(features["shot_angle"]))


def test_a_close_shot_has_higher_xg_than_a_long_shot(client: TestClient) -> None:
    close = client.post("/api/predict/shot", json={"x": 114.0, "y": 40.0}).json()
    far = client.post("/api/predict/shot", json={"x": 90.0, "y": 20.0}).json()

    assert close["predicted_xg"] > far["predicted_xg"]


def test_contextual_predict_requires_its_categorical_features(client: TestClient) -> None:
    missing = client.post(
        "/api/predict/shot",
        json={"model": "logistic_contextual", "x": 108.0, "y": 40.0},
    )
    complete = client.post(
        "/api/predict/shot",
        json={
            "model": "logistic_contextual",
            "x": 108.0,
            "y": 40.0,
            "body_part": "Right Foot",
            "shot_type": "Open Play",
            "technique": "Normal",
            "under_pressure": True,
        },
    )

    assert missing.status_code == 422
    assert complete.status_code == 200
    assert complete.json()["model"] == "logistic_contextual"


@pytest.mark.parametrize("payload", [{"x": -1, "y": 40}, {"x": 121, "y": 40}, {"x": 60, "y": 81}])
def test_impossible_coordinates_are_rejected(client: TestClient, payload: dict[str, float]) -> None:
    response = client.post("/api/predict/shot", json=payload)

    assert response.status_code == 422


def test_unknown_model_is_not_found(client: TestClient) -> None:
    response = client.post("/api/predict/shot", json={"model": "not-a-model", "x": 108, "y": 40})

    assert response.status_code == 404


def test_predict_without_artifacts_is_unavailable(processed_root: Path, tmp_path: Path) -> None:
    application = create_app(processed_root=processed_root, artifacts_root=tmp_path / "missing")

    response = TestClient(application).post("/api/predict/shot", json={"x": 108, "y": 40})

    assert response.status_code == 503


def test_statistics_summary_describes_the_fixture_shots(client: TestClient) -> None:
    response = client.get("/api/statistics/summary")

    assert response.status_code == 200
    payload = response.json()
    assert payload["shots"] == 3
    assert payload["goals"] == 2
    assert payload["shot_distance"]["n"] == 3
    assert payload["shot_angle_degrees"]["n"] == 3
    groups = {row["group"] for row in payload["conversion_by_body_part"]}
    assert "Right Foot" in groups
    assert isinstance(payload["findings"], list)


def test_matches_are_paginated(client: TestClient) -> None:
    first = client.get("/api/matches", params={"limit": 1, "offset": 0})
    second = client.get("/api/matches", params={"limit": 1, "offset": 1})

    assert first.status_code == 200
    assert first.json()["total"] == 2
    assert len(first.json()["matches"]) == 1
    assert first.json()["matches"][0]["match_id"] == 5001
    assert first.json()["matches"][0]["home_team"] == "Alpha FC"
    assert second.json()["matches"][0]["match_id"] == 5002


def test_unknown_match_is_not_found(client: TestClient) -> None:
    response = client.get("/api/matches/9999/shots")

    assert response.status_code == 404


def test_match_shots_include_coordinates_outcomes_and_xg(client: TestClient) -> None:
    response = client.get("/api/matches/5001/shots")

    assert response.status_code == 200
    payload = response.json()
    assert payload["match_id"] == 5001
    assert payload["home_team"] == "Alpha FC"
    assert payload["model"] == "logistic_baseline"
    assert len(payload["shots"]) == 2
    goals = {shot["shot_id"]: shot for shot in payload["shots"]}
    close = goals["00000000-0000-0000-0000-000000000003"]
    assert close["x"] == 110.0
    assert close["y"] == 40.0
    assert close["goal"] is True
    assert close["outcome"] == "Goal"
    assert close["predicted_xg"] is not None
    assert 0.0 <= close["predicted_xg"] <= 1.0
    assert close["statsbomb_xg"] == pytest.approx(0.42)


def test_missing_processed_data_is_unavailable(artifacts_root: Path, tmp_path: Path) -> None:
    application = create_app(processed_root=tmp_path / "never-built", artifacts_root=artifacts_root)

    response = TestClient(application).get("/api/summary")

    assert response.status_code == 503


def test_artifact_roundtrip_reloads_the_same_pipeline(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    names = persist_logistic_models(_synthetic_shots(), root)
    pipeline = artifacts.load_pipeline("logistic_baseline", root)
    info = artifacts.describe_artifact("logistic_baseline", root)

    assert names == ["logistic_baseline", "logistic_contextual"]
    assert info.validation_strategy == "group_by_match"
    close = pd.DataFrame({"shot_distance": [6.0], "shot_angle": [0.9]})
    far = pd.DataFrame({"shot_distance": [30.0], "shot_angle": [0.1]})

    assert predict_probability(pipeline, close)[0] > predict_probability(pipeline, far)[0]


def test_invalid_artifact_names_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(artifacts.ArtifactError, match="Invalid model name"):
        artifacts.artifact_dir("../secret", tmp_path)
