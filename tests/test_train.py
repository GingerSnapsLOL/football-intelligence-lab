"""Tests that boosting models are persisted so the API can list them."""

from pathlib import Path

import numpy as np
import pandas as pd

from football_intelligence.models import artifacts
from football_intelligence.models.train import persist_all_models

SEED = 20260817


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


def test_persist_all_models_writes_boosting_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    names = persist_all_models(_synthetic_shots(), root, random_state=SEED)
    listed = artifacts.list_artifacts(root)
    for expected in (
        "logistic_baseline",
        "logistic_contextual",
        "xgboost_contextual",
        "lightgbm_contextual",
        "catboost_contextual",
    ):
        assert expected in names
        assert expected in listed
        metrics = artifacts.load_metrics(expected, root)
        assert 0.0 < float(metrics["roc_auc"]) < 1.0
        assert 0.0 <= float(metrics["brier_score"]) <= 1.0
