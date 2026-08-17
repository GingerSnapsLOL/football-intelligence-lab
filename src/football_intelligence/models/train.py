"""Fit xG models and persist artifacts for the API.

The API never trains. This module is the batch job that writes
``artifacts/shot_goal_probability/<name>/`` so ``GET /api/models`` can list every
family that has actually been fitted.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from football_intelligence.evaluation.metrics import evaluate_probabilities
from football_intelligence.evaluation.validation import (
    GroupSplit,
    assert_no_group_leakage,
    group_train_test_split,
)
from football_intelligence.features.shots import read_shot_dataset
from football_intelligence.models import artifacts
from football_intelligence.models import catboost as cbm
from football_intelligence.models import lightgbm as lgbm
from football_intelligence.models import logistic as lg
from football_intelligence.models import xgboost as xgbm
from football_intelligence.models.boosting import BoostedFit
from football_intelligence.models.preprocessing import (
    TARGET,
    feature_columns,
    fit,
    predict_probability,
)

logger = logging.getLogger(__name__)

SEED = 42
TEST_SIZE = 0.25

BoostFitFn = Callable[..., BoostedFit]
BoostBuilder = Callable[..., Pipeline]


def persist_all_models(
    frame: pd.DataFrame,
    artifacts_root: Path = artifacts.DEFAULT_ARTIFACTS_ROOT,
    *,
    random_state: int = SEED,
) -> list[str]:
    """Fit logistic and boosting models on one match-grouped split and save them."""
    split = group_train_test_split(frame, test_size=TEST_SIZE, random_state=random_state)
    assert_no_group_leakage(split)
    saved: list[str] = []

    for name, builder in (
        ("logistic_baseline", lg.build_baseline_model),
        ("logistic_contextual", lg.build_contextual_model),
    ):
        pipeline = fit(builder(random_state=random_state), split.train)
        saved.append(_save(name, pipeline, split, artifacts_root, random_state=random_state))

    boosted: tuple[tuple[str, BoostBuilder, BoostFitFn], ...] = (
        ("xgboost_contextual", xgbm.build_contextual_model, xgbm.fit_with_early_stopping),
        ("lightgbm_contextual", lgbm.build_contextual_model, lgbm.fit_with_early_stopping),
        ("catboost_contextual", cbm.build_contextual_model, cbm.fit_with_early_stopping),
    )
    for name, builder, fitter in boosted:
        result = fitter(builder(random_state=random_state), split.train, random_state=random_state)
        extra = {
            "library": result.library,
            "n_trees": result.n_trees,
            "best_iteration": result.best_iteration,
            "stopped_early": result.stopped_early,
        }
        saved.append(
            _save(
                name, result.pipeline, split, artifacts_root, extra=extra, random_state=random_state
            )
        )

    logger.info("persisted %s", ", ".join(saved))
    return saved


def _save(
    name: str,
    pipeline: Pipeline,
    split: GroupSplit,
    artifacts_root: Path,
    *,
    extra: dict[str, object] | None = None,
    random_state: int = SEED,
) -> str:
    metrics = evaluate_probabilities(
        split.test[TARGET].to_numpy(), predict_probability(pipeline, split.test)
    )
    metadata: dict[str, object] = {
        "features": feature_columns(pipeline),
        "training_samples": len(split.train),
        "test_samples": len(split.test),
        "n_train_groups": split.n_train_groups,
        "n_test_groups": split.n_test_groups,
        "validation_strategy": "group_by_match",
        "group_column": "match_id",
        "test_size": TEST_SIZE,
        "random_state": random_state,
    }
    if extra:
        metadata.update(extra)
    artifacts.save_artifact(name, pipeline, metrics, metadata, artifacts_root)
    return name


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train xG models and write artifacts.")
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=artifacts.DEFAULT_ARTIFACTS_ROOT,
        help="Directory for model artifacts.",
    )
    arguments = parser.parse_args(argv)
    names = persist_all_models(read_shot_dataset(), arguments.artifacts_root)
    print("wrote", ", ".join(names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
