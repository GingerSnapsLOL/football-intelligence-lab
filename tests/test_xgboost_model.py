"""Tests for the XGBoost shot model."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from football_intelligence.evaluation.validation import group_train_test_split
from football_intelligence.models import logistic as lg
from football_intelligence.models import xgboost as xgbm
from football_intelligence.models.preprocessing import (
    ModelError,
    feature_columns,
    predict_probability,
)

SEED = 20260824


@pytest.fixture
def shots() -> pd.DataFrame:
    """Synthetic shots whose goal probability depends on distance and angle."""
    rng = np.random.default_rng(SEED)
    n = 1_200
    distance = rng.uniform(2.0, 40.0, size=n)
    angle = 2.0 * np.arctan(4.0 / distance)
    logit = 1.2 - 0.20 * distance + 2.0 * angle
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
            "statsbomb_xg": probability,
            "goal": rng.random(n) < probability,
        }
    )


# --------------------------------------------------------------------------- #
# The comparison is fair
# --------------------------------------------------------------------------- #


def test_the_boosted_model_sees_exactly_the_same_features_as_the_logistic_one() -> None:
    """A model must not win a comparison by being given different information."""
    assert feature_columns(xgbm.build_contextual_model()) == feature_columns(
        lg.build_contextual_model()
    )
    assert feature_columns(xgbm.build_baseline_model()) == feature_columns(
        lg.build_baseline_model()
    )


def test_the_boosted_model_uses_the_same_target() -> None:
    assert xgbm.TARGET == lg.TARGET == "goal"


@pytest.mark.parametrize("leaky", ["statsbomb_xg", "goal", "end_x"])
def test_the_leakage_rule_applies_to_boosting_too(leaky: str) -> None:
    with pytest.raises(ModelError, match="cannot be used as a feature"):
        xgbm.build_xgboost_model(("shot_distance", leaky), (), ())


def test_no_default_boosted_model_touches_the_benchmark_column() -> None:
    for pipeline in (xgbm.build_baseline_model(), xgbm.build_contextual_model()):
        assert "statsbomb_xg" not in feature_columns(pipeline)


# --------------------------------------------------------------------------- #
# Fitting and prediction
# --------------------------------------------------------------------------- #


def test_the_pipeline_fits_and_predicts(shots: pd.DataFrame) -> None:
    result = xgbm.fit_with_early_stopping(
        xgbm.build_contextual_model(random_state=SEED), shots, random_state=SEED
    )

    assert isinstance(result.pipeline, Pipeline)
    probabilities = predict_probability(result.pipeline, shots)
    assert probabilities.shape == (len(shots),)
    assert np.all(np.isfinite(probabilities))


def test_predicted_probabilities_lie_in_the_unit_interval(shots: pd.DataFrame) -> None:
    for builder in (xgbm.build_baseline_model, xgbm.build_contextual_model):
        result = xgbm.fit_with_early_stopping(builder(random_state=SEED), shots, random_state=SEED)

        probabilities = predict_probability(result.pipeline, shots)

        assert probabilities.min() >= 0.0
        assert probabilities.max() <= 1.0


def test_predict_probability_returns_the_positive_class(shots: pd.DataFrame) -> None:
    result = xgbm.fit_with_early_stopping(
        xgbm.build_baseline_model(random_state=SEED), shots, random_state=SEED
    )

    ours = predict_probability(result.pipeline, shots)
    both = result.pipeline.predict_proba(shots[feature_columns(result.pipeline)])

    np.testing.assert_allclose(ours, both[:, 1])
    np.testing.assert_allclose(both.sum(axis=1), 1.0, rtol=1e-6)


def test_the_model_learns_that_distance_reduces_the_chance(shots: pd.DataFrame) -> None:
    result = xgbm.fit_with_early_stopping(
        xgbm.build_baseline_model(random_state=SEED), shots, random_state=SEED
    )

    close = pd.DataFrame({"shot_distance": [5.0], "shot_angle": [2 * np.arctan(4.0 / 5.0)]})
    far = pd.DataFrame({"shot_distance": [32.0], "shot_angle": [2 * np.arctan(4.0 / 32.0)]})

    assert (
        predict_probability(result.pipeline, close)[0]
        > predict_probability(result.pipeline, far)[0]
    )


def test_predictions_are_reproducible(shots: pd.DataFrame) -> None:
    first = xgbm.fit_with_early_stopping(
        xgbm.build_contextual_model(random_state=SEED), shots, random_state=SEED
    )
    second = xgbm.fit_with_early_stopping(
        xgbm.build_contextual_model(random_state=SEED), shots, random_state=SEED
    )

    np.testing.assert_allclose(
        predict_probability(first.pipeline, shots), predict_probability(second.pipeline, shots)
    )
    assert first.n_trees == second.n_trees


# --------------------------------------------------------------------------- #
# Early stopping
# --------------------------------------------------------------------------- #


def test_early_stopping_reports_what_it_decided(shots: pd.DataFrame) -> None:
    result = xgbm.fit_with_early_stopping(
        xgbm.build_contextual_model(random_state=SEED), shots, random_state=SEED
    )

    assert 0 < result.n_trees <= xgbm.MAX_TREES
    assert 0 <= result.best_iteration < result.n_trees
    assert np.isfinite(result.best_validation_loss)
    assert result.n_validation_rows > 0
    assert result.n_validation_groups > 0


def test_early_stopping_uses_fewer_trees_than_the_budget(shots: pd.DataFrame) -> None:
    result = xgbm.fit_with_early_stopping(
        xgbm.build_contextual_model(random_state=SEED), shots, random_state=SEED
    )

    assert result.stopped_early
    assert result.n_trees < xgbm.MAX_TREES


def test_the_validation_slice_is_split_by_match_not_by_row(shots: pd.DataFrame) -> None:
    """Early stopping would be fooled by a validation slice sharing matches."""
    inner = group_train_test_split(shots, test_size=0.2, random_state=SEED)

    assert inner.overlapping_groups == set()
    # The fit uses exactly this split, so its validation size must match.
    result = xgbm.fit_with_early_stopping(
        xgbm.build_contextual_model(random_state=SEED),
        shots,
        validation_fraction=0.2,
        random_state=SEED,
    )
    assert result.n_validation_rows == len(inner.test)
    assert result.n_validation_groups == inner.n_test_groups


def test_early_stopping_watches_only_training_matches(shots: pd.DataFrame) -> None:
    """Choosing the tree count on the test set would invalidate the whole report."""
    split = group_train_test_split(shots, test_size=0.25, random_state=SEED)
    result = xgbm.fit_with_early_stopping(
        xgbm.build_contextual_model(random_state=SEED),
        split.train,
        validation_fraction=0.2,
        random_state=SEED,
    )

    # The fit carves its validation slice out of train with the same seed, so it
    # can be reconstructed and checked against the held-out matches.
    inner = group_train_test_split(split.train, test_size=0.2, random_state=SEED)

    assert result.n_validation_rows == len(inner.test)
    assert set(inner.test["match_id"]) <= set(split.train["match_id"])
    assert set(inner.test["match_id"]).isdisjoint(set(split.test["match_id"]))


def test_fitting_requires_every_declared_column(shots: pd.DataFrame) -> None:
    with pytest.raises(ModelError, match="missing required column"):
        xgbm.fit_with_early_stopping(
            xgbm.build_contextual_model(), shots.drop(columns=["technique"])
        )


def test_fitting_requires_the_group_column(shots: pd.DataFrame) -> None:
    with pytest.raises(ModelError, match="missing required column"):
        xgbm.fit_with_early_stopping(
            xgbm.build_contextual_model(), shots.drop(columns=["match_id"])
        )


def test_fitting_needs_both_outcomes(shots: pd.DataFrame) -> None:
    with pytest.raises(ModelError, match="single value"):
        xgbm.fit_with_early_stopping(xgbm.build_contextual_model(), shots.assign(goal=False))


def test_a_validation_slice_without_goals_is_reported(shots: pd.DataFrame) -> None:
    """Goals only in a handful of matches, so some slices contain none."""
    sparse = shots.copy()
    sparse["goal"] = sparse["match_id"] < 2

    with pytest.raises(ModelError, match="validation slice contains a single outcome"):
        xgbm.fit_with_early_stopping(
            xgbm.build_contextual_model(), sparse, validation_fraction=0.5, random_state=3
        )


# --------------------------------------------------------------------------- #
# Feature importance
# --------------------------------------------------------------------------- #


def test_feature_importance_covers_every_transformed_feature(shots: pd.DataFrame) -> None:
    result = xgbm.fit_with_early_stopping(
        xgbm.build_contextual_model(random_state=SEED), shots, random_state=SEED
    )

    importance = xgbm.feature_importance(result.pipeline)

    assert list(importance.columns) == ["feature", "gain", "share"]
    assert len(importance) == len(result.pipeline.named_steps["features"].get_feature_names_out())
    assert importance["gain"].is_monotonic_decreasing
    assert importance["share"].sum() == pytest.approx(1.0)


def test_geometry_dominates_importance_when_it_generated_the_data(
    shots: pd.DataFrame,
) -> None:
    result = xgbm.fit_with_early_stopping(
        xgbm.build_contextual_model(random_state=SEED), shots, random_state=SEED
    )

    importance = xgbm.feature_importance(result.pipeline).set_index("feature")
    geometry = importance.loc[["shot_distance", "shot_angle"], "share"].sum()

    # The synthetic outcome depends on nothing else, so the noise features should
    # not dominate.
    assert geometry > 0.3
