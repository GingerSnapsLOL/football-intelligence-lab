"""Tests for the LightGBM shot model."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from football_intelligence.evaluation.validation import group_train_test_split
from football_intelligence.models import lightgbm as lgbm
from football_intelligence.models import logistic as lg
from football_intelligence.models import xgboost as xgbm
from football_intelligence.models.preprocessing import (
    ModelError,
    feature_columns,
    predict_probability,
)

SEED = 20260825


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
# The three-way comparison is fair
# --------------------------------------------------------------------------- #


def test_all_three_models_see_exactly_the_same_features() -> None:
    """No model may win a comparison by being handed different information."""
    contextual = [
        feature_columns(lg.build_contextual_model()),
        feature_columns(xgbm.build_contextual_model()),
        feature_columns(lgbm.build_contextual_model()),
    ]
    baseline = [
        feature_columns(lg.build_baseline_model()),
        feature_columns(xgbm.build_baseline_model()),
        feature_columns(lgbm.build_baseline_model()),
    ]

    assert contextual[0] == contextual[1] == contextual[2]
    assert baseline[0] == baseline[1] == baseline[2]


def test_all_three_models_share_the_same_target() -> None:
    assert lgbm.TARGET == xgbm.TARGET == lg.TARGET == "goal"


@pytest.mark.parametrize("leaky", ["statsbomb_xg", "goal", "end_y"])
def test_the_leakage_rule_applies_to_lightgbm_too(leaky: str) -> None:
    with pytest.raises(ModelError, match="cannot be used as a feature"):
        lgbm.build_lightgbm_model(("shot_distance", leaky), (), ())


def test_no_default_model_touches_the_benchmark_column() -> None:
    for pipeline in (lgbm.build_baseline_model(), lgbm.build_contextual_model()):
        assert "statsbomb_xg" not in feature_columns(pipeline)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def test_the_leaf_budget_is_far_below_the_library_default() -> None:
    """Leaf-wise growth with 31 leaves would memorise a few hundred goals."""
    model = lgbm.build_contextual_model().named_steps["model"]

    assert model.num_leaves == lgbm.DEFAULT_NUM_LEAVES == 8
    assert model.max_depth == 3
    assert model.min_child_samples == lgbm.DEFAULT_MIN_CHILD_SAMPLES == 30


def test_row_subsampling_is_actually_enabled() -> None:
    """LightGBM silently ignores subsample unless subsample_freq is set."""
    model = lgbm.build_contextual_model().named_steps["model"]

    assert model.subsample == 0.8
    assert model.subsample_freq >= 1


# --------------------------------------------------------------------------- #
# Fitting and prediction
# --------------------------------------------------------------------------- #


def test_the_pipeline_fits_and_predicts(shots: pd.DataFrame) -> None:
    result = lgbm.fit_with_early_stopping(
        lgbm.build_contextual_model(random_state=SEED), shots, random_state=SEED
    )

    assert isinstance(result.pipeline, Pipeline)
    assert result.library == "lightgbm"
    probabilities = predict_probability(result.pipeline, shots)
    assert probabilities.shape == (len(shots),)
    assert np.all(np.isfinite(probabilities))


def test_predicted_probabilities_lie_in_the_unit_interval(shots: pd.DataFrame) -> None:
    for builder in (lgbm.build_baseline_model, lgbm.build_contextual_model):
        result = lgbm.fit_with_early_stopping(builder(random_state=SEED), shots, random_state=SEED)

        probabilities = predict_probability(result.pipeline, shots)

        assert probabilities.min() >= 0.0
        assert probabilities.max() <= 1.0


def test_predict_probability_returns_the_positive_class(shots: pd.DataFrame) -> None:
    result = lgbm.fit_with_early_stopping(
        lgbm.build_baseline_model(random_state=SEED), shots, random_state=SEED
    )

    ours = predict_probability(result.pipeline, shots)
    both = result.pipeline.predict_proba(shots[feature_columns(result.pipeline)])

    np.testing.assert_allclose(ours, both[:, 1])
    np.testing.assert_allclose(both.sum(axis=1), 1.0, rtol=1e-6)


def test_the_model_learns_that_distance_reduces_the_chance(shots: pd.DataFrame) -> None:
    result = lgbm.fit_with_early_stopping(
        lgbm.build_baseline_model(random_state=SEED), shots, random_state=SEED
    )

    close = pd.DataFrame({"shot_distance": [5.0], "shot_angle": [2 * np.arctan(4.0 / 5.0)]})
    far = pd.DataFrame({"shot_distance": [32.0], "shot_angle": [2 * np.arctan(4.0 / 32.0)]})

    assert (
        predict_probability(result.pipeline, close)[0]
        > predict_probability(result.pipeline, far)[0]
    )


def test_predictions_are_reproducible(shots: pd.DataFrame) -> None:
    first = lgbm.fit_with_early_stopping(
        lgbm.build_contextual_model(random_state=SEED), shots, random_state=SEED
    )
    second = lgbm.fit_with_early_stopping(
        lgbm.build_contextual_model(random_state=SEED), shots, random_state=SEED
    )

    np.testing.assert_allclose(
        predict_probability(first.pipeline, shots), predict_probability(second.pipeline, shots)
    )
    assert first.n_trees == second.n_trees


# --------------------------------------------------------------------------- #
# Early stopping, shared with the other boosting models
# --------------------------------------------------------------------------- #


def test_early_stopping_reports_what_it_decided(shots: pd.DataFrame) -> None:
    result = lgbm.fit_with_early_stopping(
        lgbm.build_contextual_model(random_state=SEED), shots, random_state=SEED
    )

    assert 0 < result.n_trees <= lgbm.MAX_TREES
    assert result.stopped_early
    assert np.isfinite(result.best_validation_loss)
    assert result.n_validation_rows > 0
    assert result.n_validation_groups > 0


def test_early_stopping_watches_only_training_matches(shots: pd.DataFrame) -> None:
    split = group_train_test_split(shots, test_size=0.25, random_state=SEED)
    result = lgbm.fit_with_early_stopping(
        lgbm.build_contextual_model(random_state=SEED),
        split.train,
        validation_fraction=0.2,
        random_state=SEED,
    )
    inner = group_train_test_split(split.train, test_size=0.2, random_state=SEED)

    assert result.n_validation_rows == len(inner.test)
    assert set(inner.test["match_id"]).isdisjoint(set(split.test["match_id"]))


def test_both_boosting_models_use_the_same_validation_slice(shots: pd.DataFrame) -> None:
    """Same split machinery, so early stopping is decided on the same data."""
    boosted = xgbm.fit_with_early_stopping(
        xgbm.build_contextual_model(random_state=SEED), shots, random_state=SEED
    )
    light = lgbm.fit_with_early_stopping(
        lgbm.build_contextual_model(random_state=SEED), shots, random_state=SEED
    )

    assert boosted.n_validation_rows == light.n_validation_rows
    assert boosted.n_validation_groups == light.n_validation_groups


def test_fitting_requires_every_declared_column(shots: pd.DataFrame) -> None:
    with pytest.raises(ModelError, match="missing required column"):
        lgbm.fit_with_early_stopping(
            lgbm.build_contextual_model(), shots.drop(columns=["body_part"])
        )


def test_fitting_needs_both_outcomes(shots: pd.DataFrame) -> None:
    with pytest.raises(ModelError, match="single value"):
        lgbm.fit_with_early_stopping(lgbm.build_contextual_model(), shots.assign(goal=False))


def test_a_validation_slice_without_goals_is_reported(shots: pd.DataFrame) -> None:
    sparse = shots.copy()
    sparse["goal"] = sparse["match_id"] < 2

    with pytest.raises(ModelError, match="validation slice contains a single outcome"):
        lgbm.fit_with_early_stopping(
            lgbm.build_contextual_model(), sparse, validation_fraction=0.5, random_state=3
        )


# --------------------------------------------------------------------------- #
# Feature importance
# --------------------------------------------------------------------------- #


def test_feature_importance_covers_every_transformed_feature(shots: pd.DataFrame) -> None:
    result = lgbm.fit_with_early_stopping(
        lgbm.build_contextual_model(random_state=SEED), shots, random_state=SEED
    )

    importance = lgbm.feature_importance(result.pipeline)

    assert list(importance.columns) == ["feature", "gain", "share"]
    assert len(importance) == len(result.pipeline.named_steps["features"].get_feature_names_out())
    assert importance["gain"].is_monotonic_decreasing
    assert importance["share"].sum() == pytest.approx(1.0)


def test_geometry_dominates_importance_when_it_generated_the_data(
    shots: pd.DataFrame,
) -> None:
    result = lgbm.fit_with_early_stopping(
        lgbm.build_contextual_model(random_state=SEED), shots, random_state=SEED
    )

    importance = lgbm.feature_importance(result.pipeline).set_index("feature")
    geometry = importance.loc[["shot_distance", "shot_angle"], "share"].sum()

    assert geometry > 0.5
