"""Tests for the CatBoost shot model, which handles categoricals natively."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from football_intelligence.evaluation.validation import group_train_test_split
from football_intelligence.models import catboost as cb
from football_intelligence.models import lightgbm as lgbm
from football_intelligence.models import logistic as lg
from football_intelligence.models import xgboost as xgbm
from football_intelligence.models.preprocessing import (
    ModelError,
    feature_columns,
    predict_probability,
)

SEED = 20260826


@pytest.fixture
def shots() -> pd.DataFrame:
    """Synthetic shots where both geometry and one category drive the outcome."""
    rng = np.random.default_rng(SEED)
    n = 1_200
    distance = rng.uniform(2.0, 40.0, size=n)
    angle = 2.0 * np.arctan(4.0 / distance)
    body_part = rng.choice(["Right Foot", "Left Foot", "Head"], size=n)
    logit = 1.2 - 0.20 * distance + 2.0 * angle + 0.8 * (body_part == "Head")
    probability = 1.0 / (1.0 + np.exp(-logit))
    return pd.DataFrame(
        {
            "match_id": np.repeat(np.arange(n // 20), 20),
            "player_id": rng.integers(0, 300, size=n),
            "team_id": rng.integers(0, 20, size=n),
            "shot_distance": distance,
            "shot_angle": angle,
            "body_part": body_part,
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
# Identifier features are refused, with a reason
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("identifier", ["player_id", "team_id", "match_id"])
def test_identifier_features_are_refused_by_default(identifier: str) -> None:
    """The high-cardinality columns CatBoost could encode are the ones to resist."""
    with pytest.raises(ModelError, match="is an identifier, not a shot property"):
        cb.build_catboost_model(("shot_distance",), (identifier,), ())


def test_the_refusal_explains_the_generalisation_problem() -> None:
    with pytest.raises(ModelError, match="median of 3 shots each"):
        cb.build_catboost_model(("shot_distance",), ("player_id",), ())

    with pytest.raises(ModelError, match="crosses the train/test boundary"):
        cb.build_catboost_model(("shot_distance",), ("team_id",), ())


def test_identifier_features_need_an_explicit_opt_in() -> None:
    pipeline = cb.build_catboost_model(
        ("shot_distance",), ("player_id",), (), allow_identifier_features=True
    )

    assert "player_id" in feature_columns(pipeline)


def test_no_default_model_uses_an_identifier() -> None:
    for pipeline in (cb.build_baseline_model(), cb.build_contextual_model()):
        columns = feature_columns(pipeline)
        assert not set(columns) & set(cb.IDENTIFIER_FEATURES)


@pytest.mark.parametrize("leaky", ["statsbomb_xg", "goal", "end_z"])
def test_the_leakage_rule_applies_to_catboost_too(leaky: str) -> None:
    with pytest.raises(ModelError, match="cannot be used as a feature"):
        cb.build_catboost_model(("shot_distance", leaky), (), ())


# --------------------------------------------------------------------------- #
# Categoricals reach the model unencoded
# --------------------------------------------------------------------------- #


def test_categorical_columns_are_declared_for_native_handling() -> None:
    pipeline = cb.build_contextual_model()

    assert cb.categorical_feature_names(pipeline) == list(cb.CONTEXT_CATEGORICAL)


def test_the_baseline_model_declares_no_categoricals() -> None:
    assert cb.categorical_feature_names(cb.build_baseline_model()) == []


def test_the_preprocessor_passes_categories_through_as_strings(shots: pd.DataFrame) -> None:
    """One-hot encoding here would discard what CatBoost is built to use."""
    pipeline = cb.build_contextual_model(random_state=SEED)
    transformed = pipeline.named_steps["features"].fit_transform(shots[feature_columns(pipeline)])

    assert isinstance(transformed, pd.DataFrame)
    assert list(transformed.columns) == feature_columns(pipeline)
    assert set(transformed["body_part"].unique()) <= {"Right Foot", "Left Foot", "Head"}


def test_all_four_models_consume_the_same_raw_columns() -> None:
    """Encoding differs; the information does not."""
    columns = [
        feature_columns(lg.build_contextual_model()),
        feature_columns(xgbm.build_contextual_model()),
        feature_columns(lgbm.build_contextual_model()),
        feature_columns(cb.build_contextual_model()),
    ]

    assert columns[0] == columns[1] == columns[2] == columns[3]


def test_all_four_models_share_the_same_target() -> None:
    assert cb.TARGET == lgbm.TARGET == xgbm.TARGET == lg.TARGET == "goal"


# --------------------------------------------------------------------------- #
# Fitting and prediction
# --------------------------------------------------------------------------- #


def test_the_pipeline_fits_and_predicts(shots: pd.DataFrame) -> None:
    result = cb.fit_with_early_stopping(
        cb.build_contextual_model(random_state=SEED), shots, random_state=SEED
    )

    assert isinstance(result.pipeline, Pipeline)
    assert result.library == "catboost"
    probabilities = predict_probability(result.pipeline, shots)
    assert probabilities.shape == (len(shots),)
    assert np.all(np.isfinite(probabilities))


def test_predicted_probabilities_lie_in_the_unit_interval(shots: pd.DataFrame) -> None:
    for builder in (cb.build_baseline_model, cb.build_contextual_model):
        result = cb.fit_with_early_stopping(builder(random_state=SEED), shots, random_state=SEED)

        probabilities = predict_probability(result.pipeline, shots)

        assert probabilities.min() >= 0.0
        assert probabilities.max() <= 1.0


def test_predict_probability_returns_the_positive_class(shots: pd.DataFrame) -> None:
    result = cb.fit_with_early_stopping(
        cb.build_baseline_model(random_state=SEED), shots, random_state=SEED
    )

    ours = predict_probability(result.pipeline, shots)
    both = result.pipeline.predict_proba(shots[feature_columns(result.pipeline)])

    np.testing.assert_allclose(ours, both[:, 1])
    np.testing.assert_allclose(both.sum(axis=1), 1.0, rtol=1e-6)


def test_the_model_learns_that_distance_reduces_the_chance(shots: pd.DataFrame) -> None:
    result = cb.fit_with_early_stopping(
        cb.build_baseline_model(random_state=SEED), shots, random_state=SEED
    )

    close = pd.DataFrame({"shot_distance": [5.0], "shot_angle": [2 * np.arctan(4.0 / 5.0)]})
    far = pd.DataFrame({"shot_distance": [32.0], "shot_angle": [2 * np.arctan(4.0 / 32.0)]})

    assert (
        predict_probability(result.pipeline, close)[0]
        > predict_probability(result.pipeline, far)[0]
    )


def test_the_model_uses_the_categorical_signal(shots: pd.DataFrame) -> None:
    """Headers were generated with a real advantage, so the model should find it."""
    result = cb.fit_with_early_stopping(
        cb.build_contextual_model(random_state=SEED), shots, random_state=SEED
    )
    template = {
        "shot_distance": [10.0],
        "shot_angle": [2 * np.arctan(4.0 / 10.0)],
        "shot_type": ["Open Play"],
        "technique": ["Normal"],
        "under_pressure": [False],
        "first_time": [False],
    }

    header = predict_probability(result.pipeline, pd.DataFrame({**template, "body_part": ["Head"]}))
    foot = predict_probability(
        result.pipeline, pd.DataFrame({**template, "body_part": ["Right Foot"]})
    )

    assert header[0] > foot[0]


def test_predictions_are_reproducible(shots: pd.DataFrame) -> None:
    first = cb.fit_with_early_stopping(
        cb.build_contextual_model(random_state=SEED), shots, random_state=SEED
    )
    second = cb.fit_with_early_stopping(
        cb.build_contextual_model(random_state=SEED), shots, random_state=SEED
    )

    np.testing.assert_allclose(
        predict_probability(first.pipeline, shots), predict_probability(second.pipeline, shots)
    )
    assert first.n_trees == second.n_trees


def test_an_unseen_category_does_not_break_prediction(shots: pd.DataFrame) -> None:
    result = cb.fit_with_early_stopping(
        cb.build_contextual_model(random_state=SEED), shots, random_state=SEED
    )
    unseen = shots.head(5).copy()
    unseen["technique"] = "Bicycle Kick From Orbit"

    probabilities = predict_probability(result.pipeline, unseen)

    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


# --------------------------------------------------------------------------- #
# Early stopping, shared with the other boosting models
# --------------------------------------------------------------------------- #


def test_early_stopping_reports_what_it_decided(shots: pd.DataFrame) -> None:
    result = cb.fit_with_early_stopping(
        cb.build_contextual_model(random_state=SEED), shots, random_state=SEED
    )

    assert 0 < result.n_trees <= cb.MAX_TREES
    assert np.isfinite(result.best_validation_loss)
    assert result.n_validation_rows > 0
    assert result.n_validation_groups > 0


def test_early_stopping_watches_only_training_matches(shots: pd.DataFrame) -> None:
    split = group_train_test_split(shots, test_size=0.25, random_state=SEED)
    result = cb.fit_with_early_stopping(
        cb.build_contextual_model(random_state=SEED),
        split.train,
        validation_fraction=0.2,
        random_state=SEED,
    )
    inner = group_train_test_split(split.train, test_size=0.2, random_state=SEED)

    assert result.n_validation_rows == len(inner.test)
    assert set(inner.test["match_id"]).isdisjoint(set(split.test["match_id"]))


def test_all_three_boosting_models_use_the_same_validation_slice(shots: pd.DataFrame) -> None:
    fits = [
        xgbm.fit_with_early_stopping(
            xgbm.build_contextual_model(random_state=SEED), shots, random_state=SEED
        ),
        lgbm.fit_with_early_stopping(
            lgbm.build_contextual_model(random_state=SEED), shots, random_state=SEED
        ),
        cb.fit_with_early_stopping(
            cb.build_contextual_model(random_state=SEED), shots, random_state=SEED
        ),
    ]

    assert len({fit.n_validation_rows for fit in fits}) == 1
    assert len({fit.n_validation_groups for fit in fits}) == 1


def test_fitting_requires_every_declared_column(shots: pd.DataFrame) -> None:
    with pytest.raises(ModelError, match="missing required column"):
        cb.fit_with_early_stopping(cb.build_contextual_model(), shots.drop(columns=["shot_type"]))


def test_fitting_needs_both_outcomes(shots: pd.DataFrame) -> None:
    with pytest.raises(ModelError, match="single value"):
        cb.fit_with_early_stopping(cb.build_contextual_model(), shots.assign(goal=False))


# --------------------------------------------------------------------------- #
# Feature importance
# --------------------------------------------------------------------------- #


def test_importance_is_reported_per_original_feature(shots: pd.DataFrame) -> None:
    """A categorical appears once, not once per level as with one-hot encoding."""
    result = cb.fit_with_early_stopping(
        cb.build_contextual_model(random_state=SEED), shots, random_state=SEED
    )

    importance = cb.feature_importance(result.pipeline)

    assert list(importance.columns) == ["feature", "importance", "share"]
    assert set(importance["feature"]) == set(feature_columns(result.pipeline))
    assert importance["importance"].is_monotonic_decreasing
    assert importance["share"].sum() == pytest.approx(1.0)


def test_geometry_dominates_importance_when_it_generated_the_data(
    shots: pd.DataFrame,
) -> None:
    result = cb.fit_with_early_stopping(
        cb.build_contextual_model(random_state=SEED), shots, random_state=SEED
    )

    importance = cb.feature_importance(result.pipeline).set_index("feature")
    geometry = importance.loc[["shot_distance", "shot_angle"], "share"].sum()

    assert geometry > 0.5
