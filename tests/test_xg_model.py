"""Tests for the logistic xG baseline, its metrics and the group-aware split."""

import numpy as np
import pandas as pd
import pytest
from sklearn import metrics as sk_metrics
from sklearn.pipeline import Pipeline

from football_intelligence.evaluation.metrics import (
    MetricError,
    compare_models,
    evaluate_probabilities,
)
from football_intelligence.evaluation.validation import (
    ValidationError,
    assert_no_group_leakage,
    group_train_test_split,
)
from football_intelligence.models.logistic import (
    BASELINE_NUMERIC,
    CONTEXT_BOOLEAN,
    CONTEXT_CATEGORICAL,
    ModelError,
    build_baseline_model,
    build_contextual_model,
    build_pipeline,
    coefficients,
    feature_columns,
    fit,
    predict_probability,
)

SEED = 20260823


def cell(frame: pd.DataFrame, row: str, column: str) -> float:
    """Read one numeric cell as a float.

    pandas-stubs types ``.loc`` as a broad scalar union, so a direct comparison
    does not type-check even when the value is always numeric.
    """
    return float(frame.loc[row, column])  # type: ignore[arg-type]


@pytest.fixture
def shots() -> pd.DataFrame:
    """A synthetic shot dataset whose goal probability really depends on distance."""
    rng = np.random.default_rng(SEED)
    n = 900
    distance = rng.uniform(2.0, 40.0, size=n)
    angle = np.arctan2(8.0 * np.maximum(120.0 - (120.0 - distance), 1.0), distance**2)
    logit = 1.5 - 0.22 * distance + 2.0 * angle
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
# Pipeline construction and fitting
# --------------------------------------------------------------------------- #


def test_the_baseline_pipeline_uses_only_distance_and_angle() -> None:
    pipeline = build_baseline_model(random_state=SEED)

    assert isinstance(pipeline, Pipeline)
    assert feature_columns(pipeline) == list(BASELINE_NUMERIC)


def test_the_contextual_pipeline_adds_the_documented_features() -> None:
    pipeline = build_contextual_model(random_state=SEED)

    assert feature_columns(pipeline) == [
        *BASELINE_NUMERIC,
        *CONTEXT_CATEGORICAL,
        *CONTEXT_BOOLEAN,
    ]


def test_a_pipeline_fits_and_predicts(shots: pd.DataFrame) -> None:
    pipeline = fit(build_baseline_model(random_state=SEED), shots)

    probabilities = predict_probability(pipeline, shots)

    assert probabilities.shape == (len(shots),)
    assert np.all(np.isfinite(probabilities))


def test_predicted_probabilities_lie_in_the_unit_interval(shots: pd.DataFrame) -> None:
    for builder in (build_baseline_model, build_contextual_model):
        pipeline = fit(builder(random_state=SEED), shots)

        probabilities = predict_probability(pipeline, shots)

        assert probabilities.min() >= 0.0
        assert probabilities.max() <= 1.0


def test_predict_probability_returns_the_positive_class_column(shots: pd.DataFrame) -> None:
    pipeline = fit(build_baseline_model(random_state=SEED), shots)

    ours = predict_probability(pipeline, shots)
    sklearn_output = pipeline.predict_proba(shots[list(BASELINE_NUMERIC)])

    np.testing.assert_allclose(ours, sklearn_output[:, 1])
    # The two columns must sum to one, so we are not accidentally returning P(miss).
    np.testing.assert_allclose(sklearn_output.sum(axis=1), 1.0)


def test_the_model_learns_that_distance_reduces_the_chance(shots: pd.DataFrame) -> None:
    pipeline = fit(build_baseline_model(random_state=SEED), shots)

    close = pd.DataFrame({"shot_distance": [6.0], "shot_angle": [0.9]})
    far = pd.DataFrame({"shot_distance": [30.0], "shot_angle": [0.1]})

    assert predict_probability(pipeline, close)[0] > predict_probability(pipeline, far)[0]


@pytest.mark.filterwarnings("ignore:Found unknown categories:UserWarning")
def test_the_contextual_model_handles_an_unseen_category(shots: pd.DataFrame) -> None:
    """Rare and unseen levels must not blow up at prediction time.

    sklearn warns that it mapped the unknown level to the infrequent bucket, which
    is precisely the behaviour being verified here.
    """
    pipeline = fit(build_contextual_model(random_state=SEED), shots)
    unseen = shots.head(5).copy()
    unseen["technique"] = "Bicycle Kick From Orbit"

    probabilities = predict_probability(pipeline, unseen)

    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_fitting_requires_every_declared_column(shots: pd.DataFrame) -> None:
    with pytest.raises(ModelError, match="missing required column"):
        fit(build_contextual_model(), shots.drop(columns=["technique"]))


def test_fitting_needs_both_outcomes(shots: pd.DataFrame) -> None:
    only_misses = shots.assign(goal=False)

    with pytest.raises(ModelError, match="single value"):
        fit(build_baseline_model(), only_misses)


def test_a_pipeline_needs_at_least_one_feature() -> None:
    with pytest.raises(ModelError, match="At least one feature"):
        build_pipeline((), (), ())


def test_features_cannot_be_repeated() -> None:
    with pytest.raises(ModelError, match="repeated across groups"):
        build_pipeline(("shot_distance",), (), ("shot_distance",))


# --------------------------------------------------------------------------- #
# The leakage rule
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("leaky", ["statsbomb_xg", "goal", "outcome", "end_x", "end_y", "end_z"])
def test_leaky_features_are_refused(leaky: str) -> None:
    with pytest.raises(ModelError, match="cannot be used as a feature"):
        build_pipeline(("shot_distance", leaky))


def test_the_statsbomb_estimate_is_refused_with_an_explanation() -> None:
    with pytest.raises(ModelError, match="another model's estimate of the target"):
        build_pipeline(("shot_distance",), (), ("statsbomb_xg",))


def test_no_default_model_touches_the_benchmark_column() -> None:
    for pipeline in (build_baseline_model(), build_contextual_model()):
        assert "statsbomb_xg" not in feature_columns(pipeline)


# --------------------------------------------------------------------------- #
# Coefficients
# --------------------------------------------------------------------------- #


def test_coefficients_are_reported_with_odds_ratios(shots: pd.DataFrame) -> None:
    pipeline = fit(build_baseline_model(random_state=SEED), shots)

    table = coefficients(pipeline)

    assert list(table.columns) == ["feature", "coefficient", "odds_ratio"]
    assert table.iloc[0]["feature"] == "(intercept)"
    np.testing.assert_allclose(table["odds_ratio"], np.exp(table["coefficient"]))


def test_the_distance_coefficient_is_negative(shots: pd.DataFrame) -> None:
    pipeline = fit(build_baseline_model(random_state=SEED), shots)

    table = coefficients(pipeline).set_index("feature")

    assert cell(table, "shot_distance", "coefficient") < 0
    assert cell(table, "shot_distance", "odds_ratio") < 1


# --------------------------------------------------------------------------- #
# Group-aware splitting
# --------------------------------------------------------------------------- #


def test_the_split_keeps_whole_matches_on_one_side(shots: pd.DataFrame) -> None:
    split = group_train_test_split(shots, test_size=0.25, random_state=SEED)

    assert split.overlapping_groups == set()
    assert_no_group_leakage(split)
    assert len(split.train) + len(split.test) == len(shots)
    assert split.n_train_groups + split.n_test_groups == shots["match_id"].nunique()


def test_the_split_is_reproducible(shots: pd.DataFrame) -> None:
    first = group_train_test_split(shots, random_state=SEED)
    second = group_train_test_split(shots, random_state=SEED)

    pd.testing.assert_frame_equal(first.test, second.test)


def test_a_random_row_split_would_leak_matches(shots: pd.DataFrame) -> None:
    """The comparison that justifies the group split."""
    rng = np.random.default_rng(SEED)
    shuffled = shots.sample(frac=1.0, random_state=1).reset_index(drop=True)
    cut = int(0.75 * len(shuffled))
    naive_train, naive_test = shuffled.iloc[:cut], shuffled.iloc[cut:]
    del rng

    shared = set(naive_train["match_id"]) & set(naive_test["match_id"])
    grouped = group_train_test_split(shots, test_size=0.25, random_state=SEED)

    assert len(shared) > 20  # nearly every match ends up on both sides
    assert grouped.overlapping_groups == set()


def test_leakage_is_reported_when_it_exists(shots: pd.DataFrame) -> None:
    split = group_train_test_split(shots, random_state=SEED)
    contaminated = type(split)(
        train=pd.concat([split.train, split.test.head(5)], ignore_index=True),
        test=split.test,
        group_column="match_id",
    )

    with pytest.raises(ValidationError, match="appear in both train and test"):
        assert_no_group_leakage(contaminated)


def test_the_split_rejects_a_missing_group_column(shots: pd.DataFrame) -> None:
    with pytest.raises(ValidationError, match="not a column"):
        group_train_test_split(shots, group_column="absent")


@pytest.mark.parametrize("test_size", [0.0, 1.0, -0.2, 1.5])
def test_the_split_rejects_an_impossible_test_size(shots: pd.DataFrame, test_size: float) -> None:
    with pytest.raises(ValidationError, match="strictly between 0 and 1"):
        group_train_test_split(shots, test_size=test_size)


def test_the_split_needs_more_than_one_group() -> None:
    single = pd.DataFrame({"match_id": [1, 1, 1], "goal": [True, False, False]})

    with pytest.raises(ValidationError, match="at least 2 distinct"):
        group_train_test_split(single)


# --------------------------------------------------------------------------- #
# Metrics, against sklearn
# --------------------------------------------------------------------------- #


@pytest.fixture
def predictions() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    probability = rng.uniform(0.01, 0.6, size=500)
    outcome = rng.random(500) < probability
    return outcome.astype(int), probability


def test_every_metric_matches_sklearn(predictions: tuple[np.ndarray, np.ndarray]) -> None:
    y_true, y_prob = predictions

    result = evaluate_probabilities(y_true, y_prob)

    assert result.roc_auc == pytest.approx(sk_metrics.roc_auc_score(y_true, y_prob))
    assert result.pr_auc == pytest.approx(sk_metrics.average_precision_score(y_true, y_prob))
    assert result.log_loss == pytest.approx(sk_metrics.log_loss(y_true, y_prob))
    assert result.brier_score == pytest.approx(sk_metrics.brier_score_loss(y_true, y_prob))


def test_prevalence_and_counts_are_reported(predictions: tuple[np.ndarray, np.ndarray]) -> None:
    y_true, y_prob = predictions

    result = evaluate_probabilities(y_true, y_prob)

    assert result.n == 500
    assert result.positives == int(y_true.sum())
    assert result.prevalence == pytest.approx(y_true.mean())


def test_a_perfect_model_scores_perfectly() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.0, 0.0, 1.0, 1.0])

    result = evaluate_probabilities(y_true, y_prob)

    assert result.roc_auc == pytest.approx(1.0)
    assert result.brier_score == pytest.approx(0.0)
    assert result.brier_skill == pytest.approx(1.0)


def test_the_base_rate_model_has_zero_skill() -> None:
    rng = np.random.default_rng(1)
    y_true = (rng.random(400) < 0.1).astype(int)
    y_prob = np.full(400, y_true.mean())

    result = evaluate_probabilities(y_true, y_prob)

    assert result.brier_skill == pytest.approx(0.0, abs=1e-12)
    assert result.log_loss_skill == pytest.approx(0.0, abs=1e-12)
    assert result.roc_auc == pytest.approx(0.5)


def test_calibration_in_the_large_detects_a_biased_model() -> None:
    rng = np.random.default_rng(2)
    y_true = (rng.random(1000) < 0.1).astype(int)
    inflated = np.full(1000, 0.3)

    result = evaluate_probabilities(y_true, inflated)

    assert result.calibration_in_the_large > 0.15
    assert result.brier_skill < 0  # worse than predicting the base rate


def test_ranking_and_calibration_are_measured_separately() -> None:
    """A model can order perfectly and still be badly calibrated."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    well_ordered_but_inflated = np.array([0.60, 0.65, 0.70, 0.90, 0.95, 0.99])

    result = evaluate_probabilities(y_true, well_ordered_but_inflated)

    assert result.roc_auc == pytest.approx(1.0)  # ordering is perfect
    assert result.brier_score > 0.15  # the probabilities are not


def test_compare_models_scores_each_set_of_predictions(
    predictions: tuple[np.ndarray, np.ndarray],
) -> None:
    y_true, y_prob = predictions

    frame = compare_models(
        y_true, {"model": y_prob, "base rate": np.full_like(y_prob, y_true.mean())}
    )

    assert list(frame.index) == ["model", "base rate"]
    assert cell(frame, "model", "roc_auc") > cell(frame, "base rate", "roc_auc")


@pytest.mark.parametrize(
    ("y_true", "y_prob", "message"),
    [
        ([0, 1], [0.5], "same shape"),
        ([], [], "empty sample"),
        ([0, 1], [0.5, 1.5], r"must lie in \[0, 1\]"),
        ([0, 1], [0.5, np.nan], "non-finite"),
        ([0, 2], [0.5, 0.5], "must be binary"),
        ([1, 1], [0.5, 0.5], "only one class"),
    ],
)
def test_malformed_metric_inputs_are_refused(
    y_true: list[float], y_prob: list[float], message: str
) -> None:
    with pytest.raises(MetricError, match=message):
        evaluate_probabilities(y_true, y_prob)


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


def test_the_whole_pipeline_runs_on_a_grouped_split(shots: pd.DataFrame) -> None:
    split = group_train_test_split(shots, test_size=0.25, random_state=SEED)
    assert_no_group_leakage(split)

    baseline = fit(build_baseline_model(random_state=SEED), split.train)
    contextual = fit(build_contextual_model(random_state=SEED), split.train)

    metrics = compare_models(
        split.test["goal"].to_numpy(),
        {
            "baseline": predict_probability(baseline, split.test),
            "contextual": predict_probability(contextual, split.test),
        },
    )

    # On data generated from distance and angle alone, the baseline should already
    # discriminate well, and both models must beat the base rate.
    assert cell(metrics, "baseline", "roc_auc") > 0.7
    assert cell(metrics, "baseline", "brier_skill") > 0
    assert cell(metrics, "contextual", "brier_skill") > 0
