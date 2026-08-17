"""Tests for sample diagnostics, using synthetic data with known properties."""

import math

import numpy as np
import pandas as pd
import pytest

from football_intelligence.statistics import diagnostics
from football_intelligence.statistics.diagnostics import (
    assess_normality,
    describe_by_group,
    describe_sample,
    modified_z_scores,
)

# A sample whose statistics can be checked by hand.
HAND_CHECKABLE = [1.0, 2.0, 3.0, 4.0, 5.0]


# --------------------------------------------------------------------------- #
# Location, spread and shape
# --------------------------------------------------------------------------- #


def test_hand_checkable_sample_matches_arithmetic() -> None:
    summary = describe_sample(HAND_CHECKABLE, name="1..5")

    assert summary.n_total == 5
    assert summary.n_observed == 5
    assert summary.n_missing == 0
    assert summary.mean == pytest.approx(3.0)
    assert summary.median == pytest.approx(3.0)
    # Sum of squared deviations is 10, divided by n-1 = 4.
    assert summary.variance == pytest.approx(2.5)
    assert summary.std == pytest.approx(math.sqrt(2.5))
    assert summary.standard_error == pytest.approx(math.sqrt(2.5) / math.sqrt(5))
    assert summary.q1 == pytest.approx(2.0)
    assert summary.q3 == pytest.approx(4.0)
    assert summary.iqr == pytest.approx(2.0)
    assert summary.mad == pytest.approx(1.0)
    assert summary.minimum == 1.0
    assert summary.maximum == 5.0
    assert summary.skewness == pytest.approx(0.0)


def test_standard_error_shrinks_with_sample_size_while_spread_does_not() -> None:
    rng = np.random.default_rng(11)
    small = describe_sample(rng.normal(0.0, 1.0, size=100))
    large = describe_sample(rng.normal(0.0, 1.0, size=10_000))

    # The observations are equally spread; only the precision of the mean improves.
    assert large.std == pytest.approx(small.std, abs=0.2)
    assert large.standard_error < small.standard_error / 5


def test_normal_sample_recovers_its_parameters() -> None:
    rng = np.random.default_rng(20260816)
    sample = rng.normal(loc=10.0, scale=2.0, size=50_000)

    summary = describe_sample(sample, name="normal")

    assert summary.mean == pytest.approx(10.0, abs=0.05)
    assert summary.std == pytest.approx(2.0, abs=0.05)
    assert summary.median == pytest.approx(10.0, abs=0.05)
    assert summary.skewness == pytest.approx(0.0, abs=0.05)
    assert summary.excess_kurtosis == pytest.approx(0.0, abs=0.05)
    # For a normal distribution the IQR is 1.349 standard deviations.
    assert summary.iqr == pytest.approx(1.349 * 2.0, abs=0.05)


def test_skewness_sign_follows_the_direction_of_the_tail() -> None:
    rng = np.random.default_rng(3)
    right_tailed = rng.exponential(scale=1.0, size=20_000)

    positive = describe_sample(right_tailed)
    negative = describe_sample(-right_tailed)

    # The exponential distribution has a theoretical skewness of exactly 2.
    assert positive.skewness == pytest.approx(2.0, abs=0.15)
    assert negative.skewness == pytest.approx(-2.0, abs=0.15)
    assert positive.mean > positive.median  # right tail drags the mean up
    assert negative.mean < negative.median


def test_excess_kurtosis_is_zero_for_normal_and_positive_for_heavy_tails() -> None:
    rng = np.random.default_rng(5)

    normal = describe_sample(rng.normal(size=40_000))
    heavy = describe_sample(rng.standard_t(df=6, size=40_000))

    assert normal.excess_kurtosis == pytest.approx(0.0, abs=0.1)
    # t with 6 degrees of freedom has a theoretical excess kurtosis of 3.
    assert heavy.excess_kurtosis > 1.5


# --------------------------------------------------------------------------- #
# Missing and malformed values
# --------------------------------------------------------------------------- #


def test_missing_values_are_counted_and_excluded() -> None:
    summary = describe_sample([1.0, 2.0, float("nan"), 4.0], name="with-missing")

    assert summary.n_total == 4
    assert summary.n_observed == 3
    assert summary.n_missing == 1
    assert summary.missing_rate == pytest.approx(0.25)
    assert summary.mean == pytest.approx(7.0 / 3.0)


def test_infinite_values_are_reported_separately_from_missing_ones() -> None:
    summary = describe_sample([1.0, 2.0, float("inf"), float("nan")], name="malformed")

    assert summary.n_observed == 2
    assert summary.n_missing == 1
    assert summary.n_infinite == 1
    assert summary.mean == pytest.approx(1.5)


def test_pandas_missing_values_are_understood() -> None:
    series = pd.Series([1.0, 2.0, None, 4.0], dtype="Float64")

    summary = describe_sample(series, name="nullable")

    assert summary.n_observed == 3
    assert summary.n_missing == 1


def test_an_all_missing_sample_reports_counts_rather_than_failing() -> None:
    summary = describe_sample([float("nan")] * 4, name="empty-ish")

    assert summary.n_total == 4
    assert summary.n_observed == 0
    assert summary.missing_rate == 1.0
    assert math.isnan(summary.mean)
    assert summary.tukey_outliers == 0


def test_an_empty_sample_is_a_caller_error() -> None:
    with pytest.raises(ValueError, match="at least one value"):
        describe_sample([])


def test_a_two_dimensional_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        describe_sample([[1.0, 2.0], [3.0, 4.0]])


def test_statistics_needing_more_data_are_nan_rather_than_wrong() -> None:
    single = describe_sample([42.0])

    assert single.mean == 42.0
    assert math.isnan(single.variance)  # undefined with one observation
    assert math.isnan(single.std)
    assert math.isnan(single.skewness)  # needs three


# --------------------------------------------------------------------------- #
# Outlier indicators
# --------------------------------------------------------------------------- #


def test_tukey_fences_flag_a_planted_outlier() -> None:
    summary = describe_sample([1.0, 2.0, 3.0, 4.0, 100.0])

    assert summary.tukey_outliers == 1
    assert summary.tukey_upper_fence < 100.0


def test_modified_z_score_flags_the_outlier_a_plain_z_score_would_hide() -> None:
    sample = [*(float(value) for value in range(1, 21)), 1000.0]

    scores = modified_z_scores(sample)
    plain_z = (np.asarray(sample) - np.mean(sample)) / np.std(sample, ddof=1)

    # The outlier inflates the mean and standard deviation it is measured against,
    # so the ordinary z-score cannot even reach 5 while the robust score is huge.
    assert abs(scores[-1]) > 50
    assert abs(plain_z[-1]) < 5


def test_modified_z_scores_are_zero_when_more_than_half_the_sample_is_identical() -> None:
    scores = modified_z_scores([5.0, 5.0, 5.0, 5.0, 9.0])

    # The MAD is zero, so no value can be scaled: report no outliers rather than
    # dividing by zero.
    assert np.all(scores == 0.0)


def test_a_constant_sample_has_no_spread_and_no_outliers() -> None:
    summary = describe_sample([7.0] * 10)

    assert summary.variance == pytest.approx(0.0)
    assert summary.iqr == pytest.approx(0.0)
    assert summary.mad == pytest.approx(0.0)
    assert summary.tukey_outliers == 0
    assert summary.modified_z_outliers == 0
    # Standardised moments divide by the spread, so they are undefined here
    # rather than zero.
    assert math.isnan(summary.skewness)
    assert math.isnan(summary.excess_kurtosis)


def test_a_clean_normal_sample_flags_few_tukey_outliers() -> None:
    rng = np.random.default_rng(17)

    summary = describe_sample(rng.normal(size=10_000))

    # Tukey's fences flag roughly 0.7% of normal data by construction.
    assert 0.002 < summary.tukey_outliers / summary.n_observed < 0.02


# --------------------------------------------------------------------------- #
# Quantiles
# --------------------------------------------------------------------------- #


def test_quantiles_are_reported_for_the_requested_probabilities() -> None:
    summary = describe_sample(list(range(101)), quantiles=(0.1, 0.5, 0.9))

    assert set(summary.quantiles) == {0.1, 0.5, 0.9}
    assert summary.quantiles[0.1] == pytest.approx(10.0)
    assert summary.quantiles[0.5] == pytest.approx(50.0)
    assert summary.quantiles[0.9] == pytest.approx(90.0)


# --------------------------------------------------------------------------- #
# Normality descriptions
# --------------------------------------------------------------------------- #


def test_normality_result_offers_no_boolean_verdict() -> None:
    """The absence of a verdict field is the point, so it is asserted."""
    rng = np.random.default_rng(1)
    result = assess_normality(rng.normal(size=200))

    for forbidden in ("is_normal", "normal", "recommended_test", "use_t_test", "passed"):
        assert not hasattr(result, forbidden)


def test_a_normal_sample_is_not_flagged_by_either_test() -> None:
    rng = np.random.default_rng(20260816)

    result = assess_normality(rng.normal(size=200), name="normal")

    assert result.shapiro_p_value is not None
    assert result.dagostino_p_value is not None
    assert result.shapiro_p_value > 0.05
    assert result.dagostino_p_value > 0.05
    assert result.skewness == pytest.approx(0.0, abs=0.3)


def test_a_strongly_skewed_sample_is_flagged_by_both_tests() -> None:
    rng = np.random.default_rng(2)

    result = assess_normality(rng.exponential(size=200), name="exponential")

    assert result.shapiro_p_value is not None
    assert result.dagostino_p_value is not None
    assert result.shapiro_p_value < 1e-6
    assert result.dagostino_p_value < 1e-6
    assert result.skewness > 1.0


def test_the_same_distribution_is_rejected_only_once_the_sample_is_large() -> None:
    """The central objection to p-value-driven test selection, made concrete.

    A Student-t sample with 30 degrees of freedom is normal enough for any
    practical purpose. Its shape does not change with n, but the verdict of a
    normality test does.
    """
    rng = np.random.default_rng(20260816)
    small = assess_normality(rng.standard_t(30, size=200), name="t(30), n=200")
    large = assess_normality(rng.standard_t(30, size=20_000), name="t(30), n=20000")

    assert small.dagostino_p_value is not None
    assert large.dagostino_p_value is not None
    assert small.dagostino_p_value > 0.05  # "normal"
    assert large.dagostino_p_value < 0.001  # "not normal"

    # Meanwhile the quantity that actually describes the departure barely moves,
    # and stays near the theoretical excess kurtosis of 6/(30-4) = 0.23.
    assert abs(small.excess_kurtosis - large.excess_kurtosis) < 1.0
    assert abs(large.excess_kurtosis) < 0.5
    assert abs(large.skewness) < 0.1


def test_shapiro_is_withheld_above_its_reliable_range() -> None:
    rng = np.random.default_rng(4)

    result = assess_normality(rng.normal(size=diagnostics.SHAPIRO_MAX_N + 1))

    assert result.shapiro_p_value is None
    assert result.shapiro_statistic is None
    assert any("Shapiro-Wilk is not reported" in note for note in result.notes)
    # D'Agostino has no such ceiling and is still reported.
    assert result.dagostino_p_value is not None


def test_small_samples_withhold_the_tests_and_say_why() -> None:
    result = assess_normality([1.0, 2.0, 3.0, 4.0, 5.0])

    assert result.dagostino_p_value is None
    assert any("little power" in note for note in result.notes)
    assert any("D'Agostino" in note for note in result.notes)


def test_large_samples_carry_the_sensitivity_warning() -> None:
    rng = np.random.default_rng(6)

    result = assess_normality(rng.normal(size=diagnostics.LARGE_SAMPLE_N + 1))

    assert any("too small to affect inference" in note for note in result.notes)


def test_every_result_states_what_it_does_not_cover() -> None:
    rng = np.random.default_rng(8)

    result = assess_normality(rng.normal(size=100))

    assert any(
        "do not determine which hypothesis test is appropriate" in note for note in result.notes
    )


# --------------------------------------------------------------------------- #
# Grouped description
# --------------------------------------------------------------------------- #


def test_describe_by_group_summarises_each_level_independently() -> None:
    frame = pd.DataFrame(
        {
            "value": [1.0, 2.0, 3.0, 10.0, 20.0],
            "group": ["a", "a", "a", "b", "b"],
        }
    )

    summary = describe_by_group(frame, "value", "group").set_index("group")

    assert summary.loc["a", "n"] == 3
    assert summary.loc["a", "mean"] == pytest.approx(2.0)
    assert summary.loc["b", "n"] == 2
    assert summary.loc["b", "mean"] == pytest.approx(15.0)
    # Ordered by sample size, largest first.
    assert list(summary.index) == ["a", "b"]


def test_describe_by_group_reports_missing_values_per_group() -> None:
    frame = pd.DataFrame({"value": [1.0, float("nan"), 3.0, 4.0], "group": ["a", "a", "b", "b"]})

    summary = describe_by_group(frame, "value", "group").set_index("group")

    assert summary.loc["a", "n"] == 1
    assert summary.loc["a", "missing"] == 1
    assert summary.loc["b", "missing"] == 0


def test_describe_by_group_rejects_unknown_columns() -> None:
    frame = pd.DataFrame({"value": [1.0], "group": ["a"]})

    with pytest.raises(KeyError, match="absent"):
        describe_by_group(frame, "absent", "group")
