"""Tests for the effect-size measures, checked against known values and SciPy."""

import numpy as np
import pytest
from scipy import stats
from scipy.stats.contingency import odds_ratio as scipy_odds_ratio

from football_intelligence.statistics.effect_sizes import (
    EffectSizeError,
    cohens_d,
    cramers_v,
    cramers_v_from_labels,
    hedges_g,
    mean_difference,
    odds_ratio,
    probability_of_superiority,
    rank_biserial_correlation,
    risk_difference,
)

# Keep bootstrap intervals cheap but stable in tests.
RESAMPLES = 800
SEED = 20260822


# --------------------------------------------------------------------------- #
# Mean difference
# --------------------------------------------------------------------------- #


def test_mean_difference_is_plain_arithmetic() -> None:
    result = mean_difference([10.0, 12.0, 14.0], [4.0, 6.0, 8.0], units="yards")

    assert result.estimate == pytest.approx(6.0)
    assert result.units == "yards"
    assert result.null_value == 0.0


def test_mean_difference_interval_matches_the_welch_interval() -> None:
    rng = np.random.default_rng(SEED)
    a = rng.normal(0.0, 1.0, size=60)
    b = rng.normal(0.5, 2.0, size=40)
    expected = stats.ttest_ind(a, b, equal_var=False).confidence_interval(0.95)

    result = mean_difference(a, b)

    assert result.confidence_interval == pytest.approx((expected.low, expected.high))
    assert result.interval_method == "Welch t"


def test_mean_difference_interval_widens_at_a_higher_confidence_level() -> None:
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=50), rng.normal(size=50)

    narrow = mean_difference(a, b, confidence_level=0.80)
    wide = mean_difference(a, b, confidence_level=0.99)

    assert wide.interval_width is not None and narrow.interval_width is not None
    assert wide.interval_width > narrow.interval_width


# --------------------------------------------------------------------------- #
# Cohen's d
# --------------------------------------------------------------------------- #


def test_cohens_d_matches_the_hand_computed_pooled_formula() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
    pooled = np.sqrt(((5 - 1) * a.var(ddof=1) + (5 - 1) * b.var(ddof=1)) / (5 + 5 - 2))
    expected = (a.mean() - b.mean()) / pooled

    result = cohens_d(a, b, n_resamples=RESAMPLES, random_state=SEED)

    assert result.estimate == pytest.approx(expected)
    assert expected == pytest.approx(-2.0 / np.sqrt(2.5))


def test_cohens_d_of_one_standard_deviation_is_one() -> None:
    rng = np.random.default_rng(SEED)
    a = rng.normal(0.0, 1.0, size=20_000)
    b = rng.normal(1.0, 1.0, size=20_000)

    result = cohens_d(a, b, n_resamples=200, random_state=SEED)

    assert result.estimate == pytest.approx(-1.0, abs=0.05)


def test_cohens_d_is_scale_free_while_the_mean_difference_is_not() -> None:
    rng = np.random.default_rng(3)
    a = rng.normal(0.0, 2.0, size=200)
    b = rng.normal(1.0, 2.0, size=200)

    original = cohens_d(a, b, n_resamples=200, random_state=1)
    rescaled = cohens_d(a * 10.0, b * 10.0, n_resamples=200, random_state=1)
    raw = mean_difference(a, b)
    raw_rescaled = mean_difference(a * 10.0, b * 10.0)

    assert rescaled.estimate == pytest.approx(original.estimate)
    assert raw_rescaled.estimate == pytest.approx(raw.estimate * 10.0)


def test_cohens_d_sign_follows_the_argument_order() -> None:
    a, b = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]

    forward = cohens_d(a, b, n_resamples=200, random_state=1)
    reverse = cohens_d(b, a, n_resamples=200, random_state=1)

    assert forward.estimate == pytest.approx(-reverse.estimate)


def test_hedges_correction_shrinks_the_estimate_and_matters_only_when_small() -> None:
    small_a, small_b = [1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]
    rng = np.random.default_rng(5)
    big_a, big_b = rng.normal(0, 1, 500), rng.normal(0.5, 1, 500)

    small_d = cohens_d(small_a, small_b, n_resamples=200, random_state=1)
    small_g = hedges_g(small_a, small_b, n_resamples=200, random_state=1)
    big_d = cohens_d(big_a, big_b, n_resamples=200, random_state=1)
    big_g = hedges_g(big_a, big_b, n_resamples=200, random_state=1)

    assert abs(small_g.estimate) < abs(small_d.estimate)
    # Four per group gives 6 degrees of freedom, where Hedges' J is 0.8686 --
    # a 13% shrinkage. At 500 per group the factor is 0.9996 and irrelevant.
    assert small_g.estimate == pytest.approx(small_d.estimate * 0.8686, rel=0.001)
    assert big_g.estimate == pytest.approx(big_d.estimate, rel=0.002)
    assert "Hedges" in big_g.name


def test_cohens_d_warns_when_groups_are_small_and_uncorrected() -> None:
    result = cohens_d([1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0], n_resamples=200, random_state=1)

    assert any("biased upward" in note for note in result.notes)


def test_cohens_d_is_undefined_without_spread() -> None:
    with pytest.raises(EffectSizeError, match="zero spread"):
        cohens_d([5.0] * 10, [5.0] * 10, n_resamples=50, random_state=1)


def test_cohens_d_bootstrap_interval_brackets_the_estimate() -> None:
    rng = np.random.default_rng(SEED)
    a = rng.normal(0.0, 1.0, size=200)
    b = rng.normal(0.6, 1.0, size=200)

    result = cohens_d(a, b, n_resamples=RESAMPLES, random_state=SEED)

    assert result.confidence_interval is not None
    low, high = result.confidence_interval
    assert low < result.estimate < high
    assert result.interval_excludes_null is True


# --------------------------------------------------------------------------- #
# Rank-based
# --------------------------------------------------------------------------- #


def test_probability_of_superiority_is_the_rescaled_u_statistic() -> None:
    rng = np.random.default_rng(7)
    a, b = rng.normal(size=40), rng.normal(size=60)
    expected = stats.mannwhitneyu(a, b).statistic / (40 * 60)

    result = probability_of_superiority(a, b, n_resamples=RESAMPLES, random_state=SEED)

    assert result.estimate == pytest.approx(expected)
    assert result.null_value == 0.5


def test_probability_of_superiority_reaches_the_extremes() -> None:
    fully_above = probability_of_superiority(
        [10.0, 11.0, 12.0], [1.0, 2.0, 3.0], n_resamples=200, random_state=1
    )
    fully_below = probability_of_superiority(
        [1.0, 2.0, 3.0], [10.0, 11.0, 12.0], n_resamples=200, random_state=1
    )

    assert fully_above.estimate == pytest.approx(1.0)
    assert fully_below.estimate == pytest.approx(0.0)


def test_probability_of_superiority_is_a_half_when_groups_are_identical() -> None:
    rng = np.random.default_rng(11)
    a, b = rng.normal(size=500), rng.normal(size=500)

    result = probability_of_superiority(a, b, n_resamples=400, random_state=1)

    assert result.estimate == pytest.approx(0.5, abs=0.05)
    assert result.interval_excludes_null is False


def test_rank_biserial_is_the_rescaled_probability_of_superiority() -> None:
    rng = np.random.default_rng(13)
    a, b = rng.normal(0, 1, 80), rng.normal(0.7, 1, 70)

    superiority = probability_of_superiority(a, b, n_resamples=200, random_state=1)
    correlation = rank_biserial_correlation(a, b, n_resamples=200, random_state=1)

    assert correlation.estimate == pytest.approx(2 * superiority.estimate - 1)
    assert correlation.null_value == 0.0
    assert -1.0 <= correlation.estimate <= 1.0


def test_rank_based_measures_ignore_a_monotone_rescaling() -> None:
    """They depend only on the ordering, unlike Cohen's d."""
    rng = np.random.default_rng(17)
    a, b = rng.uniform(1, 5, 100), rng.uniform(2, 6, 100)

    plain = probability_of_superiority(a, b, n_resamples=200, random_state=1)
    exponentiated = probability_of_superiority(
        np.exp(a), np.exp(b), n_resamples=200, random_state=1
    )

    assert exponentiated.estimate == pytest.approx(plain.estimate)


# --------------------------------------------------------------------------- #
# Categorical
# --------------------------------------------------------------------------- #


def test_odds_ratio_matches_scipy() -> None:
    table = np.array([[20, 30], [10, 60]])
    expected = scipy_odds_ratio(table)

    result = odds_ratio(table)

    assert result.estimate == pytest.approx(float(expected.statistic))
    assert result.confidence_interval == pytest.approx(tuple(expected.confidence_interval(0.95)))
    assert result.null_value == 1.0


def test_sample_odds_ratio_is_the_cross_product_ratio() -> None:
    result = odds_ratio([[20, 30], [10, 60]], kind="sample")

    assert result.estimate == pytest.approx((20 * 60) / (30 * 10))


def test_an_odds_ratio_of_one_has_an_interval_containing_one() -> None:
    result = odds_ratio([[25, 25], [25, 25]])

    assert result.estimate == pytest.approx(1.0)
    assert result.interval_excludes_null is False


def test_risk_difference_is_the_difference_in_row_rates() -> None:
    result = risk_difference([[20, 30], [10, 60]])

    assert result.estimate == pytest.approx(20 / 50 - 10 / 70)
    assert result.n_a == 50
    assert result.n_b == 70
    assert result.interval_excludes_null is True


def test_risk_difference_and_odds_ratio_disagree_when_the_outcome_is_common() -> None:
    """An odds ratio exaggerates the change in probability for frequent outcomes."""
    table = [[80, 20], [60, 40]]

    ratio = odds_ratio(table, kind="sample")
    difference = risk_difference(table)

    assert ratio.estimate == pytest.approx((80 * 40) / (20 * 60))  # 2.67
    assert difference.estimate == pytest.approx(0.2)  # only 20 percentage points
    assert ratio.estimate > 2.5
    assert any("not a risk ratio" in note for note in ratio.notes)


def test_risk_difference_interval_stays_inside_the_possible_range() -> None:
    result = risk_difference([[1, 9], [0, 10]])

    assert result.confidence_interval is not None
    low, high = result.confidence_interval
    assert -1.0 <= low <= high <= 1.0


def test_cramers_v_matches_the_hand_computed_formula() -> None:
    table = np.array([[90, 10], [10, 90]])
    chi2 = float(stats.chi2_contingency(table, correction=False).statistic)
    expected = np.sqrt(chi2 / (200 * 1))

    result = cramers_v(table)

    assert result.estimate == pytest.approx(expected)
    assert result.estimate == pytest.approx(0.8)
    assert result.confidence_interval is None  # not available from a table alone


def test_cramers_v_is_zero_for_a_perfectly_independent_table() -> None:
    result = cramers_v([[50, 50], [100, 100]])

    assert result.estimate == pytest.approx(0.0, abs=1e-12)


def test_cramers_v_does_not_grow_with_the_sample_size_but_chi_square_does() -> None:
    small = np.array([[30, 20], [20, 30]])
    large = small * 10

    chi2_small = float(stats.chi2_contingency(small, correction=False).statistic)
    chi2_large = float(stats.chi2_contingency(large, correction=False).statistic)

    assert chi2_large == pytest.approx(chi2_small * 10)
    assert cramers_v(large).estimate == pytest.approx(cramers_v(small).estimate)


def test_the_bias_correction_shrinks_cramers_v_in_a_small_table() -> None:
    table = [[6, 3], [3, 6]]

    plain = cramers_v(table)
    corrected = cramers_v(table, bias_correction=True)

    assert corrected.estimate < plain.estimate
    assert "bias-corrected" in corrected.name


def test_cramers_v_from_labels_agrees_with_the_table_version_and_adds_an_interval() -> None:
    rows = ["a"] * 100 + ["b"] * 100
    columns = ["x"] * 80 + ["y"] * 20 + ["x"] * 20 + ["y"] * 80

    from_labels = cramers_v_from_labels(rows, columns, n_resamples=RESAMPLES, random_state=SEED)
    from_table = cramers_v([[80, 20], [20, 80]])

    assert from_labels.estimate == pytest.approx(from_table.estimate)
    assert from_labels.confidence_interval is not None
    low, high = from_labels.confidence_interval
    assert 0.0 <= low < from_labels.estimate < high <= 1.0


def test_cramers_v_from_labels_rejects_mismatched_lengths() -> None:
    with pytest.raises(EffectSizeError, match="same length"):
        cramers_v_from_labels(["a", "b", "c"], ["x", "y"])


# --------------------------------------------------------------------------- #
# Shared behaviour
# --------------------------------------------------------------------------- #


def test_a_small_p_value_can_accompany_a_negligible_effect() -> None:
    """The reason this module exists."""
    rng = np.random.default_rng(SEED)
    a = rng.normal(0.0, 1.0, size=100_000)
    b = rng.normal(0.02, 1.0, size=100_000)

    p_value = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
    effect = cohens_d(a, b, n_resamples=200, random_state=SEED)

    assert p_value < 0.01
    assert abs(effect.estimate) < 0.05  # two hundredths of a standard deviation


def test_a_meaningful_estimate_can_come_with_a_useless_interval() -> None:
    """A large point estimate from a small sample says very little."""
    result = odds_ratio([[4, 1], [1, 4]])

    assert result.estimate > 5.0
    assert result.confidence_interval is not None
    low, high = result.confidence_interval
    assert low < 1.0 < high  # the interval still contains "no effect"
    assert result.interval_excludes_null is False


@pytest.mark.parametrize(
    "bad_table",
    [[[1, 2, 3], [4, 5, 6]], [[1, 2]], [[-1, 2], [3, 4]], [[0, 0], [0, 0]]],
)
def test_malformed_tables_are_refused(bad_table: list[list[int]]) -> None:
    with pytest.raises(EffectSizeError):
        odds_ratio(bad_table)


def test_a_group_too_small_is_refused() -> None:
    with pytest.raises(EffectSizeError, match="at least 2 are needed"):
        mean_difference([1.0], [1.0, 2.0, 3.0])


def test_non_finite_values_are_dropped() -> None:
    result = mean_difference([1.0, 2.0, 3.0, float("nan")], [4.0, 5.0, float("inf"), 6.0])

    assert result.n_a == 3
    assert result.n_b == 3


def test_results_render_as_readable_text() -> None:
    text = str(mean_difference([10.0, 12.0], [4.0, 6.0], units="yards"))

    assert "mean difference" in text
    assert "yards" in text
    assert "no effect = 0" in text


def test_no_measure_returns_a_qualitative_size_label() -> None:
    """Small/medium/large labels are conventions, so the module never asserts one."""
    results = [
        mean_difference([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
        cohens_d([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], n_resamples=100, random_state=1),
        odds_ratio([[10, 20], [30, 40]]),
        cramers_v([[10, 20], [30, 40]]),
    ]

    for result in results:
        for forbidden in ("magnitude_label", "size", "label", "is_large", "interpretation_label"):
            assert not hasattr(result, forbidden)
