"""Tests for the two-sample and contingency hypothesis tests.

Every wrapper is checked against a direct SciPy call, so the wrappers cannot
silently diverge from the algorithms they delegate to. The remaining tests
verify behaviour on synthetic data with known properties.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from football_intelligence.statistics.tests import (
    Alternative,
    HypothesisTestResult,
    StatisticalTestError,
    chi_square_independence_test,
    fisher_exact_test,
    group_samples,
    kolmogorov_smirnov_test,
    kruskal_wallis_test,
    mann_whitney_u_test,
    one_way_anova,
    paired_t_test,
    student_t_test,
    welch_anova,
    welch_t_test,
    wilcoxon_signed_rank_test,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260817)


# --------------------------------------------------------------------------- #
# Agreement with SciPy
# --------------------------------------------------------------------------- #


def test_student_t_test_matches_scipy(rng: np.random.Generator) -> None:
    a = rng.normal(0.0, 1.0, size=60)
    b = rng.normal(0.5, 1.0, size=70)
    expected = stats.ttest_ind(a, b, equal_var=True)

    result = student_t_test(a, b)

    assert result.statistic == pytest.approx(float(expected.statistic))
    assert result.p_value == pytest.approx(float(expected.pvalue))
    assert result.degrees_of_freedom == pytest.approx(float(expected.df))


def test_welch_t_test_matches_scipy(rng: np.random.Generator) -> None:
    a = rng.normal(0.0, 1.0, size=40)
    b = rng.normal(0.5, 3.0, size=90)
    expected = stats.ttest_ind(a, b, equal_var=False)

    result = welch_t_test(a, b)

    assert result.statistic == pytest.approx(float(expected.statistic))
    assert result.p_value == pytest.approx(float(expected.pvalue))
    assert result.degrees_of_freedom is not None
    assert result.degrees_of_freedom == pytest.approx(float(expected.df))
    # Welch's degrees of freedom are fractional in general.
    assert result.degrees_of_freedom != float(int(result.degrees_of_freedom))


def test_mann_whitney_matches_scipy(rng: np.random.Generator) -> None:
    a = rng.normal(0.0, 1.0, size=50)
    b = rng.normal(0.4, 1.0, size=60)
    expected = stats.mannwhitneyu(a, b, alternative="two-sided")

    result = mann_whitney_u_test(a, b)

    assert result.statistic == pytest.approx(float(expected.statistic))
    assert result.p_value == pytest.approx(float(expected.pvalue))


def test_kolmogorov_smirnov_matches_scipy(rng: np.random.Generator) -> None:
    a = rng.normal(size=80)
    b = rng.normal(0.6, size=95)
    expected = stats.ks_2samp(a, b)

    result = kolmogorov_smirnov_test(a, b)

    assert result.statistic == pytest.approx(float(expected.statistic))
    assert result.p_value == pytest.approx(float(expected.pvalue))


def test_chi_square_matches_scipy() -> None:
    table = [[40, 60], [30, 90]]
    expected = stats.chi2_contingency(table, correction=True)

    result = chi_square_independence_test(table)

    assert result.statistic == pytest.approx(float(expected.statistic))
    assert result.p_value == pytest.approx(float(expected.pvalue))
    assert result.degrees_of_freedom == pytest.approx(float(expected.dof))


def test_fisher_exact_matches_scipy() -> None:
    table = np.array([[8, 2], [1, 9]])
    expected = stats.fisher_exact(table)

    result = fisher_exact_test(table)

    assert result.statistic == pytest.approx(float(expected.statistic))
    assert result.p_value == pytest.approx(float(expected.pvalue))
    assert result.estimate == pytest.approx(float(expected.statistic))


@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
def test_one_sided_alternatives_are_passed_through(
    alternative: Alternative, rng: np.random.Generator
) -> None:
    a = rng.normal(0.0, 1.0, size=40)
    b = rng.normal(0.5, 1.0, size=40)

    welch = welch_t_test(a, b, alternative=alternative)
    mwu = mann_whitney_u_test(a, b, alternative=alternative)

    assert welch.p_value == pytest.approx(
        float(stats.ttest_ind(a, b, equal_var=False, alternative=alternative).pvalue)
    )
    assert mwu.p_value == pytest.approx(
        float(stats.mannwhitneyu(a, b, alternative=alternative).pvalue)
    )
    assert welch.alternative == alternative


# --------------------------------------------------------------------------- #
# Behaviour on synthetic data with known properties
# --------------------------------------------------------------------------- #


def test_identical_populations_rarely_produce_small_p_values(
    rng: np.random.Generator,
) -> None:
    """A large p-value is the expected outcome here; it never proves the null."""
    a = rng.normal(0.0, 1.0, size=500)
    b = rng.normal(0.0, 1.0, size=500)

    assert welch_t_test(a, b).p_value > 0.05
    assert mann_whitney_u_test(a, b).p_value > 0.05
    assert kolmogorov_smirnov_test(a, b).p_value > 0.05


def test_a_clear_shift_is_detected_by_the_location_tests(rng: np.random.Generator) -> None:
    a = rng.normal(0.0, 1.0, size=200)
    b = rng.normal(1.5, 1.0, size=200)

    welch = welch_t_test(a, b)
    mwu = mann_whitney_u_test(a, b)

    assert welch.p_value < 1e-10
    assert mwu.p_value < 1e-10
    assert welch.estimate is not None
    assert welch.estimate == pytest.approx(-1.5, abs=0.25)


def test_the_mean_difference_and_its_interval_describe_the_size_of_the_effect(
    rng: np.random.Generator,
) -> None:
    a = rng.normal(10.0, 1.0, size=400)
    b = rng.normal(8.0, 1.0, size=400)

    result = welch_t_test(a, b)

    assert result.estimate == pytest.approx(2.0, abs=0.2)
    assert result.confidence_interval is not None
    low, high = result.confidence_interval
    assert low < 2.0 < high
    assert result.confidence_level == 0.95


def test_a_tiny_effect_in_a_huge_sample_is_significant_but_trivial(
    rng: np.random.Generator,
) -> None:
    """The reason a p-value must never stand in for practical importance."""
    a = rng.normal(0.0, 1.0, size=200_000)
    b = rng.normal(0.02, 1.0, size=200_000)

    result = welch_t_test(a, b)

    assert result.p_value < 0.01  # "significant"
    assert result.estimate is not None
    assert abs(result.estimate) < 0.05  # 2% of a standard deviation


def test_student_and_welch_disagree_when_variances_and_group_sizes_differ(
    rng: np.random.Generator,
) -> None:
    small_wide = rng.normal(0.0, 4.0, size=15)
    large_narrow = rng.normal(0.0, 1.0, size=200)

    student = student_t_test(small_wide, large_narrow)
    welch = welch_t_test(small_wide, large_narrow)

    assert student.p_value != pytest.approx(welch.p_value)
    assert any("equal-variance assumption" in item for item in student.warnings)
    assert any("least robust" in item for item in student.warnings)
    assert not welch.warnings or all("equal-variance" not in w for w in welch.warnings)


def test_student_does_not_warn_when_variances_match(rng: np.random.Generator) -> None:
    a = rng.normal(0.0, 1.0, size=100)
    b = rng.normal(0.3, 1.0, size=100)

    result = student_t_test(a, b)

    assert all("equal-variance assumption" not in item for item in result.warnings)


# --------------------------------------------------------------------------- #
# Mann-Whitney is not a test of medians
# --------------------------------------------------------------------------- #


def test_mann_whitney_rejects_two_samples_with_the_same_median(
    rng: np.random.Generator,
) -> None:
    """The concrete counterexample to "Mann-Whitney compares medians".

    An exponential sample and a normal sample can share a median and still be
    ordered very differently, and the test picks that up decisively.
    """
    a = rng.exponential(scale=1.0, size=4000)
    b = rng.normal(loc=np.log(2.0), scale=1.0, size=4000)

    assert np.median(a) == pytest.approx(np.median(b), abs=0.05)

    result = mann_whitney_u_test(a, b)

    assert result.p_value < 1e-10
    assert result.estimate is not None
    assert result.estimate > 0.53  # probability of superiority, not 0.5


def test_probability_of_superiority_is_the_rescaled_u_statistic(
    rng: np.random.Generator,
) -> None:
    a = rng.normal(0.0, 1.0, size=120)
    b = rng.normal(0.0, 1.0, size=80)

    result = mann_whitney_u_test(a, b)

    assert result.estimate == pytest.approx(result.statistic / (120 * 80))
    assert result.estimate == pytest.approx(0.5, abs=0.1)


def test_probability_of_superiority_reaches_one_for_separated_samples() -> None:
    result = mann_whitney_u_test([10.0, 11.0, 12.0], [1.0, 2.0, 3.0])

    assert result.estimate == pytest.approx(1.0)


def test_mann_whitney_metadata_does_not_describe_a_test_of_medians() -> None:
    result = mann_whitney_u_test([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])

    assert "stochastically equal" in result.null_hypothesis
    assert "median" not in result.null_hypothesis
    assert "probability of superiority" in result.estimand
    assert "median" not in result.estimand
    # The caveat about medians appears only as an explicit extra assumption.
    assert any("same shape and spread" in item for item in result.assumptions)


def test_mann_whitney_warns_about_ties() -> None:
    result = mann_whitney_u_test([1.0, 2.0, 2.0, 3.0], [2.0, 3.0, 4.0, 5.0])

    assert any("tied value" in item for item in result.warnings)


# --------------------------------------------------------------------------- #
# Kolmogorov-Smirnov sees what a mean comparison cannot
# --------------------------------------------------------------------------- #


def test_ks_detects_a_shape_difference_that_leaves_the_mean_unchanged(
    rng: np.random.Generator,
) -> None:
    normal = rng.normal(0.0, 1.0, size=3000)
    uniform = rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=3000)  # same mean and sd

    welch = welch_t_test(normal, uniform)
    ks = kolmogorov_smirnov_test(normal, uniform)

    assert welch.p_value > 0.1  # the means really are the same
    assert ks.p_value < 1e-4  # the distributions really are not
    assert ks.estimate is not None
    assert ks.estimate > 0.05


def test_ks_statistic_is_the_largest_gap_between_the_two_cdfs() -> None:
    a = [0.0, 1.0, 2.0, 3.0]
    b = [10.0, 11.0, 12.0, 13.0]

    result = kolmogorov_smirnov_test(a, b)

    # Completely separated samples: the CDFs differ by 1 at the widest point.
    assert result.statistic == pytest.approx(1.0)
    assert result.estimate == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Contingency tables
# --------------------------------------------------------------------------- #


def test_chi_square_detects_a_planted_association() -> None:
    table = [[90, 10], [10, 90]]

    result = chi_square_independence_test(table)

    assert result.p_value < 1e-20
    assert result.degrees_of_freedom == 1
    assert result.n_total == 200
    assert result.table_shape == (2, 2)


def test_chi_square_on_an_independent_table_gives_a_large_p_value() -> None:
    # Rows are exact multiples of one another, so the margins explain everything.
    table = [[50, 50], [100, 100]]

    result = chi_square_independence_test(table)

    assert result.p_value > 0.9
    assert result.statistic == pytest.approx(0.0, abs=1e-9)


def test_chi_square_warns_about_small_expected_counts() -> None:
    table = [[1, 2], [3, 1]]

    result = chi_square_independence_test(table)

    assert any("expected count below" in item for item in result.warnings)


def test_chi_square_accepts_a_crosstab() -> None:
    frame = pd.DataFrame(
        {
            "body_part": ["Head"] * 50 + ["Foot"] * 50,
            "goal": [True] * 20 + [False] * 30 + [True] * 5 + [False] * 45,
        }
    )
    table = pd.crosstab(frame["body_part"], frame["goal"])

    result = chi_square_independence_test(table)

    assert result.n_total == 100
    assert result.p_value < 0.01


def test_chi_square_reports_that_the_statistic_is_not_an_effect_size() -> None:
    result = chi_square_independence_test([[90, 10], [10, 90]])

    assert result.estimate is None
    assert any("not an effect size" in item for item in result.notes)


def test_fisher_agrees_with_chi_square_on_a_large_clear_table() -> None:
    table = [[200, 800], [400, 600]]

    fisher = fisher_exact_test(table)
    chi_square = chi_square_independence_test(table)

    # Both should be decisive; the exact test is not required at this size.
    assert fisher.p_value < 1e-20
    assert chi_square.p_value < 1e-20


def test_fisher_odds_ratio_matches_the_hand_computed_value() -> None:
    # A table with an odds ratio of exactly 1: the conditional MLE is 1 too.
    result = fisher_exact_test([[10, 10], [10, 10]])

    assert result.estimate == pytest.approx(1.0)
    assert result.p_value > 0.9


def test_fisher_handles_an_empty_cell_and_says_so() -> None:
    result = fisher_exact_test([[10, 0], [0, 10]])

    assert result.p_value < 0.001
    assert any("empty cell" in item for item in result.warnings)


def test_fisher_rejects_tables_larger_than_two_by_two() -> None:
    with pytest.raises(StatisticalTestError, match="2x2 tables only"):
        fisher_exact_test([[1, 2], [3, 4], [5, 6]])


@pytest.mark.parametrize(
    ("table", "message"),
    [
        ([1, 2, 3], "two-dimensional"),
        ([[1, 2]], "at least 2x2"),
        ([[-1, 2], [3, 4]], "negative counts"),
        ([[0.5, 1.5], [2.0, 3.0]], "counts, not fractions"),
        ([[0, 0], [0, 0]], "empty"),
    ],
)
def test_malformed_contingency_tables_are_rejected(table: object, message: str) -> None:
    with pytest.raises(StatisticalTestError, match=message):
        chi_square_independence_test(table)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Missing data, small samples and result metadata
# --------------------------------------------------------------------------- #


def test_non_finite_values_are_dropped_and_reported() -> None:
    a = [1.0, 2.0, 3.0, float("nan"), float("inf")]
    b = [4.0, 5.0, 6.0]

    result = welch_t_test(a, b)

    assert result.n_a == 3
    assert result.n_b == 3
    assert any("non-finite" in item for item in result.warnings)


def test_a_sample_too_small_to_test_fails_clearly() -> None:
    with pytest.raises(StatisticalTestError, match="at least 2 are needed"):
        welch_t_test([1.0], [1.0, 2.0, 3.0])


def test_small_skewed_groups_get_a_caveat_about_the_approximation() -> None:
    skewed = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 9.0, 12.0]
    other = [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7]

    result = welch_t_test(skewed, other, label_a="skewed")

    assert any("central limit theorem does little work" in item for item in result.warnings)


def test_every_result_carries_the_clustering_assumption(rng: np.random.Generator) -> None:
    a = rng.normal(size=30)
    b = rng.normal(size=30)
    results = [
        student_t_test(a, b),
        welch_t_test(a, b),
        mann_whitney_u_test(a, b),
        kolmogorov_smirnov_test(a, b),
        chi_square_independence_test([[10, 20], [30, 40]]),
        fisher_exact_test([[10, 20], [30, 40]]),
    ]

    for result in results:
        assert any("clustered within players" in item for item in result.assumptions)


def test_results_expose_no_significance_verdict(rng: np.random.Generator) -> None:
    """Thresholding is a reporting decision, so the result object refuses to make it."""
    result = welch_t_test(rng.normal(size=30), rng.normal(size=30))

    for forbidden in ("significant", "is_significant", "reject", "alpha", "conclusion"):
        assert not hasattr(result, forbidden)


def test_interpretation_states_what_a_p_value_is_not(rng: np.random.Generator) -> None:
    result = welch_t_test(rng.normal(size=30), rng.normal(size=30))

    assert "not the probability that the null is true" in result.interpretation
    assert "large value is not evidence for it" in result.interpretation


def test_results_render_as_readable_text(rng: np.random.Generator) -> None:
    text = str(welch_t_test(rng.normal(size=30), rng.normal(size=30), label_a="X", label_b="Y"))

    assert "Welch's independent t-test" in text
    assert "H0" in text
    assert "mean difference" in text
    assert "95% CI" in text


def test_labels_appear_in_the_hypothesis_metadata() -> None:
    result = welch_t_test([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], label_a="Spain", label_b="Italy")

    assert "Spain" in result.null_hypothesis
    assert "Italy" in result.null_hypothesis
    assert isinstance(result, HypothesisTestResult)


# --------------------------------------------------------------------------- #
# Paired designs
# --------------------------------------------------------------------------- #


def test_paired_t_test_matches_scipy(rng: np.random.Generator) -> None:
    before = rng.normal(10.0, 2.0, size=40)
    after = before + rng.normal(0.5, 1.0, size=40)
    expected = stats.ttest_rel(before, after)

    result = paired_t_test(before, after)

    assert result.statistic == pytest.approx(float(expected.statistic))
    assert result.p_value == pytest.approx(float(expected.pvalue))
    assert result.degrees_of_freedom == 39  # pairs - 1
    assert result.n_pairs == 40
    assert result.design == "paired samples"


def test_wilcoxon_matches_scipy(rng: np.random.Generator) -> None:
    before = rng.normal(10.0, 2.0, size=30)
    after = before + rng.normal(0.4, 1.0, size=30)
    expected = stats.wilcoxon(before, after)

    result = wilcoxon_signed_rank_test(before, after)

    assert result.statistic == pytest.approx(float(expected.statistic))
    assert result.p_value == pytest.approx(float(expected.pvalue))
    assert result.design == "paired samples"


def test_pairing_recovers_an_effect_that_an_independent_test_misses(
    rng: np.random.Generator,
) -> None:
    """Why the design matters: the same numbers, analysed two ways.

    Units differ hugely from each other but each improves by a consistent small
    amount. The paired analysis sees the improvement; the independent analysis
    drowns it in between-unit variation.
    """
    unit_level = rng.normal(50.0, 15.0, size=40)
    before = unit_level + rng.normal(0.0, 1.0, size=40)
    after = unit_level + 2.0 + rng.normal(0.0, 1.0, size=40)

    paired = paired_t_test(before, after)
    independent = welch_t_test(before, after)

    assert paired.p_value < 1e-9
    assert independent.p_value > 0.4
    # Both estimate the same difference; only the precision differs.
    assert paired.estimate == pytest.approx(-2.0, abs=0.5)
    assert independent.estimate == pytest.approx(-2.0, abs=0.5)


def test_a_paired_test_refuses_samples_that_cannot_be_paired() -> None:
    with pytest.raises(StatisticalTestError, match="one-to-one correspondence"):
        paired_t_test([1.0, 2.0, 3.0], [1.0, 2.0])


def test_incomplete_pairs_are_dropped_together_and_reported() -> None:
    before = [1.0, 2.0, 3.0, float("nan"), 5.0]
    after = [1.5, 2.9, float("nan"), 4.5, 5.2]

    result = paired_t_test(before, after)

    # Two pairs lose one side each, so three complete pairs remain.
    assert result.n_pairs == 3
    assert any("incomplete pair" in item for item in result.warnings)


def test_paired_t_test_reports_the_mean_difference_with_an_interval() -> None:
    before = [10.0, 12.0, 14.0, 16.0, 18.0]
    after = [11.0, 13.5, 14.5, 17.5, 19.0]

    result = paired_t_test(before, after)

    # Differences are -1.0, -1.5, -0.5, -1.5, -1.0: mean -1.1.
    assert result.estimate == pytest.approx(-1.1)
    assert result.confidence_interval is not None
    low, high = result.confidence_interval
    assert low < -1.1 < high


@pytest.mark.filterwarnings("ignore:Precision loss occurred:RuntimeWarning")
def test_identical_within_pair_differences_are_flagged_rather_than_celebrated() -> None:
    """A degenerate case that would otherwise report an infinite t and p = 0.

    SciPy warns about the zero variance too; the point of the test is that our
    result carries the explanation rather than an unexplained p of 0.
    """
    before = [10.0, 12.0, 14.0, 16.0, 18.0]
    after = [11.0, 13.0, 15.0, 17.0, 19.0]

    result = paired_t_test(before, after)

    assert result.estimate == pytest.approx(-1.0)
    assert any("arithmetic, not evidence" in item for item in result.warnings)


def test_wilcoxon_reports_dropped_zero_differences() -> None:
    before = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    after = [1.0, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]

    result = wilcoxon_signed_rank_test(before, after)

    assert any("difference of exactly zero" in item for item in result.warnings)
    assert result.n_pairs == 8


def test_wilcoxon_metadata_does_not_describe_a_test_of_medians() -> None:
    result = wilcoxon_signed_rank_test([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 5.0, 9.0])

    assert "symmetrically about zero" in result.null_hypothesis
    assert "median" not in result.null_hypothesis
    assert any(
        "requires the difference distribution to be symmetric" in a for a in result.assumptions
    )


def test_wilcoxon_alternative_is_stated_in_signed_rank_terms_not_as_a_mean() -> None:
    """H0 and H1 must describe the same quantity: this test is not about a mean."""
    two_sided = wilcoxon_signed_rank_test([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 5.0, 9.0])
    greater = wilcoxon_signed_rank_test(
        [1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 5.0, 9.0], alternative="greater"
    )

    assert "not distributed symmetrically about zero" in two_sided.alternative_hypothesis
    assert "mean" not in two_sided.alternative_hypothesis
    assert "median" not in two_sided.alternative_hypothesis
    assert "tends to be larger" in greater.alternative_hypothesis
    assert "on average" not in greater.alternative_hypothesis

    # The paired t-test, by contrast, really is a statement about a mean.
    paired = paired_t_test([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 5.0, 9.0])
    assert "mean within-pair difference" in paired.alternative_hypothesis


def test_paired_results_state_the_effective_sample_size() -> None:
    result = paired_t_test([1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 5.0, 6.0])

    assert any("effective sample size is 4 pairs" in note for note in result.notes)


# --------------------------------------------------------------------------- #
# Multi-group designs
# --------------------------------------------------------------------------- #


@pytest.fixture
def three_groups(rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        "a": rng.normal(10.0, 2.0, size=50),
        "b": rng.normal(10.0, 2.0, size=60),
        "c": rng.normal(13.0, 2.0, size=55),
    }


def test_one_way_anova_matches_scipy(three_groups: dict[str, np.ndarray]) -> None:
    expected = stats.f_oneway(*three_groups.values())

    result = one_way_anova(three_groups)

    assert result.statistic == pytest.approx(float(expected.statistic))
    assert result.p_value == pytest.approx(float(expected.pvalue))
    assert result.design == "independent multi-group"
    assert result.group_names == ("a", "b", "c")
    assert result.group_sizes == (50, 60, 55)


def test_welch_anova_matches_statsmodels(three_groups: dict[str, np.ndarray]) -> None:
    from statsmodels.stats.oneway import anova_oneway

    expected = anova_oneway(list(three_groups.values()), use_var="unequal")

    result = welch_anova(three_groups)

    assert result.statistic == pytest.approx(float(expected.statistic))
    assert result.p_value == pytest.approx(float(expected.pvalue))


def test_kruskal_wallis_matches_scipy(three_groups: dict[str, np.ndarray]) -> None:
    expected = stats.kruskal(*three_groups.values())

    result = kruskal_wallis_test(three_groups)

    assert result.statistic == pytest.approx(float(expected.statistic))
    assert result.p_value == pytest.approx(float(expected.pvalue))


def test_anova_on_two_groups_reproduces_the_student_t_test(rng: np.random.Generator) -> None:
    """F = t^2 for two groups, which is a useful check that both are wired correctly."""
    a = rng.normal(0.0, 1.0, size=40)
    b = rng.normal(0.6, 1.0, size=40)

    anova = one_way_anova({"a": a, "b": b})
    t_test = student_t_test(a, b)

    assert anova.statistic == pytest.approx(t_test.statistic**2)
    assert anova.p_value == pytest.approx(t_test.p_value)
    assert any("equivalent to a t-test" in item for item in anova.warnings)


def test_identical_groups_give_a_large_p_value(rng: np.random.Generator) -> None:
    groups = {name: rng.normal(5.0, 1.0, size=80) for name in ("a", "b", "c", "d")}

    assert one_way_anova(groups).p_value > 0.05
    assert welch_anova(groups).p_value > 0.05
    assert kruskal_wallis_test(groups).p_value > 0.05


def test_one_shifted_group_is_detected(rng: np.random.Generator) -> None:
    groups = {
        "a": rng.normal(0.0, 1.0, size=60),
        "b": rng.normal(0.0, 1.0, size=60),
        "c": rng.normal(2.0, 1.0, size=60),
    }

    assert one_way_anova(groups).p_value < 1e-10
    assert welch_anova(groups).p_value < 1e-10
    assert kruskal_wallis_test(groups).p_value < 1e-10


def test_unequal_variances_are_flagged_and_change_the_answer(
    rng: np.random.Generator,
) -> None:
    groups = {
        "wide": rng.normal(0.0, 6.0, size=30),
        "narrow": rng.normal(0.0, 1.0, size=300),
        "mid": rng.normal(1.0, 3.0, size=100),
    }

    classic = one_way_anova(groups)
    welch = welch_anova(groups)

    assert any("Group variances differ by a factor" in item for item in classic.warnings)
    assert any("unbalanced" in item for item in classic.warnings)
    assert classic.statistic != pytest.approx(welch.statistic)
    # Welch does not make the assumption, so it does not warn about it.
    assert all("equal-variance assumption" not in item for item in welch.warnings)


def test_kruskal_metadata_does_not_describe_a_test_of_medians(
    three_groups: dict[str, np.ndarray],
) -> None:
    result = kruskal_wallis_test(three_groups)

    assert "stochastically equal" in result.null_hypothesis
    assert "median" not in result.null_hypothesis
    assert any("same shape and spread" in item for item in result.assumptions)


def test_multi_group_results_warn_that_f_is_not_an_effect_size(
    three_groups: dict[str, np.ndarray],
) -> None:
    for result in (one_way_anova(three_groups), welch_anova(three_groups)):
        assert result.estimate is None
        assert any("not an effect size" in note for note in result.notes)
        assert any("not which" in note for note in result.notes)


def test_multi_group_results_carry_the_clustering_assumption(
    three_groups: dict[str, np.ndarray],
) -> None:
    for result in (
        one_way_anova(three_groups),
        welch_anova(three_groups),
        kruskal_wallis_test(three_groups),
    ):
        assert any("one player contributes many" in item for item in result.assumptions)


def test_too_few_groups_is_an_error() -> None:
    with pytest.raises(StatisticalTestError, match="At least 2 groups"):
        one_way_anova({"only": [1.0, 2.0, 3.0]})


def test_a_group_too_small_to_analyse_is_an_error() -> None:
    with pytest.raises(StatisticalTestError, match="at least 2 are needed"):
        one_way_anova({"a": [1.0, 2.0, 3.0], "tiny": [5.0]})


def test_group_samples_splits_a_frame_largest_group_first() -> None:
    frame = pd.DataFrame(
        {
            "value": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 40.0],
            "group": ["a", "a", "a", "b", "b", "b", "b"],
        }
    )

    groups = group_samples(frame, "value", "group")

    assert list(groups) == ["b", "a"]
    assert groups["a"].tolist() == [1.0, 2.0, 3.0]
    assert groups["b"].size == 4


def test_group_samples_rejects_unknown_columns() -> None:
    frame = pd.DataFrame({"value": [1.0], "group": ["a"]})

    with pytest.raises(KeyError, match="missing"):
        group_samples(frame, "missing", "group")
