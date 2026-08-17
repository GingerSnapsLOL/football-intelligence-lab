"""Tests for the permutation tests.

The exact path is checked against a labelling count worked out by hand, and the
Monte Carlo path against SciPy's own permutation test.
"""

import math

import numpy as np
import pytest
from scipy import stats

from football_intelligence.statistics.permutation import (
    MAX_EXACT_PERMUTATIONS,
    Alternative,
    PermutationError,
    clustered_permutation_test,
    mean_difference,
    median_difference,
    permutation_test,
    welch_t_statistic,
)

# --------------------------------------------------------------------------- #
# A small example whose permutation distribution can be checked by hand
# --------------------------------------------------------------------------- #


def test_exact_p_value_matches_a_hand_computed_labelling_count() -> None:
    """Six values split three and three: 20 labellings, two of them extreme.

    The groups {1, 2, 3} and {10, 11, 12} are as separated as the pooled data
    allows, so |mean difference| = 9 is reached only by the observed labelling
    and its mirror image. The two-sided p-value must therefore be exactly
    2 / 20 = 0.1, and no amount of data can make it smaller.
    """
    a = [1.0, 2.0, 3.0]
    b = [10.0, 11.0, 12.0]

    result = permutation_test(a, b, method="exact")

    assert result.method == "exact"
    assert result.n_permutations == math.comb(6, 3) == 20
    assert result.observed == pytest.approx(-9.0)
    assert result.p_value == pytest.approx(2 / 20)


def test_the_one_sided_version_of_the_hand_computed_example() -> None:
    a = [1.0, 2.0, 3.0]
    b = [10.0, 11.0, 12.0]

    result = permutation_test(a, b, alternative="less", method="exact")

    # Only the observed labelling reaches a difference of -9.
    assert result.p_value == pytest.approx(1 / 20)


def test_the_null_distribution_of_the_hand_computed_example_is_symmetric() -> None:
    result = permutation_test([1.0, 2.0, 3.0], [10.0, 11.0, 12.0], method="exact")
    null = np.sort(result.null_distribution)

    assert null.size == 20
    assert null[0] == pytest.approx(-9.0)
    assert null[-1] == pytest.approx(9.0)
    # Every labelling has a mirror image with the opposite sign.
    np.testing.assert_allclose(null, -null[::-1], atol=1e-12)


def test_exact_enumeration_is_used_automatically_for_small_samples() -> None:
    result = permutation_test([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0])

    assert result.method == "exact"
    assert result.n_permutations == math.comb(7, 4)


def test_a_tiny_design_warns_that_it_cannot_produce_a_small_p_value() -> None:
    result = permutation_test([1.0, 2.0], [8.0, 9.0], method="exact")

    # Six labellings: the smallest two-sided p-value is 2/6 = 0.33.
    assert result.p_value >= 1 / 3
    assert any("smallest attainable p-value" in item for item in result.warnings)


# --------------------------------------------------------------------------- #
# Agreement with SciPy
# --------------------------------------------------------------------------- #


def test_monte_carlo_p_value_matches_scipy() -> None:
    rng = np.random.default_rng(20260819)
    a = rng.normal(0.0, 1.0, size=60)
    b = rng.normal(0.5, 1.0, size=70)

    ours = permutation_test(a, b, n_permutations=20_000, random_state=1, method="monte-carlo")
    reference = stats.permutation_test(
        (a, b),
        lambda x, y, axis=0: np.mean(x, axis=axis) - np.mean(y, axis=axis),
        permutation_type="independent",
        n_resamples=20_000,
        alternative="two-sided",
        random_state=2,
        vectorized=True,
    )

    assert ours.observed == pytest.approx(float(reference.statistic))
    # Both are Monte Carlo estimates of the same quantity.
    assert ours.p_value == pytest.approx(float(reference.pvalue), abs=0.02)


def test_exact_p_value_matches_scipy_exactly() -> None:
    a = np.array([1.0, 4.0, 9.0, 2.0])
    b = np.array([7.0, 3.0, 11.0, 5.0, 6.0])

    ours = permutation_test(a, b, method="exact")
    # SciPy accepts np.inf to mean "enumerate everything"; its stubs type it as int.
    reference = stats.permutation_test(  # type: ignore[call-overload]
        (a, b),
        lambda x, y, axis=0: np.mean(x, axis=axis) - np.mean(y, axis=axis),
        permutation_type="independent",
        n_resamples=np.inf,
        alternative="two-sided",
        vectorized=True,
    )

    assert ours.p_value == pytest.approx(float(reference.pvalue))


# --------------------------------------------------------------------------- #
# Behaviour on synthetic data
# --------------------------------------------------------------------------- #


def test_identical_distributions_give_a_large_p_value() -> None:
    rng = np.random.default_rng(3)
    a = rng.normal(0.0, 1.0, size=200)
    b = rng.normal(0.0, 1.0, size=200)

    result = permutation_test(a, b, n_permutations=5000, random_state=3)

    assert result.p_value > 0.05


def test_a_strong_shift_is_detected() -> None:
    rng = np.random.default_rng(4)
    a = rng.normal(0.0, 1.0, size=100)
    b = rng.normal(2.0, 1.0, size=100)

    result = permutation_test(a, b, n_permutations=5000, random_state=4)

    # The smallest attainable two-sided Monte Carlo p-value is 2 / (1 + 5000).
    assert result.p_value == pytest.approx(2 / 5001)
    assert result.observed < 0


def test_the_p_value_is_never_exactly_zero() -> None:
    """The add-one correction keeps the Monte Carlo test valid."""
    a = np.zeros(50)
    b = np.ones(50) * 100.0

    result = permutation_test(a, b, n_permutations=200, random_state=1, method="monte-carlo")

    assert result.p_value > 0
    assert result.p_value == pytest.approx(2 / 201)


def test_the_null_distribution_is_centred_on_zero_for_a_difference() -> None:
    rng = np.random.default_rng(5)
    a = rng.normal(5.0, 1.0, size=80)
    b = rng.normal(5.0, 1.0, size=60)

    result = permutation_test(a, b, n_permutations=5000, random_state=5)

    assert result.null_mean == pytest.approx(0.0, abs=0.05)
    assert result.null_std > 0


def test_type_one_error_is_close_to_nominal_under_the_null() -> None:
    """Repeating the experiment under the null should reject about 5% of the time."""
    rng = np.random.default_rng(202608)
    rejections = 0
    repetitions = 200

    for _ in range(repetitions):
        a = rng.normal(size=25)
        b = rng.normal(size=25)
        result = permutation_test(a, b, n_permutations=200, random_state=rng, method="monte-carlo")
        rejections += int(result.p_value <= 0.05)

    rate = rejections / repetitions
    assert 0.01 < rate < 0.12, f"rejection rate {rate:.3f} is far from 0.05"


# --------------------------------------------------------------------------- #
# Statistics and alternatives
# --------------------------------------------------------------------------- #


def test_a_custom_statistic_is_accepted() -> None:
    rng = np.random.default_rng(6)
    a = rng.normal(0.0, 1.0, size=60)
    b = rng.normal(0.0, 3.0, size=60)

    def spread_ratio(x: np.ndarray, y: np.ndarray) -> float:
        return float(np.std(x, ddof=1) / np.std(y, ddof=1))

    result = permutation_test(a, b, statistic=spread_ratio, n_permutations=2000, random_state=6)

    # The means match but the spreads do not, and this statistic looks at spread.
    assert result.observed < 0.5
    assert result.statistic_name == "spread_ratio"


def test_a_difference_in_spread_alone_can_be_detected() -> None:
    """The sharp null is about distributions, not just means."""
    rng = np.random.default_rng(7)
    a = rng.normal(0.0, 1.0, size=300)
    b = rng.normal(0.0, 4.0, size=300)

    means = permutation_test(a, b, n_permutations=2000, random_state=7)
    spreads = permutation_test(
        a,
        b,
        statistic=lambda x, y: float(np.std(x, ddof=1) - np.std(y, ddof=1)),
        n_permutations=2000,
        random_state=7,
    )

    assert means.p_value > 0.05  # the means really are equal
    assert spreads.p_value < 0.01  # the distributions really are not


@pytest.mark.parametrize("name", ["mean_difference", "median_difference", "welch_t"])
def test_named_statistics_resolve(name: str) -> None:
    result = permutation_test(
        [1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], statistic=name, method="exact"
    )

    assert result.statistic_name == name
    assert np.isfinite(result.observed)


def test_the_studentised_statistic_is_the_welch_t_statistic() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0, 8.0])
    b = np.array([2.0, 4.0, 6.0, 9.0, 12.0])

    expected = float(stats.ttest_ind(a, b, equal_var=False).statistic)

    assert welch_t_statistic(a, b) == pytest.approx(expected)


def test_the_builtin_difference_statistics_are_plain_arithmetic() -> None:
    a = np.array([1.0, 2.0, 6.0])
    b = np.array([1.0, 1.0, 1.0])

    assert mean_difference(a, b) == pytest.approx(2.0)
    assert median_difference(a, b) == pytest.approx(1.0)


@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
def test_alternatives_are_ordered_sensibly(alternative: Alternative) -> None:
    rng = np.random.default_rng(8)
    a = rng.normal(0.0, 1.0, size=50)
    b = rng.normal(1.0, 1.0, size=50)

    result = permutation_test(a, b, alternative=alternative, n_permutations=2000, random_state=8)

    assert 0.0 < result.p_value <= 1.0
    if alternative == "greater":
        # a has the smaller mean, so this direction should not be significant.
        assert result.p_value > 0.5


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #


def test_the_same_seed_reproduces_the_same_null_distribution() -> None:
    rng = np.random.default_rng(9)
    a = rng.normal(size=40)
    b = rng.normal(size=40)

    first = permutation_test(a, b, n_permutations=1000, random_state=42, method="monte-carlo")
    second = permutation_test(a, b, n_permutations=1000, random_state=42, method="monte-carlo")

    assert first.p_value == second.p_value
    np.testing.assert_array_equal(first.null_distribution, second.null_distribution)


def test_exact_enumeration_needs_no_seed_at_all() -> None:
    first = permutation_test([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], method="exact", random_state=1)
    second = permutation_test([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], method="exact", random_state=2)

    assert first.p_value == second.p_value


# --------------------------------------------------------------------------- #
# Cluster-level permutation
# --------------------------------------------------------------------------- #


def _clustered_data(
    rng: np.random.Generator, n_clusters_per_group: int, per_cluster: int, cluster_sd: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two groups of clusters with no real group effect, only cluster effects."""
    values: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    clusters: list[np.ndarray] = []
    for index in range(2 * n_clusters_per_group):
        offset = rng.normal(0.0, cluster_sd)
        values.append(offset + rng.normal(0.0, 1.0, size=per_cluster))
        groups.append(np.full(per_cluster, "A" if index < n_clusters_per_group else "B"))
        clusters.append(np.full(per_cluster, index))
    return np.concatenate(values), np.concatenate(groups), np.concatenate(clusters)


def test_clusters_stay_intact_and_the_observed_statistic_is_unchanged() -> None:
    values = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    groups = np.array(["A", "A", "A", "B", "B", "B"])
    clusters = np.array([1, 1, 1, 2, 2, 2])

    result = clustered_permutation_test(values, groups, clusters, method="exact")

    assert result.observed == pytest.approx(-9.0)
    assert result.n_units_a == 1
    assert result.n_units_b == 1
    # Two clusters can be labelled two ways, so the p-value cannot beat 1.0.
    assert result.n_permutations == 2
    assert any("smallest attainable" in item for item in result.warnings)


def test_row_level_permutation_is_anticonservative_when_clusters_are_real() -> None:
    """The central point: free shuffling of dependent rows invents evidence.

    There is no group effect at all here -- only cluster effects -- so a valid
    test should rarely reject. The row-level test rejects almost always because
    it treats 200 dependent rows as 200 independent ones.
    """
    rng = np.random.default_rng(20260819)
    values, groups, clusters = _clustered_data(
        rng, n_clusters_per_group=10, per_cluster=10, cluster_sd=3.0
    )

    row_level = permutation_test(
        values[groups == "A"],
        values[groups == "B"],
        n_permutations=2000,
        random_state=1,
        method="monte-carlo",
    )
    cluster_level = clustered_permutation_test(
        values, groups, clusters, n_permutations=2000, random_state=1, method="monte-carlo"
    )

    assert row_level.observed == pytest.approx(cluster_level.observed)
    # Same statistic, same data, very different null distribution.
    assert cluster_level.null_std > 2 * row_level.null_std
    assert cluster_level.p_value > row_level.p_value


def test_cluster_level_type_one_error_is_close_to_nominal() -> None:
    """Under a true null with heavy clustering, the clustered test behaves."""
    rng = np.random.default_rng(11)
    cluster_rejections = 0
    row_rejections = 0
    repetitions = 60

    for _ in range(repetitions):
        values, groups, clusters = _clustered_data(
            rng, n_clusters_per_group=8, per_cluster=8, cluster_sd=3.0
        )
        clustered = clustered_permutation_test(
            values, groups, clusters, n_permutations=500, random_state=rng, method="monte-carlo"
        )
        rows = permutation_test(
            values[groups == "A"],
            values[groups == "B"],
            n_permutations=500,
            random_state=rng,
            method="monte-carlo",
        )
        cluster_rejections += int(clustered.p_value <= 0.05)
        row_rejections += int(rows.p_value <= 0.05)

    assert cluster_rejections / repetitions < 0.20
    # The naive test rejects far more often than it should.
    assert row_rejections > cluster_rejections


def test_a_cluster_spanning_both_groups_is_refused() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    groups = np.array(["A", "B", "A", "B"])
    clusters = np.array([1, 1, 2, 2])

    with pytest.raises(PermutationError, match="constant within a cluster"):
        clustered_permutation_test(values, groups, clusters)


def test_the_number_of_groups_must_be_two() -> None:
    values = np.arange(6.0)
    clusters = np.arange(6)

    with pytest.raises(PermutationError, match="Exactly two groups"):
        clustered_permutation_test(values, np.array(["A", "B", "C", "A", "B", "C"]), clusters)


def test_mismatched_lengths_are_refused() -> None:
    with pytest.raises(PermutationError, match="same length"):
        clustered_permutation_test(np.arange(5.0), np.array(["A", "B"]), np.arange(5))


def test_cluster_results_report_the_effective_sample_size() -> None:
    rng = np.random.default_rng(12)
    values, groups, clusters = _clustered_data(rng, 6, 5, 1.0)

    result = clustered_permutation_test(
        values, groups, clusters, n_permutations=500, random_state=1, cluster_name="player"
    )

    assert result.permutation_unit == "player"
    assert any("effective sample size is 12 players" in note for note in result.notes)
    assert "player" in result.exchangeability


# --------------------------------------------------------------------------- #
# Invalid input
# --------------------------------------------------------------------------- #


def test_an_unknown_statistic_name_is_refused() -> None:
    with pytest.raises(PermutationError, match="Unknown statistic"):
        permutation_test([1.0, 2.0], [3.0, 4.0], statistic="wilcoxon")


def test_an_unknown_alternative_is_refused() -> None:
    with pytest.raises(PermutationError, match="Unknown alternative"):
        permutation_test([1.0, 2.0], [3.0, 4.0], alternative="bigger")  # type: ignore[arg-type]


def test_a_non_positive_permutation_count_is_refused() -> None:
    with pytest.raises(PermutationError, match="at least 1"):
        permutation_test([1.0, 2.0], [3.0, 4.0], n_permutations=0)


def test_a_group_too_small_to_permute_is_refused() -> None:
    with pytest.raises(PermutationError, match="at least 2 are needed"):
        permutation_test([1.0], [3.0, 4.0, 5.0])


def test_a_statistic_that_is_not_finite_is_refused() -> None:
    with pytest.raises(PermutationError, match="cannot be compared"):
        permutation_test([1.0, 2.0], [3.0, 4.0], statistic=lambda x, y: float("nan"))


def test_non_finite_observations_are_dropped() -> None:
    result = permutation_test(
        [1.0, 2.0, float("nan"), 3.0], [5.0, 6.0, float("inf"), 7.0], method="exact"
    )

    assert result.n_a == 3
    assert result.n_b == 3


def test_the_exact_threshold_is_respected() -> None:
    """Just under the ceiling enumerates; well over it does not."""
    small = permutation_test(list(range(9)), list(range(9, 18)))
    assert math.comb(18, 9) < MAX_EXACT_PERMUTATIONS
    assert small.method == "exact"

    rng = np.random.default_rng(13)
    large = permutation_test(rng.normal(size=40), rng.normal(size=40), n_permutations=500)
    assert large.method == "monte-carlo"


def test_results_render_as_readable_text() -> None:
    text = str(permutation_test([1.0, 2.0, 3.0], [7.0, 8.0, 9.0], method="exact"))

    assert "Permutation test" in text
    assert "H0" in text
    assert "exchangeability" in text
    assert "observation" in text
