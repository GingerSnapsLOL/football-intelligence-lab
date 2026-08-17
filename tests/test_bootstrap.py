"""Tests for the bootstrap, using synthetic data with known sampling behaviour."""

import numpy as np
import pandas as pd
import pytest

from football_intelligence.statistics.bootstrap import (
    BootstrapError,
    bootstrap,
    cluster_sizes,
    compare_resampling_units,
)


def mean_of(sample: np.ndarray) -> float:
    return float(np.mean(sample))


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #


def test_the_same_seed_reproduces_the_same_interval() -> None:
    sample = np.arange(100.0)

    first = bootstrap(sample, mean_of, n_resamples=500, random_state=7)
    second = bootstrap(sample, mean_of, n_resamples=500, random_state=7)

    assert first.confidence_interval == second.confidence_interval
    assert first.standard_error == second.standard_error
    np.testing.assert_array_equal(first.replicates, second.replicates)


def test_different_seeds_give_different_replicates() -> None:
    sample = np.arange(100.0)

    first = bootstrap(sample, mean_of, n_resamples=500, random_state=1)
    second = bootstrap(sample, mean_of, n_resamples=500, random_state=2)

    assert not np.array_equal(first.replicates, second.replicates)
    # But they agree to within Monte Carlo error.
    assert first.standard_error == pytest.approx(second.standard_error, rel=0.15)


def test_a_generator_can_be_passed_instead_of_a_seed() -> None:
    sample = np.arange(50.0)

    from_generator = bootstrap(
        sample, mean_of, n_resamples=200, random_state=np.random.default_rng(3)
    )
    from_seed = bootstrap(sample, mean_of, n_resamples=200, random_state=3)

    np.testing.assert_array_equal(from_generator.replicates, from_seed.replicates)


# --------------------------------------------------------------------------- #
# Agreement with known analytical results
# --------------------------------------------------------------------------- #


def test_bootstrap_standard_error_of_a_mean_matches_the_textbook_formula() -> None:
    rng = np.random.default_rng(20260818)
    sample = rng.normal(10.0, 2.0, size=500)
    analytical = float(np.std(sample, ddof=1) / np.sqrt(sample.size))

    result = bootstrap(sample, mean_of, n_resamples=4000, random_state=11)

    # For a mean the bootstrap has a known answer, so this is a real check.
    assert result.standard_error == pytest.approx(analytical, rel=0.05)
    assert result.observed == pytest.approx(float(np.mean(sample)))


def test_percentile_interval_is_close_to_the_normal_interval_for_a_mean() -> None:
    rng = np.random.default_rng(5)
    sample = rng.normal(0.0, 1.0, size=800)
    centre = float(np.mean(sample))
    half_width = 1.96 * float(np.std(sample, ddof=1) / np.sqrt(sample.size))

    result = bootstrap(sample, mean_of, n_resamples=4000, random_state=5)
    low, high = result.confidence_interval

    assert low == pytest.approx(centre - half_width, abs=0.02)
    assert high == pytest.approx(centre + half_width, abs=0.02)


def test_the_interval_contains_the_true_parameter_in_a_controlled_example() -> None:
    rng = np.random.default_rng(99)
    sample = rng.normal(loc=10.0, scale=2.0, size=400)

    result = bootstrap(sample, mean_of, n_resamples=2000, random_state=99)
    low, high = result.confidence_interval

    assert low < 10.0 < high


def test_interval_coverage_is_close_to_the_nominal_level() -> None:
    """Repeat the whole experiment and count how often the interval is right."""
    rng = np.random.default_rng(4242)
    covered = 0
    repetitions = 150

    for _ in range(repetitions):
        sample = rng.normal(loc=5.0, scale=1.0, size=120)
        result = bootstrap(sample, mean_of, n_resamples=400, random_state=rng)
        low, high = result.confidence_interval
        covered += int(low <= 5.0 <= high)

    coverage = covered / repetitions
    assert 0.88 < coverage < 0.99, f"coverage {coverage:.3f} is far from 0.95"


def test_the_interval_narrows_as_the_sample_grows() -> None:
    rng = np.random.default_rng(31)
    small = bootstrap(rng.normal(size=100), mean_of, n_resamples=1000, random_state=1)
    large = bootstrap(rng.normal(size=10_000), mean_of, n_resamples=1000, random_state=1)

    # Ten times the data should roughly cut the width by sqrt(10).
    assert large.interval_width < small.interval_width / 2


# --------------------------------------------------------------------------- #
# Arbitrary statistics
# --------------------------------------------------------------------------- #


def test_a_median_can_be_bootstrapped_although_its_standard_error_has_no_simple_formula() -> None:
    rng = np.random.default_rng(17)
    sample = rng.normal(0.0, 1.0, size=600)

    result = bootstrap(sample, lambda x: float(np.median(x)), n_resamples=2000, random_state=17)

    # For a normal sample the median's standard error is about 1.253 sigma / sqrt(n).
    expected = 1.2533 * 1.0 / np.sqrt(600)
    assert result.standard_error == pytest.approx(expected, rel=0.2)


def test_a_ratio_of_two_columns_can_be_bootstrapped_from_a_dataframe() -> None:
    rng = np.random.default_rng(8)
    frame = pd.DataFrame(
        {
            "goal": rng.random(500) < 0.1,
            "expected": rng.uniform(0.02, 0.3, size=500),
        }
    )

    result = bootstrap(
        frame,
        lambda f: float(f["goal"].sum() - f["expected"].sum()),
        n_resamples=1000,
        random_state=8,
        statistic_name="goals minus expected goals",
    )

    assert result.observed == pytest.approx(float(frame["goal"].sum() - frame["expected"].sum()))
    assert result.standard_error > 0


def test_a_proportion_can_be_bootstrapped_from_booleans() -> None:
    rng = np.random.default_rng(12)
    sample = rng.random(1000) < 0.25

    result = bootstrap(sample, mean_of, n_resamples=2000, random_state=12)

    analytical = np.sqrt(0.25 * 0.75 / 1000)
    assert result.observed == pytest.approx(0.25, abs=0.04)
    assert result.standard_error == pytest.approx(analytical, rel=0.15)


# --------------------------------------------------------------------------- #
# Cluster resampling
# --------------------------------------------------------------------------- #


def _clustered_sample(
    rng: np.random.Generator, n_clusters: int, per_cluster: int, between_sd: float
) -> tuple[np.ndarray, np.ndarray]:
    """Data where each cluster has its own offset, so rows within it agree."""
    offsets = rng.normal(0.0, between_sd, size=n_clusters)
    values = np.concatenate([offset + rng.normal(0.0, 1.0, size=per_cluster) for offset in offsets])
    labels = np.repeat(np.arange(n_clusters), per_cluster)
    return values, labels


def test_cluster_resampling_keeps_whole_clusters_together() -> None:
    """Each drawn cluster must contribute all of its rows, or none."""
    values = np.arange(12.0)
    labels = np.repeat([10, 20, 30], 4)
    seen: list[np.ndarray] = []

    def record(sample: np.ndarray) -> float:
        seen.append(np.asarray(sample).copy())
        return float(np.mean(sample))

    bootstrap(values, record, clusters=labels, n_resamples=50, random_state=2)

    blocks = {(0.0, 1.0, 2.0, 3.0), (4.0, 5.0, 6.0, 7.0), (8.0, 9.0, 10.0, 11.0)}
    for resample in seen:
        assert resample.size == 12  # three clusters of four, always
        chunks = {tuple(resample[start : start + 4]) for start in range(0, 12, 4)}
        assert chunks <= blocks


def test_cluster_bootstrap_matches_the_standard_error_of_the_cluster_means() -> None:
    """With balanced clusters there is an analytical answer to compare against."""
    rng = np.random.default_rng(20260818)
    values, labels = _clustered_sample(rng, n_clusters=60, per_cluster=25, between_sd=3.0)
    cluster_means = np.array([values[labels == label].mean() for label in np.unique(labels)])
    analytical = float(np.std(cluster_means, ddof=1) / np.sqrt(cluster_means.size))

    result = bootstrap(
        values, mean_of, clusters=labels, n_resamples=3000, random_state=1, cluster_name="match"
    )

    assert result.standard_error == pytest.approx(analytical, rel=0.1)
    assert result.n_clusters == 60
    assert result.resampling_unit == "match"


def test_resampling_rows_understates_uncertainty_when_clusters_are_real() -> None:
    """The central point of the module, on data whose structure we control."""
    rng = np.random.default_rng(20260818)
    values, labels = _clustered_sample(rng, n_clusters=60, per_cluster=25, between_sd=3.0)

    rows, clustered = compare_resampling_units(
        values, mean_of, clusters=labels, n_resamples=2000, random_state=3, cluster_name="match"
    )

    # Between-cluster spread of 3 against within-cluster spread of 1 means most of
    # the variation is between clusters, and ignoring that shrinks the interval a lot.
    assert clustered.standard_error > 3 * rows.standard_error
    assert clustered.interval_width > 3 * rows.interval_width
    assert rows.resampling_unit == "observation"


def test_the_two_units_agree_when_there_is_no_cluster_effect() -> None:
    rng = np.random.default_rng(77)
    values, labels = _clustered_sample(rng, n_clusters=60, per_cluster=25, between_sd=0.0)

    rows, clustered = compare_resampling_units(
        values, mean_of, clusters=labels, n_resamples=2000, random_state=9
    )

    # Nothing is shared within a cluster, so the unit should not matter much.
    assert clustered.standard_error == pytest.approx(rows.standard_error, rel=0.2)


def test_singleton_clusters_are_flagged_as_equivalent_to_row_resampling() -> None:
    values = np.arange(30.0)

    result = bootstrap(values, mean_of, clusters=np.arange(30), n_resamples=200, random_state=1)

    assert any("identical to resampling observations" in item for item in result.warnings)


def test_too_few_clusters_are_flagged() -> None:
    values = np.arange(40.0)
    labels = np.repeat(np.arange(5), 8)

    result = bootstrap(values, mean_of, clusters=labels, n_resamples=500, random_state=1)

    assert any("Only 5 independent" in item for item in result.warnings)


def test_cluster_sizes_reports_the_structure() -> None:
    labels = ["a", "a", "a", "b", "b", "c"]

    sizes = cluster_sizes(labels)

    assert sizes.loc["a"] == 3
    assert sizes.loc["c"] == 1


# --------------------------------------------------------------------------- #
# Invalid input
# --------------------------------------------------------------------------- #


def test_an_empty_sample_is_rejected() -> None:
    with pytest.raises(BootstrapError, match="empty sample"):
        bootstrap(np.array([]), mean_of)


def test_a_scalar_is_rejected() -> None:
    # mypy rejects this too; the runtime check covers untyped callers.
    with pytest.raises(BootstrapError, match="Cannot bootstrap a scalar"):
        bootstrap(4.0, mean_of)  # type: ignore[arg-type]


@pytest.mark.parametrize("n_resamples", [0, -5])
def test_a_non_positive_resample_count_is_rejected(n_resamples: int) -> None:
    with pytest.raises(BootstrapError, match="at least 1"):
        bootstrap(np.arange(10.0), mean_of, n_resamples=n_resamples)


@pytest.mark.parametrize("level", [0.0, 1.0, 1.5, -0.1])
def test_an_impossible_confidence_level_is_rejected(level: float) -> None:
    with pytest.raises(BootstrapError, match="strictly between 0 and 1"):
        bootstrap(np.arange(10.0), mean_of, confidence_level=level)


def test_mismatched_cluster_labels_are_rejected() -> None:
    with pytest.raises(BootstrapError, match="Every row must be assigned"):
        bootstrap(np.arange(10.0), mean_of, clusters=np.arange(3))


def test_missing_cluster_labels_are_rejected() -> None:
    labels = np.array([1.0, 1.0, np.nan, 2.0])

    with pytest.raises(BootstrapError, match="missing values"):
        bootstrap(np.arange(4.0), mean_of, clusters=labels)


def test_a_statistic_that_fails_on_the_original_data_is_reported() -> None:
    def explode(sample: np.ndarray) -> float:
        raise RuntimeError("no")

    with pytest.raises(BootstrapError, match="raised on the original data"):
        bootstrap(np.arange(10.0), explode)


def test_a_statistic_returning_several_numbers_is_rejected() -> None:
    with pytest.raises(BootstrapError, match="single number"):
        bootstrap(np.arange(10.0), lambda x: np.asarray([1.0, 2.0]))  # type: ignore[arg-type,return-value]


def test_a_statistic_that_is_not_finite_on_the_original_data_is_rejected() -> None:
    with pytest.raises(BootstrapError, match="not finite"):
        bootstrap(np.arange(10.0), lambda x: float("nan"))


def test_resamples_that_fail_are_counted_rather_than_crashing() -> None:
    """A rate with an empty denominator is undefined for some resamples."""
    values = np.array([0.0] * 20 + [1.0])

    def ratio_over_ones(sample: np.ndarray) -> float:
        ones = float(np.sum(sample))
        if ones == 0:
            raise ZeroDivisionError("no positives in this resample")
        return float(np.sum(sample) / ones)

    result = bootstrap(values, ratio_over_ones, n_resamples=500, random_state=1)

    assert result.n_failed_resamples > 0
    assert any("non-finite value" in item for item in result.warnings)
    assert np.isfinite(result.confidence_interval).all()


def test_few_resamples_are_flagged() -> None:
    result = bootstrap(np.arange(50.0), mean_of, n_resamples=100, random_state=1)

    assert any("few for a percentile interval" in item for item in result.warnings)


def test_results_render_as_readable_text() -> None:
    text = str(
        bootstrap(
            np.arange(100.0),
            mean_of,
            n_resamples=500,
            random_state=1,
            statistic_name="mean shot distance",
        )
    )

    assert "mean shot distance" in text
    assert "95% percentile CI" in text
    assert "observation" in text
