"""Tests for the multiple-testing corrections, checked against statsmodels."""

import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from football_intelligence.statistics.multiple_testing import (
    MultipleTestingError,
    adjust,
    benjamini_hochberg,
    bonferroni,
    compare_corrections,
)

FAMILY = [0.001, 0.008, 0.021, 0.04, 0.06, 0.2, 0.35, 0.7, 0.9, 0.99]


# --------------------------------------------------------------------------- #
# Agreement with statsmodels
# --------------------------------------------------------------------------- #


def test_bonferroni_matches_statsmodels() -> None:
    expected_reject, expected_p, _, _ = multipletests(FAMILY, alpha=0.05, method="bonferroni")

    result = bonferroni(FAMILY)

    np.testing.assert_allclose(result.adjusted_p_values, expected_p)
    np.testing.assert_array_equal(result.rejected, expected_reject)


def test_benjamini_hochberg_matches_statsmodels() -> None:
    expected_reject, expected_p, _, _ = multipletests(FAMILY, alpha=0.05, method="fdr_bh")

    result = benjamini_hochberg(FAMILY)

    np.testing.assert_allclose(result.adjusted_p_values, expected_p)
    np.testing.assert_array_equal(result.rejected, expected_reject)


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1, 0.2])
def test_both_methods_match_statsmodels_at_any_alpha(alpha: float) -> None:
    rng = np.random.default_rng(20260821)
    family = rng.uniform(0.0, 1.0, size=50)

    for method, statsmodels_name in (
        ("bonferroni", "bonferroni"),
        ("benjamini-hochberg", "fdr_bh"),
    ):
        expected_reject, expected_p, _, _ = multipletests(
            family, alpha=alpha, method=statsmodels_name
        )
        result = adjust(family, method=method, alpha=alpha)  # type: ignore[arg-type]

        np.testing.assert_allclose(result.adjusted_p_values, expected_p)
        np.testing.assert_array_equal(result.rejected, expected_reject)
        assert result.alpha == alpha


def test_rejection_flags_agree_with_comparing_adjusted_p_to_alpha() -> None:
    rng = np.random.default_rng(7)
    family = rng.uniform(0.0, 0.3, size=40)

    for result in (bonferroni(family), benjamini_hochberg(family)):
        np.testing.assert_array_equal(result.rejected, result.adjusted_p_values <= result.alpha)


# --------------------------------------------------------------------------- #
# Known arithmetic
# --------------------------------------------------------------------------- #


def test_bonferroni_multiplies_by_the_family_size() -> None:
    result = bonferroni([0.01, 0.02, 0.03, 0.04], alpha=0.05)

    np.testing.assert_allclose(result.adjusted_p_values, [0.04, 0.08, 0.12, 0.16])
    # Only the first survives alpha / 4 = 0.0125.
    assert result.rejected.tolist() == [True, False, False, False]


def test_bonferroni_caps_adjusted_p_values_at_one() -> None:
    result = bonferroni([0.5, 0.9])

    assert result.adjusted_p_values.max() == pytest.approx(1.0)


def test_benjamini_hochberg_is_more_powerful_than_bonferroni() -> None:
    fwer = bonferroni(FAMILY)
    fdr = benjamini_hochberg(FAMILY)

    assert fdr.n_rejected >= fwer.n_rejected
    assert fdr.n_rejected > fwer.n_rejected  # on this family, strictly
    assert np.all(fdr.adjusted_p_values <= fwer.adjusted_p_values + 1e-12)


def test_a_family_of_one_leaves_the_p_value_alone() -> None:
    for result in (bonferroni([0.03]), benjamini_hochberg([0.03])):
        assert result.adjusted_p_values[0] == pytest.approx(0.03)
        assert result.rejected[0]
        assert any("family of one" in note for note in result.notes)


def test_adjusted_p_values_are_never_smaller_than_the_originals() -> None:
    rng = np.random.default_rng(3)
    family = rng.uniform(0.0, 1.0, size=100)

    for result in (bonferroni(family), benjamini_hochberg(family)):
        assert np.all(result.adjusted_p_values >= result.p_values - 1e-12)


# --------------------------------------------------------------------------- #
# The point of the module
# --------------------------------------------------------------------------- #


def test_uncorrected_testing_finds_something_from_pure_noise() -> None:
    """Under the global null, uncorrected testing produces false positives."""
    rng = np.random.default_rng(20260821)
    # p-values under a true null are uniform on [0, 1].
    family = rng.uniform(0.0, 1.0, size=100)

    fwer = bonferroni(family)
    fdr = benjamini_hochberg(family)

    assert fwer.n_rejected_uncorrected >= 3  # about 5 expected
    assert fwer.n_rejected == 0
    assert fdr.n_rejected == 0


def test_the_family_wise_error_rate_of_uncorrected_testing_is_reported() -> None:
    result = bonferroni([0.5] * 35)

    assert result.n_tests == 35
    assert result.expected_false_positives_uncorrected == pytest.approx(1.75)
    # 1 - 0.95 ** 35
    assert result.probability_of_any_false_positive == pytest.approx(0.8339, abs=1e-4)


def test_the_same_p_value_is_adjusted_differently_in_a_larger_family() -> None:
    """Adjusted p-values are not effect sizes: the family size changes them."""
    small = bonferroni([0.01, 0.4, 0.6])
    large = bonferroni([0.01, *[0.4] * 49])

    assert small.p_values[0] == large.p_values[0] == 0.01
    assert small.adjusted_p_values[0] == pytest.approx(0.03)
    assert large.adjusted_p_values[0] == pytest.approx(0.5)
    # Same data, same test, different conclusion purely from the company it keeps.
    assert small.rejected[0]
    assert not large.rejected[0]


def test_results_warn_that_adjusted_p_values_are_not_effect_sizes() -> None:
    for result in (bonferroni(FAMILY), benjamini_hochberg(FAMILY)):
        assert any("not effect sizes" in note for note in result.notes)


def test_benjamini_hochberg_states_its_dependence_assumption() -> None:
    result = benjamini_hochberg(FAMILY)

    assert any("independent or positively dependent" in note for note in result.notes)
    assert result.controls == "false discovery rate"


def test_bonferroni_states_that_it_needs_no_dependence_assumption() -> None:
    result = bonferroni(FAMILY)

    assert any("no assumption about dependence" in note for note in result.notes)
    assert result.controls == "family-wise error rate"


# --------------------------------------------------------------------------- #
# Tabulation
# --------------------------------------------------------------------------- #


def test_to_frame_sorts_by_raw_p_value_and_keeps_labels() -> None:
    result = benjamini_hochberg([0.4, 0.01, 0.2], labels=["c", "a", "b"])

    frame = result.to_frame()

    assert list(frame.index) == ["a", "b", "c"]
    assert frame["p_value"].is_monotonic_increasing
    assert "benjamini-hochberg_adjusted" in frame.columns


def test_compare_corrections_puts_both_methods_side_by_side() -> None:
    frame = compare_corrections(FAMILY, labels=[f"p{i}" for i in range(len(FAMILY))])

    assert list(frame.columns) == [
        "p_value",
        "bonferroni_p",
        "benjamini_hochberg_p",
        "rejected_uncorrected",
        "rejected_bonferroni",
        "rejected_benjamini_hochberg",
    ]
    assert len(frame) == len(FAMILY)
    assert frame["rejected_uncorrected"].sum() >= frame["rejected_benjamini_hochberg"].sum()
    assert frame["rejected_benjamini_hochberg"].sum() >= frame["rejected_bonferroni"].sum()


def test_default_labels_are_generated_when_none_are_supplied() -> None:
    frame = bonferroni([0.01, 0.02]).to_frame()

    assert set(frame.index) == {"test 1", "test 2"}


# --------------------------------------------------------------------------- #
# Invalid input
# --------------------------------------------------------------------------- #


def test_an_empty_family_is_refused() -> None:
    with pytest.raises(MultipleTestingError, match="empty family"):
        bonferroni([])


@pytest.mark.parametrize("bad", [[-0.1, 0.5], [0.5, 1.2]])
def test_p_values_outside_the_unit_interval_are_refused(bad: list[float]) -> None:
    with pytest.raises(MultipleTestingError, match=r"must lie in \[0, 1\]"):
        bonferroni(bad)


def test_non_finite_p_values_are_refused() -> None:
    with pytest.raises(MultipleTestingError, match="must all be finite"):
        benjamini_hochberg([0.01, float("nan")])


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.5, 2.0])
def test_an_impossible_alpha_is_refused(alpha: float) -> None:
    with pytest.raises(MultipleTestingError, match="strictly between 0 and 1"):
        bonferroni([0.01, 0.2], alpha=alpha)


def test_a_label_count_mismatch_is_refused() -> None:
    with pytest.raises(MultipleTestingError, match="2 labels for 3 p-values"):
        bonferroni([0.01, 0.2, 0.3], labels=["a", "b"])


def test_an_unknown_method_is_refused() -> None:
    with pytest.raises(MultipleTestingError, match="Unknown method"):
        adjust([0.01, 0.2], method="holm")  # type: ignore[arg-type]


def test_a_two_dimensional_family_is_refused() -> None:
    with pytest.raises(MultipleTestingError, match="one-dimensional"):
        bonferroni([[0.01, 0.2], [0.3, 0.4]])


def test_results_render_as_readable_text() -> None:
    text = str(benjamini_hochberg(FAMILY))

    assert "benjamini-hochberg" in text
    assert "false discovery rate" in text
    assert "tests in family" in text
