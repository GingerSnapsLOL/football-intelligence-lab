"""Hypothesis tests for two-sample, paired and multi-group designs.

Thin, typed wrappers over SciPy and statsmodels. The algorithms are theirs; what
this module adds is a consistent result object that carries the things a p-value
alone cannot: which hypothesis was tested, which quantity it concerns, how large
the observed effect is, what the test assumes, and which of those assumptions
look strained by the data at hand.

Choosing by design, not by data
-------------------------------

The first question is never "is the data normal?" but "what is the design?".
Every result records the ``design`` it assumes, and the three implemented
families are not interchangeable:

``independent samples``
    Two groups of separate units. Nothing links observation *i* of one group to
    observation *i* of the other, and the groups may differ in size.
    :func:`welch_t_test`, :func:`student_t_test`, :func:`mann_whitney_u_test`,
    :func:`kolmogorov_smirnov_test`.

``paired samples``
    One set of units measured twice, so the data arrives as *n* matched pairs
    and the analysis is of the *n* differences.
    :func:`paired_t_test`, :func:`wilcoxon_signed_rank_test`.

``independent multi-group``
    Three or more groups of separate units.
    :func:`one_way_anova`, :func:`welch_anova`, :func:`kruskal_wallis_test`.

Mixing these up is not a minor inefficiency. Feeding paired data to an
independent-samples test throws away the matching that makes the design powerful
*and* violates the independence assumption. Feeding unpaired data to a paired
test is not even defined: :func:`paired_t_test` refuses samples of different
lengths, and equal lengths alone do not create a pairing -- the correspondence
has to come from the design.

One design is deliberately **absent**: repeated-measures multi-group, where the
same unit appears in more than one group. None of the functions here is valid
for it; it needs a mixed model with a subject random effect (task T28).

Why ordinary one-way ANOVA is not valid for repeated shots from one player
--------------------------------------------------------------------------

One-way ANOVA compares between-group variance to within-group variance, and the
within-group term is only an honest yardstick if the observations inside a group
are independent draws. Shots are not: one player contributes many of them, and
they share everything unmodelled about that player -- role, position, shot
selection, finishing habits. Residuals are therefore positively correlated
within player, with three consequences:

1. The within-group mean square underestimates the variance of a group mean, so
   F is inflated and the p-value is too small. With an intra-class correlation
   of rho and an average of m shots per player, the variance of a group mean is
   understated by roughly a factor of ``1 + (m - 1) * rho``; at m = 4.5 shots per
   player and rho = 0.1 that is already 35%.
2. The effective sample size is closer to the number of players than to the
   number of shots.
3. Group membership can be confounded with player identity. Headers come
   disproportionately from centre-backs and target strikers, so a "body part
   effect" may partly be a player-composition effect.

Aggregating to one value per player per group fixes the first two but not the
third, and if a player appears in several groups the design becomes repeated
measures rather than one-way. The valid analyses are a mixed model with a player
random intercept (T28), or inference that resamples whole clusters (T09/T10).

How to read a :class:`HypothesisTestResult`
-------------------------------------------

- ``p_value`` is the probability of a statistic at least as extreme as the one
  observed **if the null hypothesis were true**. It is not the probability that
  the null is true, and a large p-value is not evidence that the null is true --
  it is compatible with a real effect the study lacked the power to detect.
- ``estimate`` answers "how big?", which is the question ``p_value`` never
  answers. A tiny p-value from a large sample can accompany an effect too small
  to matter; a large p-value from a small sample can hide an important one.
- ``estimand`` names the quantity the test concerns. Two tests applied to the
  same two samples can disagree simply because they are asking different
  questions -- see :func:`mann_whitney_u_test` versus :func:`welch_t_test`.

There is deliberately no ``significant`` field and no ``alpha`` argument.
Thresholding a p-value is a reporting decision, and for families of tests it
belongs with the multiple-testing correction rather than in the test itself.

Choosing a test is not automated here, and no function in this module inspects a
normality test in order to pick another test. See
``football_intelligence.statistics.diagnostics`` for why that pattern is invalid.
"""

import argparse
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from football_intelligence.statistics.diagnostics import to_float_sample

logger = logging.getLogger(__name__)

Alternative = Literal["two-sided", "less", "greater"]

#: The study designs this module supports. Repeated-measures multi-group designs
#: are deliberately not among them; see the module docstring.
Design = Literal["independent samples", "paired samples", "independent multi-group"]

DEFAULT_CONFIDENCE_LEVEL: Final = 0.95

#: Cochran's conventional rule for the chi-square approximation.
MIN_EXPECTED_COUNT: Final = 5.0

#: Below this many observations per group, a t approximation leans heavily on
#: the shape of the data rather than on the central limit theorem.
SMALL_GROUP_N: Final = 15

#: Ratio of sample variances beyond which the equal-variance assumption of
#: Student's t-test looks doubtful.
VARIANCE_RATIO_WARNING: Final = 2.0

#: Standing caveat: every test here assumes observations are independent, and
#: football event data is clustered by player, match and team.
CLUSTERING_ASSUMPTION: Final = (
    "Observations are assumed independent. Shots are clustered within players, "
    "matches and teams, so a test run on raw shot rows will understate the "
    "standard error and overstate significance."
)


class StatisticalTestError(ValueError):
    """Raised when a test cannot be run on the data as supplied."""


@dataclass(frozen=True, slots=True)
class HypothesisTestResult:
    """Outcome of one hypothesis test.

    ``n_a``/``n_b`` are the usable group sizes for two-sample tests and ``None``
    for contingency tests, where ``n_total`` and ``table_shape`` describe the
    table instead.
    """

    test_name: str
    design: Design
    statistic_name: str
    statistic: float
    p_value: float
    alternative: str
    null_hypothesis: str
    alternative_hypothesis: str
    estimand: str
    n_total: int
    n_a: int | None = None
    n_b: int | None = None
    n_pairs: int | None = None
    group_names: tuple[str, ...] | None = None
    group_sizes: tuple[int, ...] | None = None
    table_shape: tuple[int, int] | None = None
    degrees_of_freedom: float | None = None
    estimate: float | None = None
    estimate_name: str | None = None
    confidence_interval: tuple[float, float] | None = None
    confidence_level: float | None = None
    assumptions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def interpretation(self) -> str:
        """A sentence that states what the p-value does and does not establish."""
        return (
            f"p = {self.p_value:.4g} is the probability, if the null hypothesis were true, of "
            f"data at least as extreme as this sample ({self.statistic_name} = "
            f"{self.statistic:.4g}). It is not the probability that the null is true, a large "
            "value is not evidence for it, and it says nothing about whether the effect is "
            "large enough to matter."
        )

    def __str__(self) -> str:
        alternative_label = f"H1 ({self.alternative})"
        lines = [
            f"{self.test_name}",
            f"  {'design':<17} {self.design}",
            f"  {'H0':<17} {self.null_hypothesis}",
            f"  {alternative_label:<17} {self.alternative_hypothesis}",
            f"  {'estimand':<17} {self.estimand}",
            f"  {self.statistic_name:<17} {self.statistic:.6g}"
            + (
                f"   (df = {self.degrees_of_freedom:.4g})"
                if self.degrees_of_freedom is not None
                else ""
            ),
            f"  {'p-value':<17} {self.p_value:.6g}",
        ]
        if self.n_pairs is not None:
            lines.append(f"  {'n':<17} {self.n_pairs:,} matched pairs")
        elif self.group_names is not None and self.group_sizes is not None:
            rendered = ", ".join(
                f"{name} ({size:,})"
                for name, size in zip(self.group_names, self.group_sizes, strict=True)
            )
            lines.append(f"  {'n':<17} {self.n_total:,} in {len(self.group_names)} groups")
            lines.append(f"  {'groups':<17} {rendered}")
        elif self.n_a is not None and self.n_b is not None:
            lines.append(f"  {'n':<17} {self.n_a:,} vs {self.n_b:,}")
        else:
            lines.append(f"  {'n':<17} {self.n_total:,} in a {self.table_shape} table")
        if self.estimate is not None:
            line = f"  {self.estimate_name:<17} {self.estimate:.6g}"
            if self.confidence_interval is not None and self.confidence_level is not None:
                low, high = self.confidence_interval
                line += f"   {self.confidence_level:.0%} CI [{low:.6g}, {high:.6g}]"
            lines.append(line)
        lines.append(f"  {'reading':<17} {self.interpretation}")
        lines.extend(f"  {'assumption':<17} {item}" for item in self.assumptions)
        lines.extend(f"  {'WARNING':<17} {item}" for item in self.warnings)
        lines.extend(f"  {'note':<17} {item}" for item in self.notes)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #


def _clean_pair(
    a: npt.ArrayLike, b: npt.ArrayLike, *, label_a: str, label_b: str
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], tuple[str, ...]]:
    """Drop non-finite values from both samples, reporting what was removed."""
    array_a = to_float_sample(a)
    array_b = to_float_sample(b)
    clean_a = array_a[np.isfinite(array_a)]
    clean_b = array_b[np.isfinite(array_b)]

    warnings: list[str] = []
    dropped_a = array_a.size - clean_a.size
    dropped_b = array_b.size - clean_b.size
    if dropped_a or dropped_b:
        warnings.append(
            f"Dropped {dropped_a} non-finite value(s) from {label_a} and {dropped_b} from "
            f"{label_b}. Check whether they are missing at random before reading the result."
        )
    for clean, label in ((clean_a, label_a), (clean_b, label_b)):
        if clean.size < 2:
            raise StatisticalTestError(
                f"Sample {label!r} has {clean.size} usable observation(s); at least 2 are needed."
            )
    return clean_a, clean_b, tuple(warnings)


def _as_contingency_table(table: npt.ArrayLike | pd.DataFrame) -> npt.NDArray[np.int64]:
    """Validate and convert a contingency table of counts."""
    array = np.asarray(table.to_numpy() if isinstance(table, pd.DataFrame) else table)
    if array.ndim != 2:
        raise StatisticalTestError(f"Expected a two-dimensional table, got shape {array.shape}.")
    if array.size == 0 or array.shape[0] < 2 or array.shape[1] < 2:
        raise StatisticalTestError(f"A contingency table needs at least 2x2, got {array.shape}.")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.allclose(array, np.round(array)):
            raise StatisticalTestError("Contingency tables must contain counts, not fractions.")
        array = np.round(array).astype(np.int64)
    if (array < 0).any():
        raise StatisticalTestError("Contingency tables cannot contain negative counts.")
    if array.sum() == 0:
        raise StatisticalTestError("Contingency table is empty.")
    return array.astype(np.int64)


def _t_test(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    equal_var: bool,
    alternative: Alternative,
    confidence_level: float,
    label_a: str,
    label_b: str,
) -> HypothesisTestResult:
    """Shared implementation of the Student and Welch independent t-tests."""
    clean_a, clean_b, warnings = _clean_pair(a, b, label_a=label_a, label_b=label_b)
    result = stats.ttest_ind(clean_a, clean_b, equal_var=equal_var, alternative=alternative)
    interval = result.confidence_interval(confidence_level=confidence_level)

    variance_a = float(np.var(clean_a, ddof=1))
    variance_b = float(np.var(clean_b, ddof=1))
    ratio = (
        max(variance_a, variance_b) / min(variance_a, variance_b)
        if min(variance_a, variance_b) > 0
        else float("inf")
    )

    warning_list = list(warnings)
    assumptions = [
        CLUSTERING_ASSUMPTION,
        "The sampling distribution of each group mean is approximately normal, by the central "
        "limit theorem or by the observations themselves being normal. Normality of the raw "
        "observations is not required for large samples.",
    ]

    if equal_var:
        assumptions.append("Both groups are assumed to have the same population variance.")
        if ratio > VARIANCE_RATIO_WARNING:
            warning_list.append(
                f"Sample variances differ by a factor of {ratio:.2f} "
                f"({variance_a:.4g} vs {variance_b:.4g}), which strains the equal-variance "
                "assumption. Welch's test does not make it and is the safer default."
            )
        if len(clean_a) != len(clean_b) and ratio > VARIANCE_RATIO_WARNING:
            warning_list.append(
                f"Group sizes are unequal ({len(clean_a)} vs {len(clean_b)}) and the variances "
                "differ. This is the configuration in which Student's t-test is least robust: "
                "its error rate can be badly wrong in either direction."
            )
    else:
        assumptions.append(
            "Group variances may differ; the degrees of freedom are adjusted "
            "(Welch-Satterthwaite) and are usually fractional."
        )

    for sample, label in ((clean_a, label_a), (clean_b, label_b)):
        if SMALL_GROUP_N > sample.size >= 3 and float(np.ptp(sample)) > 0.0:
            skewness = float(stats.skew(sample, bias=False))
            if abs(skewness) > 1.0:
                warning_list.append(
                    f"Group {label!r} has only {sample.size} observations with sample skewness "
                    f"{skewness:+.2f}. At this size the central limit theorem does little work, "
                    "so the t approximation rests on the shape of the data."
                )

    name = "Student's independent t-test" if equal_var else "Welch's independent t-test"
    return HypothesisTestResult(
        test_name=name,
        design="independent samples",
        statistic_name="t",
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        alternative=alternative,
        null_hypothesis=f"the population means of {label_a} and {label_b} are equal",
        alternative_hypothesis=_mean_alternative(alternative, label_a, label_b),
        estimand=f"difference in population means, mean({label_a}) - mean({label_b})",
        n_total=int(clean_a.size + clean_b.size),
        n_a=int(clean_a.size),
        n_b=int(clean_b.size),
        degrees_of_freedom=float(result.df),
        estimate=float(np.mean(clean_a) - np.mean(clean_b)),
        estimate_name="mean difference",
        confidence_interval=(float(interval.low), float(interval.high)),
        confidence_level=confidence_level,
        assumptions=tuple(assumptions),
        warnings=tuple(warning_list),
        notes=(
            "The confidence interval, not the p-value, carries the information about how large "
            "the difference plausibly is.",
        ),
    )


def _mean_alternative(alternative: Alternative, label_a: str, label_b: str) -> str:
    if alternative == "two-sided":
        return f"the means of {label_a} and {label_b} differ"
    if alternative == "less":
        return f"the mean of {label_a} is smaller than that of {label_b}"
    return f"the mean of {label_a} is larger than that of {label_b}"


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def student_t_test(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    alternative: Alternative = "two-sided",
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    label_a: str = "A",
    label_b: str = "B",
) -> HypothesisTestResult:
    """Independent two-sample t-test assuming equal population variances.

    Appropriate when the two groups are independent, the question is about a
    difference in *means*, and there is a substantive reason to believe the
    variances are equal (for example a randomised design where only the mean is
    expected to move).

    Prefer :func:`welch_t_test` otherwise: it costs almost no power when the
    variances really are equal and remains valid when they are not.
    """
    return _t_test(
        a,
        b,
        equal_var=True,
        alternative=alternative,
        confidence_level=confidence_level,
        label_a=label_a,
        label_b=label_b,
    )


def welch_t_test(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    alternative: Alternative = "two-sided",
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    label_a: str = "A",
    label_b: str = "B",
) -> HypothesisTestResult:
    """Independent two-sample t-test that does not assume equal variances.

    The sensible default for comparing two group means. Robust to unequal
    variances and unequal group sizes, and for large samples it does not need the
    observations themselves to be normal -- only the sampling distribution of
    each mean, which the central limit theorem supplies.
    """
    return _t_test(
        a,
        b,
        equal_var=False,
        alternative=alternative,
        confidence_level=confidence_level,
        label_a=label_a,
        label_b=label_b,
    )


def mann_whitney_u_test(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    alternative: Alternative = "two-sided",
    label_a: str = "A",
    label_b: str = "B",
) -> HypothesisTestResult:
    """Mann-Whitney U test: a rank-based test of stochastic ordering.

    **This is not a test of medians.** Its null is that a randomly chosen
    observation from one group is as likely to exceed one from the other as the
    reverse. Two distributions with identical medians can be rejected by this
    test, and two with different medians can fail to be. Reading it as a
    comparison of medians requires the extra assumption that the distributions
    have the same shape and spread and differ only by a shift -- an assumption
    the test itself neither makes nor checks.

    The reported estimate is the probability of superiority,
    ``P(A > B) + 0.5 * P(A = B)``, which is exactly what the U statistic
    measures, rescaled to [0, 1]. It equals 0.5 under the null.

    Appropriate when the question is about which group tends to produce larger
    values, when the outcome is ordinal, or when heavy tails make a mean an
    uninformative summary.
    """
    clean_a, clean_b, warnings = _clean_pair(a, b, label_a=label_a, label_b=label_b)
    result = stats.mannwhitneyu(clean_a, clean_b, alternative=alternative)

    n_a, n_b = int(clean_a.size), int(clean_b.size)
    probability_of_superiority = float(result.statistic) / (n_a * n_b)

    warning_list = list(warnings)
    combined = np.concatenate([clean_a, clean_b])
    tie_count = combined.size - np.unique(combined).size
    if tie_count:
        warning_list.append(
            f"{tie_count} tied value(s) across the two samples; SciPy uses the tie-corrected "
            "normal approximation rather than the exact distribution."
        )

    return HypothesisTestResult(
        test_name="Mann-Whitney U test",
        design="independent samples",
        statistic_name="U",
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        alternative=alternative,
        null_hypothesis=(
            f"{label_a} and {label_b} are stochastically equal: a random observation from one "
            "is as likely to exceed a random observation from the other as the reverse"
        ),
        alternative_hypothesis=_stochastic_alternative(alternative, label_a, label_b),
        estimand=f"probability of superiority, P({label_a} > {label_b}) + 0.5 P(equal)",
        n_total=n_a + n_b,
        n_a=n_a,
        n_b=n_b,
        estimate=probability_of_superiority,
        estimate_name="P(superiority)",
        assumptions=(
            CLUSTERING_ASSUMPTION,
            "Observations are at least ordinal and comparable between groups.",
            "Reading the result as a difference in medians additionally requires that the two "
            "distributions have the same shape and spread. Without that, a rejection means the "
            "groups are ordered differently, not that their medians differ.",
        ),
        warnings=tuple(warning_list),
        notes=(
            "A difference in spread or shape alone can produce a small p-value here, so a "
            "rejection is not by itself evidence of a location shift.",
        ),
    )


def _stochastic_alternative(alternative: Alternative, label_a: str, label_b: str) -> str:
    if alternative == "two-sided":
        return f"{label_a} and {label_b} are not stochastically equal"
    if alternative == "less":
        return f"{label_a} tends to produce smaller values than {label_b}"
    return f"{label_a} tends to produce larger values than {label_b}"


def kolmogorov_smirnov_test(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    alternative: Alternative = "two-sided",
    label_a: str = "A",
    label_b: str = "B",
) -> HypothesisTestResult:
    """Two-sample Kolmogorov-Smirnov test of whether two distributions differ.

    The statistic D is the largest vertical gap between the two empirical
    cumulative distribution functions, so it is both the test statistic and a
    directly interpretable measure of how far apart the distributions are.

    Appropriate when the question is about the *whole distribution* rather than
    one summary of it: two samples can have the same mean, the same median and
    still be shaped very differently, and only a test like this will notice.

    The one-sided alternatives concern the ordering of the CDFs, not of the
    means: ``"less"`` means the CDF of ``a`` does not lie below that of ``b``.
    """
    clean_a, clean_b, warnings = _clean_pair(a, b, label_a=label_a, label_b=label_b)
    result = stats.ks_2samp(clean_a, clean_b, alternative=alternative)

    warning_list = list(warnings)
    combined = np.concatenate([clean_a, clean_b])
    if combined.size != np.unique(combined).size:
        warning_list.append(
            "The samples contain tied values. The two-sample KS test assumes continuous data; "
            "with ties the p-value is conservative."
        )

    return HypothesisTestResult(
        test_name="Two-sample Kolmogorov-Smirnov test",
        design="independent samples",
        statistic_name="D",
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        alternative=alternative,
        null_hypothesis=f"{label_a} and {label_b} are drawn from the same distribution",
        alternative_hypothesis=(
            f"{label_a} and {label_b} are drawn from different distributions"
            if alternative == "two-sided"
            else f"the cumulative distribution of {label_a} is not {alternative} than "
            f"that of {label_b}"
        ),
        estimand="supremum distance between the two cumulative distribution functions",
        n_total=int(clean_a.size + clean_b.size),
        n_a=int(clean_a.size),
        n_b=int(clean_b.size),
        estimate=float(result.statistic),
        estimate_name="D (max CDF gap)",
        assumptions=(
            CLUSTERING_ASSUMPTION,
            "The underlying distributions are continuous; the exact p-value assumes no ties.",
        ),
        notes=(
            "Sensitive to any difference in location, scale or shape, which makes a rejection "
            "hard to attribute to any one of them. It has most power near the centre of the "
            "distributions and little in the tails.",
        ),
    )


def chi_square_independence_test(
    table: npt.ArrayLike | pd.DataFrame,
    *,
    correction: bool = True,
) -> HypothesisTestResult:
    """Chi-square test of independence for a contingency table of counts.

    Appropriate for asking whether two categorical variables are associated, when
    every observation falls in exactly one cell and the expected counts are large
    enough for the chi-square approximation to hold.

    ``correction`` applies Yates' continuity correction, which SciPy uses for
    2x2 tables by default. For small 2x2 tables :func:`fisher_exact_test` avoids
    the approximation entirely.
    """
    counts = _as_contingency_table(table)
    result = stats.chi2_contingency(counts, correction=correction)
    expected = np.asarray(result.expected_freq, dtype=np.float64)

    warning_list: list[str] = []
    small_cells = int((expected < MIN_EXPECTED_COUNT).sum())
    if small_cells:
        warning_list.append(
            f"{small_cells} of {expected.size} cells have an expected count below "
            f"{MIN_EXPECTED_COUNT:g} (minimum {expected.min():.2f}). The chi-square "
            "approximation is unreliable here; consider an exact or Monte Carlo test."
        )
    if counts.shape == (2, 2) and correction:
        warning_list.append(
            "Yates' continuity correction was applied because the table is 2x2. It makes the "
            "test conservative; Fisher's exact test is the usual alternative for small counts."
        )

    total = int(counts.sum())
    return HypothesisTestResult(
        test_name="Chi-square test of independence",
        design="independent samples",
        statistic_name="chi2",
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        alternative="two-sided",
        null_hypothesis="the row and column variables are independent",
        alternative_hypothesis="the row and column variables are associated",
        estimand="departure of the joint distribution from the product of its margins",
        n_total=total,
        table_shape=(int(counts.shape[0]), int(counts.shape[1])),
        degrees_of_freedom=float(result.dof),
        assumptions=(
            CLUSTERING_ASSUMPTION,
            "Every observation contributes to exactly one cell; the counts are frequencies, "
            "not proportions, rates or repeated measurements of the same unit.",
            "Expected cell counts are large enough (conventionally at least "
            f"{MIN_EXPECTED_COUNT:g}).",
        ),
        warnings=tuple(warning_list),
        notes=(
            "chi2 grows with the sample size, so it is not an effect size: with enough data a "
            "negligible association is significant. Report a measure of association alongside it.",
            "Association is not causation; a table cannot rule out a common cause.",
        ),
    )


def fisher_exact_test(
    table: npt.ArrayLike | pd.DataFrame,
    *,
    alternative: Alternative = "two-sided",
) -> HypothesisTestResult:
    """Fisher's exact test for a 2x2 contingency table.

    Appropriate when counts are small enough that the chi-square approximation is
    doubtful. The p-value is exact given both margins, which is also its main
    subtlety: conditioning on margins that were not actually fixed by the design
    makes the test somewhat conservative.

    The estimate is SciPy's conditional maximum-likelihood odds ratio, which is
    0 or infinite when a cell is empty.
    """
    counts = _as_contingency_table(table)
    if counts.shape != (2, 2):
        raise StatisticalTestError(
            f"Fisher's exact test as implemented in SciPy handles 2x2 tables only, got "
            f"{counts.shape}. Use chi_square_independence_test for larger tables."
        )

    result = stats.fisher_exact(counts, alternative=alternative)

    warning_list: list[str] = []
    if (counts == 0).any():
        warning_list.append(
            "The table contains an empty cell, so the odds ratio is 0 or infinite and its "
            "confidence interval is unbounded. The p-value remains valid."
        )

    total = int(counts.sum())
    return HypothesisTestResult(
        test_name="Fisher's exact test",
        design="independent samples",
        statistic_name="odds ratio",
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        alternative=alternative,
        null_hypothesis="the row and column variables are independent",
        alternative_hypothesis=(
            "the row and column variables are associated"
            if alternative == "two-sided"
            else f"the odds ratio is {alternative} than 1"
        ),
        estimand="odds ratio between the two rows",
        n_total=total,
        table_shape=(2, 2),
        estimate=float(result.statistic),
        estimate_name="odds ratio",
        assumptions=(
            CLUSTERING_ASSUMPTION,
            "Every observation contributes to exactly one cell.",
            "The p-value is computed conditional on both row and column margins.",
        ),
        warnings=tuple(warning_list),
        notes=(
            "Exact means the null distribution is computed exactly, not that the result is more "
            "correct than a well-specified approximation. Conditioning on both margins makes it "
            "conservative when the margins were not fixed by design.",
        ),
    )


# --------------------------------------------------------------------------- #
# Paired designs
# --------------------------------------------------------------------------- #


def _clean_pairs(
    a: npt.ArrayLike, b: npt.ArrayLike, *, label_a: str, label_b: str
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], tuple[str, ...]]:
    """Align two samples into complete matched pairs.

    Raises:
        StatisticalTestError: if the samples cannot form pairs at all.
    """
    array_a = to_float_sample(a)
    array_b = to_float_sample(b)
    if array_a.size != array_b.size:
        raise StatisticalTestError(
            f"A paired test needs a one-to-one correspondence, but {label_a} has "
            f"{array_a.size} values and {label_b} has {array_b.size}. If the observations "
            "cannot be matched into pairs by design, the design is not paired and an "
            "independent-samples test is the appropriate choice."
        )

    complete = np.isfinite(array_a) & np.isfinite(array_b)
    warnings: list[str] = []
    dropped = int((~complete).sum())
    if dropped:
        warnings.append(
            f"Dropped {dropped} incomplete pair(s): a pair is unusable if either side is "
            "missing. Check whether the pairs that survived differ systematically from the "
            "ones that did not."
        )
    clean_a = array_a[complete]
    clean_b = array_b[complete]
    if clean_a.size < 2:
        raise StatisticalTestError(
            f"Only {clean_a.size} complete pair(s) remain; at least 2 are needed."
        )
    return clean_a, clean_b, tuple(warnings)


def paired_t_test(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    alternative: Alternative = "two-sided",
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    label_a: str = "A",
    label_b: str = "B",
) -> HypothesisTestResult:
    """Paired-samples t-test on the within-pair differences.

    Appropriate when the same units are measured twice -- before and after, under
    two conditions, or in two periods -- so that observation *i* of one sample
    corresponds to observation *i* of the other **by design**. The pairing must
    come from how the data was collected; two samples that merely happen to be
    the same length are not paired.

    The analysis is of the *n* differences, so the effective sample size is the
    number of pairs. In exchange, every stable difference between units cancels
    out, which is what makes a paired design more powerful than an independent
    one when the pairing is real.
    """
    clean_a, clean_b, warnings = _clean_pairs(a, b, label_a=label_a, label_b=label_b)
    result = stats.ttest_rel(clean_a, clean_b, alternative=alternative)
    interval = result.confidence_interval(confidence_level=confidence_level)
    differences = clean_a - clean_b
    n_pairs = int(differences.size)

    warning_list = list(warnings)
    if float(np.ptp(differences)) == 0.0:
        warning_list.append(
            "Every pair has the same difference, so the within-pair variance is zero and the "
            "t statistic is infinite. That is arithmetic, not evidence: with no variability "
            "there is nothing to base a standard error on."
        )
    elif n_pairs < SMALL_GROUP_N and n_pairs >= 3:
        skewness = float(stats.skew(differences, bias=False))
        if abs(skewness) > 1.0:
            warning_list.append(
                f"Only {n_pairs} pairs with difference skewness {skewness:+.2f}. The central "
                "limit theorem does little work here, so the t approximation rests on the "
                "shape of the differences."
            )

    return HypothesisTestResult(
        test_name="Paired t-test",
        design="paired samples",
        statistic_name="t",
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        alternative=alternative,
        null_hypothesis=(
            f"the mean of the within-pair differences ({label_a} - {label_b}) is zero"
        ),
        alternative_hypothesis=_paired_alternative(alternative, label_a, label_b),
        estimand=f"mean within-pair difference, {label_a} - {label_b}",
        n_total=n_pairs * 2,
        n_pairs=n_pairs,
        degrees_of_freedom=float(result.df),
        estimate=float(np.mean(differences)),
        estimate_name="mean difference",
        confidence_interval=(float(interval.low), float(interval.high)),
        confidence_level=confidence_level,
        assumptions=(
            "The pairing is real: each pair is one unit measured twice, matched by design "
            "rather than by position in an array.",
            "The pairs are independent of one another. The two measurements within a pair are "
            "expected to be correlated -- that is the point of the design -- but two different "
            "pairs must not be linked.",
            "The within-pair differences are approximately normal, or there are enough pairs "
            "for the central limit theorem to apply to their mean. The original measurements "
            "need not be normal.",
        ),
        warnings=tuple(warning_list),
        notes=(
            f"The effective sample size is {n_pairs} pairs, not {n_pairs * 2} measurements: "
            f"the test has {n_pairs - 1} degrees of freedom.",
        ),
    )


def _paired_alternative(alternative: Alternative, label_a: str, label_b: str) -> str:
    if alternative == "two-sided":
        return f"the mean within-pair difference ({label_a} - {label_b}) is not zero"
    if alternative == "less":
        return f"{label_a} is on average smaller than {label_b} within pairs"
    return f"{label_a} is on average larger than {label_b} within pairs"


def _signed_rank_alternative(alternative: Alternative, label_a: str, label_b: str) -> str:
    """Phrase the alternative in signed-rank terms, never as a claim about a mean."""
    if alternative == "two-sided":
        return (
            f"the within-pair differences ({label_a} - {label_b}) are not distributed "
            "symmetrically about zero"
        )
    if alternative == "less":
        return f"{label_a} tends to be smaller than {label_b} within pairs"
    return f"{label_a} tends to be larger than {label_b} within pairs"


def wilcoxon_signed_rank_test(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    alternative: Alternative = "two-sided",
    label_a: str = "A",
    label_b: str = "B",
) -> HypothesisTestResult:
    """Wilcoxon signed-rank test: the rank-based counterpart of the paired t-test.

    Its null is that the distribution of the within-pair differences is symmetric
    about zero. **It is not simply a test of whether the median difference is
    zero**: reading it that way requires the additional assumption that the
    difference distribution is symmetric, which the test neither makes nor checks.

    Appropriate for paired designs where the differences are ordinal, heavy
    tailed, or otherwise poorly summarised by a mean.

    Pairs with a difference of exactly zero are discarded by SciPy's default
    handling, which reduces the effective sample size.
    """
    clean_a, clean_b, warnings = _clean_pairs(a, b, label_a=label_a, label_b=label_b)
    result = stats.wilcoxon(clean_a, clean_b, alternative=alternative)
    differences = clean_a - clean_b
    n_pairs = int(differences.size)

    warning_list = list(warnings)
    zero_differences = int((differences == 0).sum())
    if zero_differences:
        warning_list.append(
            f"{zero_differences} pair(s) have a difference of exactly zero and are dropped by "
            "the signed-rank procedure, so the test uses "
            f"{n_pairs - zero_differences} of {n_pairs} pairs."
        )
    magnitudes = np.abs(differences[differences != 0])
    if magnitudes.size != np.unique(magnitudes).size:
        warning_list.append(
            "Tied absolute differences are present; SciPy uses the tie-corrected normal "
            "approximation rather than the exact distribution."
        )

    return HypothesisTestResult(
        test_name="Wilcoxon signed-rank test",
        design="paired samples",
        statistic_name="W",
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        alternative=alternative,
        null_hypothesis=(
            f"the within-pair differences ({label_a} - {label_b}) are distributed symmetrically "
            "about zero"
        ),
        alternative_hypothesis=_signed_rank_alternative(alternative, label_a, label_b),
        estimand=(
            f"location of the within-pair difference distribution ({label_a} - {label_b}) "
            "relative to zero, in signed-rank terms"
        ),
        n_total=n_pairs * 2,
        n_pairs=n_pairs,
        estimate=float(np.median(differences)),
        estimate_name="median difference",
        assumptions=(
            "The pairing is real and the pairs are independent of one another.",
            "The differences are at least ordinal and their magnitudes are comparable across "
            "pairs, since the procedure ranks them against each other.",
            "Reading the result as a statement about the median difference additionally "
            "requires the difference distribution to be symmetric. Without symmetry, a "
            "rejection means the signed ranks are unbalanced, not that the median is non-zero.",
        ),
        warnings=tuple(warning_list),
        notes=(
            "The reported median difference is descriptive. The estimator that corresponds to "
            "this test is the Hodges-Lehmann pseudomedian, the median of all pairwise averages "
            "of the differences.",
        ),
    )


# --------------------------------------------------------------------------- #
# Multi-group designs
# --------------------------------------------------------------------------- #


def group_samples(
    frame: pd.DataFrame, value_column: str, group_column: str
) -> dict[str, npt.NDArray[np.float64]]:
    """Split ``value_column`` into one array per level of ``group_column``.

    A convenience for feeding a tidy DataFrame to the multi-group tests. Groups
    come back in order of decreasing size.
    """
    for column in (value_column, group_column):
        if column not in frame.columns:
            raise KeyError(f"{column!r} is not a column of the frame.")
    ordered = frame[group_column].value_counts().index
    return {
        str(level): to_float_sample(frame.loc[frame[group_column] == level, value_column])
        for level in ordered
    }


def _clean_groups(
    groups: Mapping[str, npt.ArrayLike], *, minimum_groups: int = 2
) -> tuple[tuple[str, ...], tuple[npt.NDArray[np.float64], ...], tuple[str, ...]]:
    """Validate and clean a mapping of group name to sample."""
    if len(groups) < minimum_groups:
        raise StatisticalTestError(
            f"At least {minimum_groups} groups are needed, got {len(groups)}."
        )

    names: list[str] = []
    samples: list[npt.NDArray[np.float64]] = []
    warnings: list[str] = []
    dropped_total = 0
    for name, values in groups.items():
        array = to_float_sample(values)
        clean = array[np.isfinite(array)]
        dropped_total += array.size - clean.size
        if clean.size < 2:
            raise StatisticalTestError(
                f"Group {name!r} has {clean.size} usable observation(s); at least 2 are needed."
            )
        names.append(str(name))
        samples.append(clean)

    if dropped_total:
        warnings.append(f"Dropped {dropped_total} non-finite observation(s) across all groups.")
    if len(groups) == minimum_groups == 2:
        warnings.append(
            "With only two groups this is equivalent to a t-test; the multi-group machinery "
            "adds nothing."
        )
    return tuple(names), tuple(samples), tuple(warnings)


def _spread_warnings(names: Sequence[str], samples: Sequence[npt.NDArray[np.float64]]) -> list[str]:
    """Warn when the groups look too unequal for the equal-variance assumption."""
    variances = [float(np.var(sample, ddof=1)) for sample in samples]
    sizes = [sample.size for sample in samples]
    warnings: list[str] = []
    if min(variances) > 0:
        ratio = max(variances) / min(variances)
        if ratio > VARIANCE_RATIO_WARNING:
            widest = names[int(np.argmax(variances))]
            narrowest = names[int(np.argmin(variances))]
            warnings.append(
                f"Group variances differ by a factor of {ratio:.2f} ({widest} is the widest, "
                f"{narrowest} the narrowest), which strains the equal-variance assumption. "
                "Welch's ANOVA does not make it."
            )
            if max(sizes) / min(sizes) > VARIANCE_RATIO_WARNING:
                warnings.append(
                    f"Group sizes are also unbalanced ({min(sizes):,} to {max(sizes):,}). "
                    "Unequal variances combined with unequal group sizes is the configuration "
                    "in which the classic F test is least trustworthy."
                )
    for name, sample in zip(names, samples, strict=True):
        if sample.size < SMALL_GROUP_N:
            warnings.append(f"Group {name!r} has only {sample.size} observations.")
    return warnings


MULTI_GROUP_NOTES: Final = (
    "F is not an effect size: it grows with the sample size. Report a measure such as "
    "eta-squared or omega-squared alongside it.",
    "A small p-value says at least one group differs from the others, not which. Follow-up "
    "comparisons need multiple-testing control.",
)

CLUSTERED_ANOVA_ASSUMPTION: Final = (
    "Observations within each group are independent draws. Shots are not: one player "
    "contributes many, so residuals are correlated within player, the within-group mean "
    "square is too small, and F is inflated. See the module docstring."
)


def one_way_anova(groups: Mapping[str, npt.ArrayLike]) -> HypothesisTestResult:
    """Classic one-way ANOVA (Fisher's F test) across three or more groups.

    Appropriate when independent units are split into several groups and the
    question is whether *any* group mean differs. It assumes equal variances
    across groups; when that is doubtful, :func:`welch_anova` is the safer test,
    and this function warns when the sample variances disagree.
    """
    names, samples, warnings = _clean_groups(groups)
    result = stats.f_oneway(*samples)
    sizes = tuple(int(sample.size) for sample in samples)
    total = int(sum(sizes))

    return HypothesisTestResult(
        test_name="One-way ANOVA",
        design="independent multi-group",
        statistic_name="F",
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        alternative="two-sided",
        null_hypothesis="all group means are equal",
        alternative_hypothesis="at least one group mean differs from the others",
        estimand="variation between group means relative to variation within groups",
        n_total=total,
        group_names=names,
        group_sizes=sizes,
        degrees_of_freedom=float(len(samples) - 1),
        assumptions=(
            CLUSTERED_ANOVA_ASSUMPTION,
            "All groups share a common variance.",
            "Residuals are approximately normal, or the groups are large enough for the "
            "central limit theorem to cover the group means.",
        ),
        warnings=(*warnings, *_spread_warnings(names, samples)),
        notes=MULTI_GROUP_NOTES,
    )


def welch_anova(groups: Mapping[str, npt.ArrayLike]) -> HypothesisTestResult:
    """Welch's ANOVA: a one-way comparison that does not assume equal variances.

    The multi-group analogue of Welch's t-test, and the sensible default whenever
    the groups may differ in spread or size. Uses the statsmodels implementation.
    """
    from statsmodels.stats.oneway import anova_oneway

    names, samples, warnings = _clean_groups(groups)
    result = anova_oneway(list(samples), use_var="unequal")
    sizes = tuple(int(sample.size) for sample in samples)

    return HypothesisTestResult(
        test_name="Welch's ANOVA",
        design="independent multi-group",
        statistic_name="F",
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        alternative="two-sided",
        null_hypothesis="all group means are equal",
        alternative_hypothesis="at least one group mean differs from the others",
        estimand="variation between group means, weighted by the precision of each",
        n_total=int(sum(sizes)),
        group_names=names,
        group_sizes=sizes,
        degrees_of_freedom=float(result.df_denom),
        assumptions=(
            CLUSTERED_ANOVA_ASSUMPTION,
            "Group variances may differ; the denominator degrees of freedom are adjusted and "
            "are usually fractional.",
            "Residuals are approximately normal, or the groups are large enough for the "
            "central limit theorem to cover the group means.",
        ),
        warnings=tuple(warnings),
        notes=(
            *MULTI_GROUP_NOTES,
            "Because each group is weighted by its own precision, this F can be larger or "
            "smaller than the classic one on the same data.",
        ),
    )


def kruskal_wallis_test(groups: Mapping[str, npt.ArrayLike]) -> HypothesisTestResult:
    """Kruskal-Wallis H test: the rank-based multi-group comparison.

    The extension of the Mann-Whitney U test to three or more groups, and it
    inherits the same caveat: **it is not a test of medians.** Its null is that
    all groups are stochastically equal, and groups with identical medians but
    different shapes can be rejected.

    Appropriate when the outcome is ordinal or heavy tailed and the question is
    whether some groups tend to produce larger values than others.
    """
    names, samples, warnings = _clean_groups(groups)
    result = stats.kruskal(*samples)
    sizes = tuple(int(sample.size) for sample in samples)

    warning_list = list(warnings)
    combined = np.concatenate(samples)
    if combined.size != np.unique(combined).size:
        warning_list.append(
            "Tied values are present across groups; the tie-corrected statistic is used."
        )

    return HypothesisTestResult(
        test_name="Kruskal-Wallis H test",
        design="independent multi-group",
        statistic_name="H",
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        alternative="two-sided",
        null_hypothesis=(
            "all groups are stochastically equal: a random observation from any group is as "
            "likely to exceed a random observation from any other as the reverse"
        ),
        alternative_hypothesis="at least one group tends to produce larger values than another",
        estimand="differences in mean rank between groups",
        n_total=int(sum(sizes)),
        group_names=names,
        group_sizes=sizes,
        degrees_of_freedom=float(len(samples) - 1),
        assumptions=(
            CLUSTERED_ANOVA_ASSUMPTION,
            "Observations are at least ordinal and comparable across groups.",
            "Reading the result as a comparison of medians additionally requires the groups to "
            "have the same shape and spread.",
        ),
        warnings=tuple(warning_list),
        notes=(
            "A difference in spread or shape alone can produce a small p-value, so a rejection "
            "is not by itself evidence that the groups are centred differently.",
        ),
    )


# --------------------------------------------------------------------------- #
# Football demonstrations
# --------------------------------------------------------------------------- #


def _print_section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main(argv: Sequence[str] | None = None) -> int:
    """Demonstrate the tests on real football questions, with data-driven groups."""
    from football_intelligence.features import shots as shot_features

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        prog="python -m football_intelligence.statistics.tests",
        description="Run demonstration comparisons on the canonical shot dataset.",
    )
    parser.add_argument("--dataset", type=Path, default=shot_features.DEFAULT_DATASET_PATH)
    arguments = parser.parse_args(argv)

    try:
        dataset = shot_features.read_shot_dataset(arguments.dataset)
    except shot_features.ShotFeatureError as error:
        logger.error("%s", error)
        return 1

    # Groups are picked from the data, never by name.
    open_play = dataset[dataset["shot_type"] == "Open Play"]
    busiest = open_play["team"].value_counts().head(2)
    team_a, team_b = (str(name) for name in busiest.index)
    distance_a = open_play.loc[open_play["team"] == team_a, "shot_distance"]
    distance_b = open_play.loc[open_play["team"] == team_b, "shot_distance"]

    _print_section(
        f"1. Open-play shot distance: {team_a} vs {team_b}\n"
        f"   (the two teams with the most open-play shots: "
        f"{busiest.iloc[0]} and {busiest.iloc[1]})"
    )
    print("Question: do these teams shoot from different distances on average?")
    print()
    print(welch_t_test(distance_a, distance_b, label_a=team_a, label_b=team_b))
    print()
    print("The same two samples, asking a different question -- which team tends to")
    print("shoot from further out, regardless of the mean:")
    print()
    print(mann_whitney_u_test(distance_a, distance_b, label_a=team_a, label_b=team_b))

    _print_section(f"2. Whole distance distribution: {team_a} vs {team_b}")
    print("Question: are the shot-distance distributions the same shape, not just")
    print("the same on average?")
    print()
    print(kolmogorov_smirnov_test(distance_a, distance_b, label_a=team_a, label_b=team_b))

    _print_section("3. Goal outcome by body part (open play only)")
    body_parts = open_play["body_part"].value_counts()
    kept = [str(part) for part in body_parts[body_parts >= 30].index]
    subset = open_play[open_play["body_part"].isin(kept)]
    table = pd.crosstab(subset["body_part"], subset["goal"])
    print("Question: is scoring associated with the body part used?")
    print()
    print(table.to_string())
    print()
    print(chi_square_independence_test(table))

    _print_section("4. Headers versus foot shots, as a 2x2 table")
    binary = open_play.assign(
        header=open_play["body_part"] == "Head",
    )
    binary = binary[binary["body_part"].isin(["Head", "Right Foot", "Left Foot"])]
    two_by_two = pd.crosstab(binary["header"], binary["goal"])
    print(two_by_two.to_string())
    print()
    print(fisher_exact_test(two_by_two))
    print()
    print(
        "Both tests describe an association only. Headers are taken from a mean of\n"
        "9.8 yards and foot shots from about 21, so body part is confounded with\n"
        "distance: this says nothing about whether heading causes a different\n"
        "conversion rate at a comparable chance."
    )

    _demonstrate_paired_design(dataset)
    _demonstrate_multi_group_design(dataset)
    return 0


def _demonstrate_paired_design(dataset: pd.DataFrame) -> None:
    """A genuinely paired football comparison: one team, two phases of one tournament."""
    _print_section("5. PAIRED DESIGN: group stage vs knockout, within the same team")
    print(
        "Unit of analysis: a team in one tournament. Each unit is measured twice --\n"
        "its mean shot distance in the group stage and in the knockout rounds -- so\n"
        "the pairing comes from the design, not from lining up two arrays.\n"
        "Every team is its own control, which removes the large differences in style\n"
        "between teams."
    )

    phased = dataset.assign(phase=np.where(dataset["stage"] == "Group Stage", "group", "knockout"))
    per_phase = phased.groupby(["competition_id", "team", "phase"], as_index=False).agg(
        mean_distance=("shot_distance", "mean"), shots=("shot_distance", "size")
    )
    paired = per_phase[per_phase["phase"] == "group"].merge(
        per_phase[per_phase["phase"] == "knockout"],
        on=["competition_id", "team"],
        suffixes=("_group", "_knockout"),
    )
    # Only teams that reached the knockout rounds can have both measurements, and a
    # mean over one or two shots is too noisy to treat as a measurement.
    paired = paired[(paired["shots_group"] >= 5) & (paired["shots_knockout"] >= 5)]

    print(f"\n{len(paired)} team-tournaments have at least 5 shots in each phase.\n")
    print(
        paired[["team", "mean_distance_group", "mean_distance_knockout"]]
        .head(6)
        .round(2)
        .to_string(index=False)
    )
    print()
    print(
        paired_t_test(
            paired["mean_distance_group"],
            paired["mean_distance_knockout"],
            label_a="group stage",
            label_b="knockout",
        )
    )
    print()
    print(
        wilcoxon_signed_rank_test(
            paired["mean_distance_group"],
            paired["mean_distance_knockout"],
            label_a="group stage",
            label_b="knockout",
        )
    )
    print()
    print(
        "Limits of this design:\n"
        "  - Only teams that qualified can appear, so the result describes qualifiers.\n"
        "  - Knockout matches are between two teams that are both in the sample, so the\n"
        "    pairs are not fully independent of each other.\n"
        "  - Each unit's mean is an average over 5 to 76 shots, so the pair differences\n"
        "    have unequal precision while the paired t-test weights them equally.\n"
        "\n"
        "What would NOT be a paired design: comparing headers with footed shots and\n"
        "calling it paired because both come from the same players. There is no 1:1\n"
        "correspondence -- players take different numbers of each, many take only one\n"
        "kind -- so there is nothing to difference. That comparison is the independent\n"
        "multi-group design shown next."
    )


def _demonstrate_multi_group_design(dataset: pd.DataFrame) -> None:
    """Multi-group comparison, and what clustering does to it."""
    _print_section("6. MULTI-GROUP DESIGN: shot distance by body part")

    counted = dataset["body_part"].value_counts()
    kept = [str(part) for part in counted[counted >= 30].index]
    subset = dataset[dataset["body_part"].isin(kept)]
    groups = group_samples(subset, "shot_distance", "body_part")

    print("Question: do shots with different body parts come from different distances?")
    print()
    for name, sample in groups.items():
        print(
            f"  {name:<12} n={sample.size:>5,}  mean={sample.mean():6.2f}  "
            f"sd={sample.std(ddof=1):5.2f}"
        )
    print()
    print(one_way_anova(groups))
    print()
    print(welch_anova(groups))
    print()
    print(kruskal_wallis_test(groups))

    _print_section("7. What clustering does to that ANOVA")
    classic = one_way_anova(groups)

    player_means = subset.groupby(["player_id", "body_part"], as_index=False).agg(
        mean_distance=("shot_distance", "mean")
    )
    aggregated = one_way_anova(group_samples(player_means, "mean_distance", "body_part"))
    spanning = player_means.groupby("player_id").size()
    multiple_groups = int((spanning > 1).sum())

    print(
        f"one shot per row      F = {classic.statistic:8.2f}   p = {classic.p_value:.3g}   "
        f"n = {classic.n_total:,}"
    )
    print(
        f"one player-mean/row   F = {aggregated.statistic:8.2f}   p = {aggregated.p_value:.3g}   "
        f"n = {aggregated.n_total:,}"
    )
    print()
    print(
        "The shot-level F is inflated because a player's shots are not independent\n"
        "draws: 651 players supply 2,900 shots, so the within-group mean square is\n"
        "measuring variation within players as if it were variation between them.\n"
        "\n"
        f"Aggregating is not a fix either. {multiple_groups} of {len(spanning)} players appear in\n"
        "more than one body-part group, so the aggregated comparison is a repeated-\n"
        "measures design, not a one-way one. Neither number above is a valid test.\n"
        "The valid analysis is a mixed model with a player random intercept (T28), or\n"
        "inference that resamples whole players (T09/T10)."
    )


if __name__ == "__main__":
    raise SystemExit(main())
