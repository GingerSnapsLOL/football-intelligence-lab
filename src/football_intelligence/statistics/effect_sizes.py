"""Effect sizes: how big is it, and how well do we know?

A p-value answers "could this have arisen by chance?". It never answers "does it
matter?", and it cannot: with enough data any non-zero difference becomes
statistically significant, and with too little data an important one goes
undetected. Every test in this package therefore needs a companion here, and
every companion comes with an interval.

Two kinds of measure
--------------------

**Unstandardised** measures stay in the units of the problem: a mean difference
in yards, a risk difference in percentage points, an odds ratio. They are the
easier to interpret and should be the default whenever the units mean something
to the reader -- "two yards closer" is a sentence a coach can act on, while "a
Cohen's d of 0.25" is not.

**Standardised** measures divide by a measure of spread so that quantities in
different units can be compared: Cohen's d, Cramer's V, the probability of
superiority. They are useful for comparing across studies and necessary when the
raw units are arbitrary, but they inherit whatever the denominator does. A d of
0.5 in a sample with an unusually narrow spread is not the same finding as a d of
0.5 in a broad one.

On benchmark labels
-------------------

Cohen's suggestions that d = 0.2/0.5/0.8 are "small/medium/large" are convenient
and widely abused. Cohen offered them for use when nothing better was available
and warned against exactly the mechanical application they now receive. Nothing in
this module returns such a label: what counts as a large effect is a question
about football, not about arithmetic, and an eleven-yard difference in shooting
distance matters regardless of what it standardises to.

Confidence intervals
--------------------

Every measure here reports one, because a point estimate without an interval
recreates the problem the effect size was supposed to solve. The interval comes
from a trusted implementation where a good one exists (SciPy for odds ratios,
statsmodels for risk differences, the Welch t interval for mean differences) and
otherwise from the project's own bootstrap, resampling **within groups** so that
the group sizes stay fixed by design.
"""

import logging
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats
from scipy.special import gammaln
from scipy.stats.contingency import odds_ratio as scipy_odds_ratio

from football_intelligence.statistics.bootstrap import DEFAULT_RESAMPLES, bootstrap
from football_intelligence.statistics.diagnostics import to_float_sample

logger = logging.getLogger(__name__)

DEFAULT_CONFIDENCE_LEVEL: Final = 0.95

OddsRatioKind = Literal["conditional", "sample"]


class EffectSizeError(ValueError):
    """Raised when an effect size cannot be computed from the data as supplied."""


@dataclass(frozen=True, slots=True)
class EffectSize:
    """One effect-size estimate with its uncertainty.

    ``null_value`` is the value the measure takes when there is no effect: 0 for
    differences and correlations, 1 for ratios, 0.5 for the probability of
    superiority. Comparing the interval to it shows at a glance whether "no
    effect" remains plausible -- without turning the interval into a hypothesis
    test, which would discard the information about magnitude it exists to carry.
    """

    name: str
    estimate: float
    null_value: float
    confidence_interval: tuple[float, float] | None = None
    confidence_level: float | None = None
    interval_method: str | None = None
    units: str | None = None
    n_a: int | None = None
    n_b: int | None = None
    n_total: int | None = None
    notes: tuple[str, ...] = ()

    @property
    def interval_excludes_null(self) -> bool | None:
        """Whether the interval excludes the no-effect value, or None if unknown."""
        if self.confidence_interval is None:
            return None
        low, high = self.confidence_interval
        return not (low <= self.null_value <= high)

    @property
    def interval_width(self) -> float | None:
        if self.confidence_interval is None:
            return None
        low, high = self.confidence_interval
        return high - low

    def __str__(self) -> str:
        units = f" {self.units}" if self.units else ""
        lines = [f"{self.name}", f"  estimate          {self.estimate:.4g}{units}"]
        if self.confidence_interval is not None and self.confidence_level is not None:
            low, high = self.confidence_interval
            lines.append(
                f"  {self.confidence_level:.0%} CI            "
                f"[{low:.4g}, {high:.4g}]{units}   ({self.interval_method})"
            )
            verdict = "excludes" if self.interval_excludes_null else "includes"
            lines.append(f"  no effect = {self.null_value:g}     the interval {verdict} it")
        if self.n_a is not None and self.n_b is not None:
            lines.append(f"  n                 {self.n_a:,} vs {self.n_b:,}")
        elif self.n_total is not None:
            lines.append(f"  n                 {self.n_total:,}")
        lines.extend(f"  note              {item}" for item in self.notes)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #


def _clean_pair(
    a: npt.ArrayLike, b: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    array_a = to_float_sample(a)
    array_b = to_float_sample(b)
    clean_a = array_a[np.isfinite(array_a)]
    clean_b = array_b[np.isfinite(array_b)]
    for clean, label in ((clean_a, "a"), (clean_b, "b")):
        if clean.size < 2:
            raise EffectSizeError(
                f"Group {label!r} has {clean.size} usable observation(s); at least 2 are needed."
            )
    return clean_a, clean_b


def _two_by_two(table: npt.ArrayLike | pd.DataFrame) -> npt.NDArray[np.int64]:
    counts = _counts(table)
    if counts.shape != (2, 2):
        raise EffectSizeError(f"A 2x2 table is required, got shape {counts.shape}.")
    return counts


def _counts(table: npt.ArrayLike | pd.DataFrame) -> npt.NDArray[np.int64]:
    array = np.asarray(table.to_numpy() if isinstance(table, pd.DataFrame) else table)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 2:
        raise EffectSizeError(
            f"A contingency table of at least 2x2 is required, got {array.shape}."
        )
    if not np.issubdtype(array.dtype, np.integer):
        if not np.allclose(array, np.round(array)):
            raise EffectSizeError("Contingency tables must contain counts, not fractions.")
        array = np.round(array).astype(np.int64)
    if (array < 0).any():
        raise EffectSizeError("Contingency tables cannot contain negative counts.")
    if array.sum() == 0:
        raise EffectSizeError("Contingency table is empty.")
    return array.astype(np.int64)


def _bootstrap_interval(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    statistic: object,
    *,
    confidence_level: float,
    n_resamples: int,
    random_state: int | np.random.Generator | None,
) -> tuple[float, float]:
    """Percentile interval for a two-sample statistic, resampling within groups."""
    frame = pd.DataFrame(
        {
            "value": np.concatenate([a, b]),
            "group": np.concatenate([np.zeros(a.size, dtype=int), np.ones(b.size, dtype=int)]),
        }
    )

    def evaluate(sample: pd.DataFrame) -> float:
        left = sample.loc[sample["group"] == 0, "value"].to_numpy()
        right = sample.loc[sample["group"] == 1, "value"].to_numpy()
        return float(statistic(left, right))  # type: ignore[operator]

    result = bootstrap(
        frame,
        evaluate,
        strata=frame["group"].to_numpy(),
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        random_state=random_state,
    )
    return result.confidence_interval


# --------------------------------------------------------------------------- #
# Numeric outcomes
# --------------------------------------------------------------------------- #


def mean_difference(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    units: str | None = None,
) -> EffectSize:
    """``mean(a) - mean(b)``, in the original units, with a Welch t interval.

    The unstandardised measure, and usually the one to lead with: it is the
    quantity a reader can act on. The interval is the Welch one, so it does not
    assume equal variances and matches the interval reported by
    ``tests.welch_t_test``.
    """
    clean_a, clean_b = _clean_pair(a, b)
    interval = stats.ttest_ind(clean_a, clean_b, equal_var=False).confidence_interval(
        confidence_level=confidence_level
    )
    return EffectSize(
        name="mean difference",
        estimate=float(np.mean(clean_a) - np.mean(clean_b)),
        null_value=0.0,
        confidence_interval=(float(interval.low), float(interval.high)),
        confidence_level=confidence_level,
        interval_method="Welch t",
        units=units,
        n_a=int(clean_a.size),
        n_b=int(clean_b.size),
        notes=(
            "Unstandardised: read it in the units of the measurement, which is normally more "
            "informative than a standardised alternative.",
        ),
    )


def _pooled_standard_deviation(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    numerator = (a.size - 1) * np.var(a, ddof=1) + (b.size - 1) * np.var(b, ddof=1)
    return float(np.sqrt(numerator / (a.size + b.size - 2)))


def _cohens_d_value(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64], *, bias_correction: bool
) -> float:
    pooled = _pooled_standard_deviation(a, b)
    if pooled == 0:
        raise EffectSizeError(
            "Both groups have zero spread, so a standardised difference is undefined. Report "
            "the raw mean difference instead."
        )
    d = float((np.mean(a) - np.mean(b)) / pooled)
    if bias_correction:
        degrees = a.size + b.size - 2
        # Hedges' exact correction factor J = G(df/2) / (sqrt(df/2) G((df-1)/2)),
        # computed through log-gammas so it stays stable for large samples.
        log_j = gammaln(degrees / 2) - np.log(np.sqrt(degrees / 2)) - gammaln((degrees - 1) / 2)
        d *= float(np.exp(log_j))
    return d


def cohens_d(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    bias_correction: bool = False,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    n_resamples: int = DEFAULT_RESAMPLES,
    random_state: int | np.random.Generator | None = None,
) -> EffectSize:
    """Standardised mean difference, using the **pooled** standard deviation.

    Convention, stated explicitly because several incompatible ones share the
    name:

    - the numerator is ``mean(a) - mean(b)``, so the sign follows the argument
      order;
    - the denominator is the pooled sample standard deviation,
      ``sqrt(((n_a - 1) s_a^2 + (n_b - 1) s_b^2) / (n_a + n_b - 2))``, which uses
      ``ddof = 1`` within each group. This is Cohen's ``d_s``;
    - it is **not** Glass's delta (which divides by one group's SD) and not the
      version that divides by the average of the two SDs.

    Pooling assumes the two groups have comparable spread. When they do not, the
    denominator describes neither group and the raw
    :func:`mean_difference` is the more honest summary.

    Args:
        bias_correction: Apply Hedges' factor, which removes the upward bias that
            matters below roughly 20 observations per group. The corrected value
            is usually called Hedges' g.

    The interval is a bootstrap percentile interval resampling within each group,
    since the exact interval requires inverting a noncentral t distribution.
    """
    clean_a, clean_b = _clean_pair(a, b)
    estimate = _cohens_d_value(clean_a, clean_b, bias_correction=bias_correction)
    interval = _bootstrap_interval(
        clean_a,
        clean_b,
        lambda x, y: _cohens_d_value(x, y, bias_correction=bias_correction),
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        random_state=random_state,
    )
    notes = [
        "Pooled standard deviation (Cohen's d_s); the sign follows the argument order.",
        "Cohen's 0.2/0.5/0.8 labels are conventions, not facts about football. Judge the raw "
        "difference against what matters in the game.",
    ]
    if bias_correction:
        notes.append("Hedges' small-sample correction applied.")
    if min(clean_a.size, clean_b.size) < 20 and not bias_correction:
        notes.append(
            f"The smaller group has {min(clean_a.size, clean_b.size)} observations, where d is "
            "biased upward; consider bias_correction=True."
        )
    return EffectSize(
        name="Cohen's d" + (" (Hedges' g)" if bias_correction else ""),
        estimate=estimate,
        null_value=0.0,
        confidence_interval=interval,
        confidence_level=confidence_level,
        interval_method="bootstrap percentile, resampled within groups",
        units="pooled SD",
        n_a=int(clean_a.size),
        n_b=int(clean_b.size),
        notes=tuple(notes),
    )


def hedges_g(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    n_resamples: int = DEFAULT_RESAMPLES,
    random_state: int | np.random.Generator | None = None,
) -> EffectSize:
    """Cohen's d with Hedges' small-sample bias correction applied."""
    return cohens_d(
        a,
        b,
        bias_correction=True,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        random_state=random_state,
    )


# --------------------------------------------------------------------------- #
# Rank-based
# --------------------------------------------------------------------------- #


def _probability_of_superiority_value(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> float:
    statistic = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(statistic) / (a.size * b.size)


def probability_of_superiority(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    n_resamples: int = DEFAULT_RESAMPLES,
    random_state: int | np.random.Generator | None = None,
) -> EffectSize:
    """``P(A > B) + 0.5 P(A = B)``: the effect size the Mann-Whitney U test estimates.

    Also called the common-language effect size or Vargha-Delaney A. It is exactly
    the U statistic rescaled to [0, 1], and answers a question anyone can picture:
    pick one observation from each group at random, how often is the first larger?
    0.5 means no ordering at all.

    Unlike Cohen's d it needs no assumption about spread or shape, and unlike a
    difference in medians it is what the rank test actually concerns.
    """
    clean_a, clean_b = _clean_pair(a, b)
    interval = _bootstrap_interval(
        clean_a,
        clean_b,
        _probability_of_superiority_value,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        random_state=random_state,
    )
    return EffectSize(
        name="probability of superiority",
        estimate=_probability_of_superiority_value(clean_a, clean_b),
        null_value=0.5,
        confidence_interval=interval,
        confidence_level=confidence_level,
        interval_method="bootstrap percentile, resampled within groups",
        n_a=int(clean_a.size),
        n_b=int(clean_b.size),
        notes=(
            "The estimand of the Mann-Whitney U test. It describes ordering, not a difference "
            "in medians, and needs no assumption about the shapes of the two distributions.",
        ),
    )


def rank_biserial_correlation(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    n_resamples: int = DEFAULT_RESAMPLES,
    random_state: int | np.random.Generator | None = None,
) -> EffectSize:
    """``2 * P(superiority) - 1``, on a correlation-like scale from -1 to +1.

    The same information as :func:`probability_of_superiority`, rescaled so that 0
    means no ordering and the sign shows the direction. Useful when a reader
    expects an effect size centred at zero.
    """
    clean_a, clean_b = _clean_pair(a, b)

    def value(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
        return 2.0 * _probability_of_superiority_value(x, y) - 1.0

    interval = _bootstrap_interval(
        clean_a,
        clean_b,
        value,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        random_state=random_state,
    )
    return EffectSize(
        name="rank-biserial correlation",
        estimate=value(clean_a, clean_b),
        null_value=0.0,
        confidence_interval=interval,
        confidence_level=confidence_level,
        interval_method="bootstrap percentile, resampled within groups",
        n_a=int(clean_a.size),
        n_b=int(clean_b.size),
        notes=("A monotone rescaling of the probability of superiority; the two agree exactly.",),
    )


# --------------------------------------------------------------------------- #
# Categorical outcomes
# --------------------------------------------------------------------------- #


def odds_ratio(
    table: npt.ArrayLike | pd.DataFrame,
    *,
    kind: OddsRatioKind = "conditional",
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> EffectSize:
    """Odds ratio for a 2x2 table, with SciPy's interval.

    The table is read as ``[[a, b], [c, d]]`` and the ratio is the odds of the
    first column in row 1 against the odds of the first column in row 2.

    Args:
        kind: ``"conditional"`` uses the conditional maximum-likelihood estimate
            and an exact interval, matching Fisher's test. ``"sample"`` uses the
            plain ``ad / bc`` with an interval based on the log odds ratio.

    An odds ratio is not a risk ratio. When the outcome is common the two diverge
    sharply, and an odds ratio of 2 can correspond to a much smaller change in
    probability -- which is why :func:`risk_difference` belongs next to it.
    """
    counts = _two_by_two(table)
    result = scipy_odds_ratio(counts, kind=kind)
    interval = result.confidence_interval(confidence_level=confidence_level)

    notes = [
        "An odds ratio is not a risk ratio and not a difference in probability; report a risk "
        "difference alongside it when the outcome is common.",
    ]
    if (counts == 0).any():
        notes.append("The table has an empty cell, so the estimate is 0 or infinite.")
    return EffectSize(
        name=f"odds ratio ({kind})",
        estimate=float(result.statistic),
        null_value=1.0,
        confidence_interval=(float(interval.low), float(interval.high)),
        confidence_level=confidence_level,
        interval_method="exact conditional" if kind == "conditional" else "log odds ratio",
        n_total=int(counts.sum()),
        notes=tuple(notes),
    )


def risk_difference(
    table: npt.ArrayLike | pd.DataFrame,
    *,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    method: str = "newcomb",
) -> EffectSize:
    """Difference in the probability of the first-column outcome between two rows.

    The table is read as ``[[a, b], [c, d]]``; the estimate is
    ``a / (a + b) - c / (c + d)``, i.e. row 1's rate minus row 2's, in the units
    of the outcome itself. Usually the most directly interpretable measure for a
    2x2 table: "three goals per hundred shots" is a sentence, "an odds ratio of
    1.3" is a calculation.

    The interval comes from statsmodels; the default Newcombe hybrid-score method
    behaves well with small counts and rates near 0 or 1, where a Wald interval
    can run outside [-1, 1].
    """
    counts = _two_by_two(table)
    (successes_a, failures_a), (successes_b, failures_b) = counts
    n_a = int(successes_a + failures_a)
    n_b = int(successes_b + failures_b)
    if n_a == 0 or n_b == 0:
        raise EffectSizeError("Both rows of the table need at least one observation.")

    from statsmodels.stats.proportion import confint_proportions_2indep

    rate_a = successes_a / n_a
    rate_b = successes_b / n_b
    low, high = confint_proportions_2indep(
        int(successes_a),
        n_a,
        int(successes_b),
        n_b,
        compare="diff",
        method=method,
        alpha=1.0 - confidence_level,
    )
    return EffectSize(
        name="risk difference",
        estimate=float(rate_a - rate_b),
        null_value=0.0,
        confidence_interval=(float(low), float(high)),
        confidence_level=confidence_level,
        interval_method=f"statsmodels {method}",
        units="proportion",
        n_a=n_a,
        n_b=n_b,
        notes=(
            f"Row rates are {rate_a:.4f} and {rate_b:.4f}.",
            "In the units of the outcome, so it answers 'how many more per hundred?' directly.",
        ),
    )


def _cramers_v_value(counts: npt.NDArray[np.int64], *, bias_correction: bool) -> float:
    # The uncorrected chi-square is the right input: Yates' continuity correction
    # is designed for a hypothesis test, not for a measure of association.
    chi2 = float(stats.chi2_contingency(counts, correction=False).statistic)
    n = int(counts.sum())
    rows, columns = counts.shape
    phi2 = chi2 / n
    if not bias_correction:
        denominator = min(rows - 1, columns - 1)
        return float(np.sqrt(phi2 / denominator)) if denominator > 0 else float("nan")

    # Bergsma (2013): removes the upward bias that is severe in small tables.
    phi2_corrected = max(0.0, phi2 - (rows - 1) * (columns - 1) / (n - 1))
    rows_corrected = rows - (rows - 1) ** 2 / (n - 1)
    columns_corrected = columns - (columns - 1) ** 2 / (n - 1)
    denominator = min(rows_corrected - 1, columns_corrected - 1)
    return float(np.sqrt(phi2_corrected / denominator)) if denominator > 0 else float("nan")


def cramers_v(table: npt.ArrayLike | pd.DataFrame, *, bias_correction: bool = False) -> EffectSize:
    """Cramer's V: association between two categorical variables, on [0, 1].

    ``V = sqrt(chi2 / (n * min(rows - 1, columns - 1)))``. Unlike the chi-square
    statistic it does not grow with the sample size, which is exactly why a
    contingency test needs it: chi2 answers "is there an association?" and V
    answers "how strong?".

    Args:
        bias_correction: Apply Bergsma's correction, which removes an upward bias
            that is severe in small tables. Off by default so the reported value
            matches the familiar textbook formula.

    No interval is computed from a table alone; use :func:`cramers_v_from_labels`
    with the underlying observations to obtain a bootstrap interval.
    """
    counts = _counts(table)
    return EffectSize(
        name="Cramer's V" + (" (bias-corrected)" if bias_correction else ""),
        estimate=_cramers_v_value(counts, bias_correction=bias_correction),
        null_value=0.0,
        n_total=int(counts.sum()),
        notes=(
            "Bounded in [0, 1] and, unlike chi-square, does not grow with the sample size.",
            "No confidence interval from a table alone: pass the underlying rows to "
            "cramers_v_from_labels to bootstrap one.",
        ),
    )


def cramers_v_from_labels(
    rows: npt.ArrayLike,
    columns: npt.ArrayLike,
    *,
    bias_correction: bool = False,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    n_resamples: int = DEFAULT_RESAMPLES,
    random_state: int | np.random.Generator | None = None,
) -> EffectSize:
    """Cramer's V from two aligned label arrays, with a bootstrap interval.

    Resamples the observations, rebuilding the contingency table each time, so the
    interval reflects the uncertainty in the whole table rather than in one cell.
    """
    row_labels = np.asarray(rows)
    column_labels = np.asarray(columns)
    if row_labels.shape != column_labels.shape:
        raise EffectSizeError(
            f"rows and columns must be the same length, got {row_labels.size} and "
            f"{column_labels.size}."
        )

    frame = pd.DataFrame({"row": row_labels, "column": column_labels})
    observed = _cramers_v_value(
        _counts(pd.crosstab(row_labels, column_labels)), bias_correction=bias_correction
    )

    def evaluate(sample: pd.DataFrame) -> float:
        # Cross-tabulate the arrays, not the Series: a resampled frame carries
        # duplicate index labels, which pandas cannot align.
        table = pd.crosstab(sample["row"].to_numpy(), sample["column"].to_numpy())
        if table.shape[0] < 2 or table.shape[1] < 2:
            raise EffectSizeError("A resample lost a whole row or column of the table.")
        return _cramers_v_value(_counts(table), bias_correction=bias_correction)

    result = bootstrap(
        frame,
        evaluate,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        random_state=random_state,
    )
    return EffectSize(
        name="Cramer's V" + (" (bias-corrected)" if bias_correction else ""),
        estimate=observed,
        null_value=0.0,
        confidence_interval=result.confidence_interval,
        confidence_level=confidence_level,
        interval_method="bootstrap percentile over observations",
        n_total=len(frame),
        notes=(
            "Bounded in [0, 1] and, unlike chi-square, does not grow with the sample size.",
            "V cannot go below 0, so its interval is asymmetric and never contains negative "
            "values; a lower bound near 0 is the signal that no association is established.",
        ),
    )
