"""Descriptive diagnostics for numeric samples.

This module describes data. It does **not** choose statistical tests, and it
deliberately offers no function that would let it: there is no ``is_normal``
flag, no ``recommended_test`` field, and no boolean anywhere that a caller could
branch on to pick between a t-test and a rank-based alternative.

Why that restriction exists
---------------------------

The pattern

.. code-block:: python

    if normality_p > 0.05:  # do not do this
        use_t_test()
    else:
        use_mann_whitney()

is wrong for several independent reasons, and each of them survives even when
the code is written carefully:

1. **A normality test answers a question nobody asked.** Its null is "the
   observations were drawn from a normal distribution". Almost no real quantity
   is exactly normal, so with enough data the null is false by construction and
   rejecting it carries no information about whether a mean-based method is
   appropriate.

2. **Power scales with n, so the verdict tracks sample size, not shape.** A
   Student-t sample with 30 degrees of freedom is normal enough for any
   practical purpose; D'Agostino's test returns p = 0.32 for 200 of them and
   p = 6e-10 for 20,000 of them. The distribution never changed. Its excess
   kurtosis is 0.23 in both cases, which is why the *magnitude* measures below
   (skewness, excess kurtosis, quantiles) are the informative ones and the
   p-value is not.

3. **Normality of the observations is usually not the assumption in question.**
   t-based inference concerns the sampling distribution of a mean, which the
   central limit theorem makes approximately normal for large n even when the
   observations are skewed. Linear models assume approximately normal
   *residuals*, not normal predictors or outcomes. Testing the raw column tests
   none of these.

4. **The two tests answer different questions about different estimands.** A
   Welch t-test compares means; Mann-Whitney concerns stochastic ordering. Which
   one is right depends on the research question, the pairing and clustering
   structure, and what quantity is meant to be estimated -- never on a
   normality p-value.

5. **Conditioning the test on the data invalidates its error rates.** Choosing a
   procedure after looking at a preliminary test inflates the type I error rate
   of whatever runs second.

Normality statistics are still exposed here, because looking at them is useful
when they are read as descriptions rather than decisions. They come with the
sample size attached and with explicit notes about how to read them.
"""

import argparse
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

DEFAULT_QUANTILES: Final = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)

#: Tukey's fence multiplier for the inter-quartile outlier indicator.
TUKEY_MULTIPLIER: Final = 1.5

#: Threshold for the Iglewicz-Hoaglin modified z-score.
MODIFIED_Z_THRESHOLD: Final = 3.5

#: 0.75 quantile of the standard normal: makes the MAD comparable to a standard
#: deviation for normally distributed data.
MAD_SCALE: Final = 0.6744897501960817

#: Above this, scipy itself warns that the Shapiro-Wilk p-value is unreliable.
SHAPIRO_MAX_N: Final = 5000
SHAPIRO_MIN_N: Final = 3

#: D'Agostino-Pearson is asymptotic; scipy returns NaN below 8 and the test is
#: not worth reading below roughly 20.
DAGOSTINO_MIN_N: Final = 20

#: Beyond this, a normality test starts rejecting practically irrelevant
#: deviations, so its p-value should not be read as evidence about method choice.
LARGE_SAMPLE_N: Final = 300


def to_float_sample(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Convert input to a 1-D float array, mapping pandas missing values to NaN."""
    if isinstance(values, pd.Series | pd.Index):
        array = values.to_numpy(dtype="float64", na_value=np.nan)
    else:
        array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"Expected a one-dimensional sample, got shape {array.shape}.")
    return array


@dataclass(frozen=True, slots=True)
class SampleDiagnostics:
    """Descriptive summary of one numeric sample.

    ``variance`` and ``std`` use ``ddof=1`` (sample estimators). ``skewness`` and
    ``excess_kurtosis`` are bias-corrected; excess kurtosis is 0 for a normal
    distribution.

    ``standard_error`` describes the *sampling distribution of the mean*, not the
    spread of the observations. The distinction matters: a strongly skewed sample
    can still have a near-normal sampling distribution for its mean.
    """

    name: str
    n_total: int
    n_observed: int
    n_missing: int
    n_infinite: int
    missing_rate: float
    mean: float
    median: float
    variance: float
    std: float
    standard_error: float
    minimum: float
    maximum: float
    q1: float
    q3: float
    iqr: float
    mad: float
    skewness: float
    excess_kurtosis: float
    tukey_lower_fence: float
    tukey_upper_fence: float
    tukey_outliers: int
    modified_z_outliers: int
    quantiles: dict[float, float] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"{self.name}",
            f"  n (total / observed / missing)  {self.n_total:,} / "
            f"{self.n_observed:,} / {self.n_missing:,} ({self.missing_rate:.2%})",
            f"  mean +- sd                      {self.mean:.4f} +- {self.std:.4f}",
            f"  standard error of the mean      {self.standard_error:.4f}",
            f"  median [Q1, Q3], IQR            {self.median:.4f} "
            f"[{self.q1:.4f}, {self.q3:.4f}], {self.iqr:.4f}",
            f"  min / max                       {self.minimum:.4f} / {self.maximum:.4f}",
            f"  variance                        {self.variance:.4f}",
            f"  MAD                             {self.mad:.4f}",
            f"  skewness                        {self.skewness:+.4f}",
            f"  excess kurtosis                 {self.excess_kurtosis:+.4f}",
            f"  Tukey fences                    "
            f"[{self.tukey_lower_fence:.3f}, {self.tukey_upper_fence:.3f}] "
            f"-> {self.tukey_outliers:,} outside "
            f"({self.tukey_outliers / max(self.n_observed, 1):.2%})",
            f"  modified z > {MODIFIED_Z_THRESHOLD}              -> "
            f"{self.modified_z_outliers:,} outside",
        ]
        if self.n_infinite:
            lines.append(f"  infinite values excluded        {self.n_infinite:,}")
        if self.quantiles:
            rendered = "  ".join(f"p{q * 100:g}={v:.3f}" for q, v in sorted(self.quantiles.items()))
            lines.append(f"  quantiles                       {rendered}")
        return "\n".join(lines)


def _shape_measures(observed: npt.NDArray[np.float64]) -> tuple[float, float]:
    """Bias-corrected skewness and excess kurtosis, NaN where they are undefined.

    Both are standardised moments, so a sample with no spread divides by zero.
    scipy would return a value with a precision-loss warning; NaN is the honest
    answer.
    """
    if observed.size < 3 or float(np.ptp(observed)) == 0.0:
        return float("nan"), float("nan")
    return (
        float(stats.skew(observed, bias=False)),
        float(stats.kurtosis(observed, fisher=True, bias=False)),
    )


def modified_z_scores(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Iglewicz-Hoaglin modified z-scores, based on the median and the MAD.

    Robust to the outliers it is meant to detect, unlike a mean/standard-deviation
    z-score which those same outliers inflate. Returns all zeros when the MAD is
    zero (more than half the sample identical), since no value is then unusual by
    this criterion.
    """
    array = to_float_sample(values)
    observed = array[np.isfinite(array)]
    if observed.size == 0:
        return np.full(array.shape, np.nan)

    median = float(np.median(observed))
    mad = float(np.median(np.abs(observed - median)))
    if mad == 0.0:
        return np.zeros_like(array)
    return MAD_SCALE * (array - median) / mad


def describe_sample(
    values: npt.ArrayLike,
    *,
    name: str = "sample",
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> SampleDiagnostics:
    """Summarise a numeric sample.

    NaN is treated as missing and infinite values as malformed; both are excluded
    from the statistics and reported in their own counts rather than silently
    dropped. Statistics that are undefined for the available sample size come
    back as NaN (variance needs two observations, skewness three).

    Raises:
        ValueError: if the input is empty or not one-dimensional.
    """
    array = to_float_sample(values)
    n_total = int(array.size)
    if n_total == 0:
        raise ValueError("describe_sample() requires at least one value, got an empty sample.")

    n_missing = int(np.isnan(array).sum())
    n_infinite = int(np.isinf(array).sum())
    observed = array[np.isfinite(array)]
    n_observed = int(observed.size)

    if n_observed == 0:
        nan = float("nan")
        return SampleDiagnostics(
            name=name,
            n_total=n_total,
            n_observed=0,
            n_missing=n_missing,
            n_infinite=n_infinite,
            missing_rate=n_missing / n_total,
            mean=nan,
            median=nan,
            variance=nan,
            std=nan,
            standard_error=nan,
            minimum=nan,
            maximum=nan,
            q1=nan,
            q3=nan,
            iqr=nan,
            mad=nan,
            skewness=nan,
            excess_kurtosis=nan,
            tukey_lower_fence=nan,
            tukey_upper_fence=nan,
            tukey_outliers=0,
            modified_z_outliers=0,
            quantiles={float(q): nan for q in quantiles},
        )

    variance = float(np.var(observed, ddof=1)) if n_observed > 1 else float("nan")
    std = math.sqrt(variance) if n_observed > 1 else float("nan")
    q1, q3 = (float(value) for value in np.percentile(observed, [25.0, 75.0]))
    iqr = q3 - q1
    median = float(np.median(observed))
    mad = float(np.median(np.abs(observed - median)))

    skewness, excess_kurtosis = _shape_measures(observed)

    lower_fence = q1 - TUKEY_MULTIPLIER * iqr
    upper_fence = q3 + TUKEY_MULTIPLIER * iqr
    tukey_outliers = int(((observed < lower_fence) | (observed > upper_fence)).sum())
    robust_scores = modified_z_scores(observed)
    modified_outliers = int((np.abs(robust_scores) > MODIFIED_Z_THRESHOLD).sum())

    return SampleDiagnostics(
        name=name,
        n_total=n_total,
        n_observed=n_observed,
        n_missing=n_missing,
        n_infinite=n_infinite,
        missing_rate=n_missing / n_total,
        mean=float(np.mean(observed)),
        median=median,
        variance=variance,
        std=std,
        standard_error=std / math.sqrt(n_observed) if n_observed > 1 else float("nan"),
        minimum=float(np.min(observed)),
        maximum=float(np.max(observed)),
        q1=q1,
        q3=q3,
        iqr=iqr,
        mad=mad,
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
        tukey_lower_fence=lower_fence,
        tukey_upper_fence=upper_fence,
        tukey_outliers=tukey_outliers,
        modified_z_outliers=modified_outliers,
        quantiles={float(q): float(np.percentile(observed, q * 100.0)) for q in quantiles},
    )


@dataclass(frozen=True, slots=True)
class NormalityDiagnostics:
    """Normality *descriptions* for one sample.

    There is deliberately no boolean verdict. Read ``skewness`` and
    ``excess_kurtosis`` for the shape, the p-values only alongside
    ``n_observed``, and ``notes`` for what the numbers cannot tell you.

    Test statistics are ``None`` when the sample size makes them unreliable
    rather than being reported with a caveat nobody reads.
    """

    name: str
    n_observed: int
    skewness: float
    excess_kurtosis: float
    shapiro_statistic: float | None
    shapiro_p_value: float | None
    dagostino_statistic: float | None
    dagostino_p_value: float | None
    notes: tuple[str, ...] = ()

    def __str__(self) -> str:
        def render(statistic: float | None, p_value: float | None) -> str:
            if statistic is None or p_value is None:
                return "not reported at this sample size"
            return f"statistic={statistic:.5f}  p={p_value:.4g}"

        shapiro = render(self.shapiro_statistic, self.shapiro_p_value)
        dagostino = render(self.dagostino_statistic, self.dagostino_p_value)
        lines = [
            f"{self.name} (n = {self.n_observed:,})",
            f"  skewness                 {self.skewness:+.4f}   (0 if symmetric)",
            f"  excess kurtosis          {self.excess_kurtosis:+.4f}   (0 if normal)",
            f"  Shapiro-Wilk             {shapiro}",
            f"  D'Agostino-Pearson       {dagostino}",
        ]
        lines.extend(f"  note: {note}" for note in self.notes)
        return "\n".join(lines)


def assess_normality(values: npt.ArrayLike, *, name: str = "sample") -> NormalityDiagnostics:
    """Describe how far a sample departs from normality, without judging it.

    Reports shape measures that do not depend on sample size, plus Shapiro-Wilk
    and D'Agostino-Pearson where their sample-size requirements are met, plus
    notes on how to read them.

    The result intentionally cannot be used as a switch: see the module docstring
    for why selecting a hypothesis test from a normality p-value is invalid.
    """
    array = to_float_sample(values)
    observed = array[np.isfinite(array)]
    n_observed = int(observed.size)
    notes: list[str] = []

    skewness, excess_kurtosis = _shape_measures(observed)
    if math.isnan(skewness):
        notes.append(
            "Shape measures are undefined for this sample (fewer than 3 observations, "
            "or no spread at all)."
        )

    shapiro_statistic: float | None = None
    shapiro_p_value: float | None = None
    if SHAPIRO_MIN_N <= n_observed <= SHAPIRO_MAX_N:
        result = stats.shapiro(observed)
        shapiro_statistic = float(result.statistic)
        shapiro_p_value = float(result.pvalue)
    elif n_observed > SHAPIRO_MAX_N:
        notes.append(
            f"Shapiro-Wilk is not reported above n = {SHAPIRO_MAX_N:,}, where its p-value "
            "is unreliable; use the shape measures and a QQ plot instead."
        )
    else:
        notes.append(f"Shapiro-Wilk needs at least {SHAPIRO_MIN_N} observations.")

    dagostino_statistic: float | None = None
    dagostino_p_value: float | None = None
    if n_observed >= DAGOSTINO_MIN_N:
        omnibus = stats.normaltest(observed)
        dagostino_statistic = float(omnibus.statistic)
        dagostino_p_value = float(omnibus.pvalue)
    else:
        notes.append(
            f"D'Agostino-Pearson needs about {DAGOSTINO_MIN_N} observations to be meaningful."
        )

    if n_observed >= LARGE_SAMPLE_N:
        notes.append(
            f"At n = {n_observed:,} a normality test detects departures far too small to "
            "affect inference; treat a small p-value as a statement about power, not "
            "about whether a mean-based method is usable."
        )
    if 0 < n_observed < DAGOSTINO_MIN_N:
        notes.append(
            "At this sample size a normality test has little power, so a large p-value is "
            "not evidence of normality."
        )
    notes.append(
        "These describe the observations. They do not describe the sampling distribution "
        "of a statistic, nor model residuals, and they do not determine which hypothesis "
        "test is appropriate."
    )

    return NormalityDiagnostics(
        name=name,
        n_observed=n_observed,
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
        shapiro_statistic=shapiro_statistic,
        shapiro_p_value=shapiro_p_value,
        dagostino_statistic=dagostino_statistic,
        dagostino_p_value=dagostino_p_value,
        notes=tuple(notes),
    )


def describe_by_group(
    frame: pd.DataFrame,
    value_column: str,
    group_column: str,
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> pd.DataFrame:
    """Summarise ``value_column`` separately for each level of ``group_column``.

    Groups are described independently; nothing here compares them. Note that a
    group column with repeated units (the same player or match appearing many
    times) makes the rows non-independent, which matters for any later inference
    but not for these descriptions.
    """
    for column in (value_column, group_column):
        if column not in frame.columns:
            raise KeyError(f"{column!r} is not a column of the frame.")

    rows: list[dict[str, float | int | str]] = []
    for group, values in frame.groupby(group_column, observed=True)[value_column]:
        summary = describe_sample(values, name=str(group), quantiles=quantiles)
        rows.append(
            {
                group_column: str(group),
                "n": summary.n_observed,
                "missing": summary.n_missing,
                "mean": summary.mean,
                "std": summary.std,
                "standard_error": summary.standard_error,
                "median": summary.median,
                "q1": summary.q1,
                "q3": summary.q3,
                "iqr": summary.iqr,
                "skewness": summary.skewness,
                "excess_kurtosis": summary.excess_kurtosis,
                "tukey_outliers": summary.tukey_outliers,
            }
        )
    return pd.DataFrame(rows).sort_values("n", ascending=False, ignore_index=True)


# --------------------------------------------------------------------------- #
# Demonstration on the real shot dataset
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    from football_intelligence.features import shots as shot_features

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        prog="python -m football_intelligence.statistics.diagnostics",
        description="Run sample diagnostics over the canonical shot dataset.",
    )
    parser.add_argument("--dataset", type=Path, default=shot_features.DEFAULT_DATASET_PATH)
    arguments = parser.parse_args(argv)

    try:
        dataset = shot_features.read_shot_dataset(arguments.dataset)
    except shot_features.ShotFeatureError as error:
        logger.error("%s", error)
        return 1

    # The stored angle is in radians; degrees are easier to read.
    angle_degrees = np.degrees(dataset["shot_angle"])
    pd.set_option("display.width", 150)

    print("=" * 78)
    print("SAMPLE DIAGNOSTICS")
    print("=" * 78)
    print(describe_sample(dataset["shot_distance"], name="shot_distance (yards)"))
    print()
    print(describe_sample(angle_degrees, name="shot_angle (degrees)"))

    print()
    print("=" * 78)
    print("NORMALITY DESCRIPTIONS (not a test-selection rule)")
    print("=" * 78)
    print(assess_normality(dataset["shot_distance"], name="shot_distance (yards)"))
    print()
    print(assess_normality(angle_degrees, name="shot_angle (degrees)"))

    print()
    print("=" * 78)
    print("SHOT DISTANCE BY BODY PART")
    print("=" * 78)
    print(describe_by_group(dataset, "shot_distance", "body_part").round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
