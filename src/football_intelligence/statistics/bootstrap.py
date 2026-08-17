"""Bootstrap uncertainty estimation, with an explicit resampling unit.

Why the bootstrap works
-----------------------

A confidence interval needs the sampling distribution of a statistic: how much
it would move if the study were repeated. We cannot repeat the study, but we do
have one sample from the population, and the empirical distribution of that
sample is the best available estimate of the population it came from. So we
treat the sample as a stand-in population and draw new samples of the same size
*from it*, with replacement. The spread of the statistic across those resamples
estimates the spread it would have across real repetitions.

Nothing about this requires a formula for the statistic's variance, which is the
point: the same machinery works for a median, a conversion rate, a ratio of two
sums, or an entire model's evaluation metric, including quantities whose
analytical standard error is unknown or intractable.

What is being resampled
-----------------------

Whatever unit you say is exchangeable. That choice is the whole design, and
getting it wrong produces intervals that are confidently too narrow.

- ``clusters=None`` resamples individual rows. This assumes rows are independent
  draws -- that knowing one shot tells you nothing about the next.
- ``clusters=<labels>`` resamples whole clusters, keeping every row of each
  drawn cluster together. This assumes *clusters* are independent while allowing
  arbitrary dependence inside them.

Why it matters for football
---------------------------

Shots are not independent draws. Shots from one match share the two teams, the
game state, the referee and the pitch; shots from one player share that player's
role and shooting habits. Resampling shots individually assumes 2,918
independent observations exist, which is a claim about the data, not a neutral
default.

The variance of a mean is inflated by roughly ``1 + (m - 1) * rho`` for clusters
of size ``m`` with intra-class correlation ``rho``. The cluster bootstrap
estimates that inflation from the data instead of assuming a value for ``rho``,
and the size of the effect is an empirical question rather than a foregone
conclusion. On this dataset:

- resampling **matches** widens the standard error of mean shot distance by only
  about 4%, because shot distance varies far more within a match -- across both
  teams and every situation -- than it does between matches;
- resampling **players** widens it by about 60%, because a player's shots really
  do resemble each other. A centre-back heads from eight yards and a midfielder
  shoots from twenty-five, match after match.

So the choice of cluster is a claim about *which* dependence matters for *this*
statistic. Picking the wrong one can leave an interval far too confident, and
picking a cluster with no real dependence inside it costs almost nothing.

Bootstrap versus analytical intervals
-------------------------------------

An analytical interval such as ``mean +- 1.96 * s / sqrt(n)`` is faster, exact
under its assumptions, and needs no random numbers. Prefer it when a correct
formula exists and its assumptions hold. Prefer the bootstrap when:

- the statistic has no convenient closed-form standard error (a median, a ratio,
  goals minus expected goals, an AUC);
- the sampling distribution is skewed, so a symmetric ``+-`` interval is wrong;
- the dependence structure is awkward, since resampling clusters handles it
  without deriving a variance estimator.

The bootstrap is not magic. It inherits any bias in the sample, needs enough
data for the empirical distribution to resemble the population, and struggles
with statistics that depend on extreme order statistics such as the maximum.

Only percentile intervals are implemented here. BCa additionally corrects for
bias and skewness via a jackknife acceleration estimate; it is deliberately left
out until a concrete case needs it, and the percentile interval's assumptions are
stated on every result.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

logger = logging.getLogger(__name__)

#: Data the bootstrap can resample: rows are observations.
BootstrapData = npt.NDArray[Any] | pd.DataFrame

#: A statistic maps a resampled dataset to a single number. The argument stays
#: ``Any`` so that a narrower callable such as ``(NDArray) -> float`` is still
#: accepted; callable parameters are contravariant.
Statistic = Callable[[Any], float]

DEFAULT_RESAMPLES: Final = 10_000
DEFAULT_CONFIDENCE_LEVEL: Final = 0.95

#: Resamples are generated in blocks so the random draws are vectorised without
#: materialising an ``n_resamples x n_rows`` index matrix.
_CHUNK_SIZE: Final = 512

#: Below this many resamples a percentile interval is visibly grainy.
MIN_RECOMMENDED_RESAMPLES: Final = 1_000

#: Below this many independent units the bootstrap has too few distinct
#: resamples to describe a tail.
MIN_RECOMMENDED_UNITS: Final = 20


class BootstrapError(ValueError):
    """Raised when a bootstrap cannot be run on the inputs as supplied."""


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Outcome of one bootstrap.

    ``standard_error`` is the standard deviation of the statistic across
    resamples, which estimates the standard error of the statistic itself.
    ``bias`` is the difference between the mean of the resampled statistics and
    the value computed on the original data; a large bias relative to the
    standard error is a sign that the percentile interval may be misleading.
    """

    statistic_name: str
    observed: float
    bootstrap_mean: float
    bias: float
    standard_error: float
    confidence_interval: tuple[float, float]
    confidence_level: float
    n_resamples: int
    n_observations: int
    resampling_unit: str
    n_clusters: int | None = None
    n_failed_resamples: int = 0
    warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    replicates: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty(0), repr=False)

    @property
    def interval_width(self) -> float:
        low, high = self.confidence_interval
        return high - low

    def __str__(self) -> str:
        low, high = self.confidence_interval
        unit = (
            f"{self.n_clusters:,} {self.resampling_unit} clusters"
            if self.n_clusters is not None
            else f"{self.n_observations:,} individual observations"
        )
        lines = [
            f"{self.statistic_name}",
            f"  observed          {self.observed:.6g}",
            f"  standard error    {self.standard_error:.6g}",
            f"  {self.confidence_level:.0%} percentile CI  "
            f"[{low:.6g}, {high:.6g}]   width {self.interval_width:.6g}",
            f"  resampled         {unit} ({self.n_observations:,} rows)",
            f"  resamples         {self.n_resamples:,}",
            f"  bias              {self.bias:+.6g}",
        ]
        if self.n_failed_resamples:
            lines.append(f"  failed resamples  {self.n_failed_resamples:,}")
        lines.extend(f"  WARNING           {item}" for item in self.warnings)
        lines.extend(f"  note              {item}" for item in self.notes)
        return "\n".join(lines)


def _take(data: BootstrapData, indices: npt.NDArray[np.int64]) -> BootstrapData:
    """Select rows by position, for a DataFrame or an array."""
    if isinstance(data, pd.DataFrame):
        return data.take(indices)
    return data[indices]


def _cluster_positions(
    clusters: npt.ArrayLike, n_rows: int
) -> tuple[list[npt.NDArray[np.int64]], npt.NDArray[Any]]:
    """Group row positions by cluster label, preserving first-appearance order."""
    labels = np.asarray(clusters)
    if labels.ndim != 1:
        raise BootstrapError(f"Cluster labels must be one-dimensional, got shape {labels.shape}.")
    if labels.size != n_rows:
        raise BootstrapError(
            f"Got {labels.size} cluster labels for {n_rows} rows. Every row must be assigned "
            "to exactly one cluster."
        )
    if pd.isna(labels).any():
        raise BootstrapError(
            "Cluster labels contain missing values. A row with no cluster cannot be resampled: "
            "decide explicitly whether it forms its own cluster or should be dropped."
        )

    unique, inverse = np.unique(labels, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    boundaries = np.searchsorted(inverse[order], np.arange(unique.size + 1))
    positions = [
        order[boundaries[index] : boundaries[index + 1]].astype(np.int64)
        for index in range(unique.size)
    ]
    return positions, unique


def bootstrap(
    data: BootstrapData,
    statistic: Statistic,
    *,
    clusters: npt.ArrayLike | None = None,
    strata: npt.ArrayLike | None = None,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    random_state: int | np.random.Generator | None = None,
    statistic_name: str = "statistic",
    cluster_name: str = "cluster",
) -> BootstrapResult:
    """Estimate the uncertainty of ``statistic`` by resampling ``data``.

    Args:
        data: A 1-D or 2-D array, or a DataFrame. Rows are the observations.
        statistic: Callable mapping a resampled ``data`` to one number. It is
            called once on the original data and once per resample.
        clusters: Optional label per row. When given, **whole clusters** are
            resampled with replacement rather than individual rows, which is the
            correct unit whenever rows within a cluster are dependent. The number
            of clusters drawn equals the number observed, so the resampled data
            varies in length.
        strata: Optional label per row. When given, rows are resampled **within**
            each stratum, so every stratum keeps its original size. This is the
            correct scheme for a two-sample statistic such as Cohen's d, where the
            group sizes are fixed by the design and should not vary across
            resamples. Mutually exclusive with ``clusters``.
        n_resamples: Number of resamples.
        confidence_level: Coverage of the percentile interval.
        random_state: Seed or ``Generator``. Passing an integer makes the result
            exactly reproducible.
        statistic_name: Label used in the result and its rendering.
        cluster_name: What one cluster is, for readable output ("match", "player").

    Returns:
        A :class:`BootstrapResult` with the observed value, the standard error,
        a percentile interval and the full replicate distribution.

    Raises:
        BootstrapError: for empty data, invalid arguments, a statistic that does
            not return a finite number on the original data, or malformed
            cluster labels.
    """
    if isinstance(data, pd.DataFrame):
        n_rows = len(data)
    else:
        data = np.asarray(data)
        if data.ndim == 0:
            raise BootstrapError("Cannot bootstrap a scalar; supply a sample of observations.")
        n_rows = int(data.shape[0])

    if n_rows == 0:
        raise BootstrapError("Cannot bootstrap an empty sample.")
    if n_resamples < 1:
        raise BootstrapError(f"n_resamples must be at least 1, got {n_resamples}.")
    if not 0.0 < confidence_level < 1.0:
        raise BootstrapError(
            f"confidence_level must lie strictly between 0 and 1, got {confidence_level}."
        )

    observed = _evaluate(statistic, data, context="the original data")
    generator = np.random.default_rng(random_state)

    if clusters is not None and strata is not None:
        raise BootstrapError(
            "Pass either clusters or strata, not both: one resamples whole groups and the "
            "other resamples within them, and combining them needs a design decision this "
            "function cannot make for you."
        )

    warnings: list[str] = []
    strata_positions: list[npt.NDArray[np.int64]] | None = None
    if strata is not None:
        strata_positions, _ = _cluster_positions(strata, n_rows)
        positions: list[npt.NDArray[np.int64]] | None = None
        n_units = n_rows
        unit = "observation within stratum"
    elif clusters is None:
        positions = None
        n_units = n_rows
        unit = "observation"
    else:
        positions, unique = _cluster_positions(clusters, n_rows)
        n_units = len(positions)
        unit = cluster_name
        if n_units == n_rows:
            warnings.append(
                "Every cluster contains exactly one row, so this is identical to resampling "
                "observations. Check that the cluster labels are the ones you intended."
            )
        del unique

    if n_resamples < MIN_RECOMMENDED_RESAMPLES:
        warnings.append(
            f"{n_resamples} resamples is few for a percentile interval; the tails are "
            f"estimated from a handful of values. {MIN_RECOMMENDED_RESAMPLES:,} or more is "
            "usual."
        )
    if n_units < MIN_RECOMMENDED_UNITS:
        warnings.append(
            f"Only {n_units} independent units ({unit}). With so few the bootstrap has very "
            "few distinct resamples to work with and the interval is unreliable."
        )

    if strata_positions is not None:
        replicates = _resample_within_strata(
            data, statistic, strata_positions, n_resamples, generator
        )
    else:
        replicates = _resample(data, statistic, positions, n_rows, n_units, n_resamples, generator)

    finite = np.isfinite(replicates)
    n_failed = int((~finite).sum())
    if n_failed == n_resamples:
        raise BootstrapError(
            "Every resample produced a non-finite statistic; the statistic cannot be "
            "bootstrapped on this data."
        )
    if n_failed:
        warnings.append(
            f"{n_failed} of {n_resamples} resamples produced a non-finite value and are "
            "excluded from the interval. This usually means the statistic is undefined for "
            "some resamples, for example a rate with an empty denominator."
        )
    usable = replicates[finite]

    tail = (1.0 - confidence_level) / 2.0
    low, high = (float(value) for value in np.quantile(usable, [tail, 1.0 - tail], method="linear"))
    bootstrap_mean = float(np.mean(usable))
    standard_error = float(np.std(usable, ddof=1)) if usable.size > 1 else float("nan")
    bias = bootstrap_mean - observed

    if standard_error > 0 and abs(bias) > 0.25 * standard_error:
        warnings.append(
            f"The bootstrap distribution is centred {bias:+.4g} away from the observed value, "
            f"which is {abs(bias) / standard_error:.2f} standard errors. A percentile interval "
            "assumes this bias is small; treat the interval with caution."
        )

    return BootstrapResult(
        statistic_name=statistic_name,
        observed=observed,
        bootstrap_mean=bootstrap_mean,
        bias=bias,
        standard_error=standard_error,
        confidence_interval=(low, high),
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        n_observations=n_rows,
        resampling_unit=unit,
        n_clusters=None if positions is None else n_units,
        n_failed_resamples=n_failed,
        warnings=tuple(warnings),
        notes=(
            "Percentile intervals assume the bootstrap distribution is a good stand-in for the "
            "sampling distribution and is not strongly biased or skewed.",
            _resampling_note(positions, strata_positions, unit),
        ),
        replicates=replicates,
    )


def _resampling_note(
    positions: list[npt.NDArray[np.int64]] | None,
    strata_positions: list[npt.NDArray[np.int64]] | None,
    unit: str,
) -> str:
    """Describe, in the result, exactly what was resampled."""
    if strata_positions is not None:
        return (
            f"Rows were resampled within each of the {len(strata_positions)} strata, so every "
            "stratum kept its original size. This is the right scheme when group sizes are "
            "fixed by the design."
        )
    if positions is None:
        return "Rows were resampled individually, which assumes they are independent."
    return f"Whole {unit} clusters were resampled, so any dependence within a {unit} is preserved."


def _evaluate(statistic: Statistic, data: BootstrapData, *, context: str) -> float:
    """Call the statistic and insist it returns a single finite number."""
    try:
        value = statistic(data)
    except Exception as error:
        raise BootstrapError(f"The statistic raised on {context}: {error!r}") from error

    array = np.asarray(value, dtype=np.float64)
    if array.size != 1:
        raise BootstrapError(
            f"The statistic must return a single number; on {context} it returned "
            f"something of size {array.size}."
        )
    result = float(array.reshape(()))
    if context == "the original data" and not np.isfinite(result):
        raise BootstrapError(
            f"The statistic is not finite on {context} ({result}); there is nothing to bootstrap."
        )
    return result


def _resample_within_strata(
    data: BootstrapData,
    statistic: Statistic,
    strata_positions: Sequence[npt.NDArray[np.int64]],
    n_resamples: int,
    generator: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Resample rows inside each stratum, keeping every stratum at its own size."""
    replicates = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        indices = np.concatenate(
            [rows[generator.integers(0, rows.size, size=rows.size)] for rows in strata_positions]
        )
        try:
            replicates[index] = _evaluate(statistic, _take(data, indices), context="a resample")
        except BootstrapError:
            replicates[index] = np.nan
    return replicates


def _resample(
    data: BootstrapData,
    statistic: Statistic,
    positions: list[npt.NDArray[np.int64]] | None,
    n_rows: int,
    n_units: int,
    n_resamples: int,
    generator: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Draw the resamples and evaluate the statistic on each.

    Random draws are vectorised in blocks; the per-resample loop remains because
    the statistic is an arbitrary callable that must see each resampled dataset.
    """
    replicates = np.empty(n_resamples, dtype=np.float64)
    filled = 0

    while filled < n_resamples:
        block = min(_CHUNK_SIZE, n_resamples - filled)
        draws = generator.integers(0, n_units, size=(block, n_units if positions else n_rows))
        for row in draws:
            if positions is None:
                indices = row
            else:
                indices = np.concatenate([positions[choice] for choice in row])
            try:
                replicates[filled] = _evaluate(
                    statistic, _take(data, indices), context="a resample"
                )
            except BootstrapError:
                replicates[filled] = np.nan
            filled += 1

    return replicates


def cluster_sizes(clusters: npt.ArrayLike) -> pd.Series:
    """Number of rows per cluster, largest first. Useful before choosing a unit."""
    return pd.Series(np.asarray(clusters)).value_counts()


def compare_resampling_units(
    data: BootstrapData,
    statistic: Statistic,
    *,
    clusters: npt.ArrayLike,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    random_state: int | np.random.Generator | None = None,
    statistic_name: str = "statistic",
    cluster_name: str = "cluster",
) -> tuple[BootstrapResult, BootstrapResult]:
    """Run the same bootstrap twice, resampling rows and then whole clusters.

    Returns ``(rows, clustered)``. The ratio of their standard errors estimates
    how much the dependence inside clusters inflates the uncertainty; a ratio
    well above 1 means the row-level interval is too narrow to believe.
    """
    rows = bootstrap(
        data,
        statistic,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        random_state=random_state,
        statistic_name=f"{statistic_name} (resampling shots)",
    )
    clustered = bootstrap(
        data,
        statistic,
        clusters=clusters,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        random_state=random_state,
        statistic_name=f"{statistic_name} (resampling {cluster_name} clusters)",
        cluster_name=cluster_name,
    )
    return rows, clustered


# --------------------------------------------------------------------------- #
# Football demonstration
# --------------------------------------------------------------------------- #


def _mean(sample: npt.NDArray[np.float64]) -> float:
    return float(np.mean(sample))


def main(argv: Sequence[str] | None = None) -> int:
    """Bootstrap two football quantities under three different resampling units."""
    import argparse
    from pathlib import Path

    from football_intelligence.features import shots as shot_features

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        prog="python -m football_intelligence.statistics.bootstrap",
        description="Bootstrap football quantities, comparing resampling units.",
    )
    parser.add_argument("--dataset", type=Path, default=shot_features.DEFAULT_DATASET_PATH)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=20260818)
    arguments = parser.parse_args(argv)

    try:
        data = shot_features.read_shot_dataset(arguments.dataset)
    except shot_features.ShotFeatureError as error:
        logger.error("%s", error)
        return 1

    distance = data["shot_distance"].to_numpy()
    goal = data["goal"].to_numpy()
    matches = data["match_id"].to_numpy()
    players = data["player_id"].to_numpy()
    common = {"n_resamples": arguments.resamples, "random_state": arguments.seed}

    print("=" * 78)
    print("1. MEAN SHOT DISTANCE: does the resampling unit change the interval?")
    print("=" * 78)
    print(
        f"{len(data):,} shots, from {len(np.unique(matches))} matches and "
        f"{len(np.unique(players))} players.\n"
    )
    by_shot, by_match = compare_resampling_units(
        distance,
        _mean,
        clusters=matches,
        statistic_name="mean shot distance",
        cluster_name="match",
        **common,
    )
    by_player = bootstrap(
        distance,
        _mean,
        clusters=players,
        statistic_name="mean shot distance (resampling players)",
        cluster_name="player",
        **common,
    )
    for result in (by_shot, by_match, by_player):
        print(result)
        print()

    analytical = float(np.std(distance, ddof=1) / np.sqrt(distance.size))
    print("standard error under each assumption:")
    print(f"  analytical, assuming independence   {analytical:.4f}")
    print(f"  bootstrap, resampling shots         {by_shot.standard_error:.4f}")
    print(
        f"  bootstrap, resampling matches       {by_match.standard_error:.4f}"
        f"   ({by_match.standard_error / by_shot.standard_error:.2f}x)"
    )
    print(
        f"  bootstrap, resampling players       {by_player.standard_error:.4f}"
        f"   ({by_player.standard_error / by_shot.standard_error:.2f}x)"
    )
    print(
        "\nThe shot-level bootstrap reproduces the textbook formula, which is a useful\n"
        "check that it works. Matching on match barely matters: shot distance varies\n"
        "far more within a match than between matches. Matching on player matters a\n"
        "lot, because a player takes similar shots every week. The dependence that\n"
        "counts here is the player, and only asking the data reveals that."
    )

    print()
    print("=" * 78)
    print("2. CONVERSION RATE: the same comparison for a proportion")
    print("=" * 78)
    rate_by_shot, rate_by_match = compare_resampling_units(
        goal,
        _mean,
        clusters=matches,
        statistic_name="conversion rate",
        cluster_name="match",
        **common,
    )
    rate_by_player = bootstrap(
        goal,
        _mean,
        clusters=players,
        statistic_name="conversion rate (resampling players)",
        cluster_name="player",
        **common,
    )
    for result in (rate_by_shot, rate_by_match, rate_by_player):
        print(result)
        print()

    rate = float(np.mean(goal))
    binomial = float(np.sqrt(rate * (1.0 - rate) / goal.size))
    print("standard error under each assumption:")
    print(f"  analytical binomial                 {binomial:.5f}")
    print(f"  bootstrap, resampling shots         {rate_by_shot.standard_error:.5f}")
    print(
        f"  bootstrap, resampling matches       {rate_by_match.standard_error:.5f}"
        f"   ({rate_by_match.standard_error / rate_by_shot.standard_error:.2f}x)"
    )
    print(
        f"  bootstrap, resampling players       {rate_by_player.standard_error:.5f}"
        f"   ({rate_by_player.standard_error / rate_by_shot.standard_error:.2f}x)"
    )
    print(
        "\nConversion is far less clustered than distance. Whether a shot goes in is\n"
        "close to a coin flip given the chance, so a match shares almost nothing\n"
        "(1.03x) and a player only a little (1.17x, the trace of finishing skill and\n"
        "of habitually taking better or worse chances). The binomial formula is nearly\n"
        "adequate here, which is worth knowing -- and is not something to assume in\n"
        "advance, since the same formula on the same rows was 63% too narrow for\n"
        "mean shot distance."
    )

    print()
    print("=" * 78)
    print("3. A STATISTIC WITH NO CONVENIENT FORMULA: goals minus expected goals")
    print("=" * 78)
    frame = data[["goal", "statsbomb_xg"]]
    overperformance = bootstrap(
        frame,
        lambda f: float((f["goal"].sum() - f["statsbomb_xg"].sum()) / len(f) * 100.0),
        clusters=matches,
        statistic_name="goals minus expected goals, per 100 shots",
        cluster_name="match",
        **common,
    )
    print(overperformance)
    print(
        "\nThis is a difference between a count and a sum of model predictions, divided\n"
        "by a sample size that itself varies across resamples. There is no standard\n"
        "error formula to look up, and the bootstrap needs none. The interval spans\n"
        "zero, so these tournaments do not show finishing that differs from the\n"
        "StatsBomb model by more than sampling noise."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
