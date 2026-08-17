"""Multiple-testing corrections: Bonferroni and Benjamini-Hochberg.

The problem
-----------

A p-value threshold of 0.05 means that, when the null is true, one test in twenty
rejects anyway. Run a family of *m* independent tests under the global null and
the probability that at least one of them "finds" something is
``1 - (1 - alpha) ** m``: 23% for 5 tests, **83% for 35**, 99.4% for 100. Testing
every player in a squad and reporting the ones that came out below 0.05 is
therefore a procedure that almost always produces a finding, whether or not
anything is there.

Two different guarantees
------------------------

**Family-wise error rate (FWER)** is the probability of making *at least one*
false rejection anywhere in the family. Controlling it at 0.05 means: if nothing
is real, there is a 95% chance the whole report contains no false claims.
:func:`bonferroni` controls FWER by comparing each p-value to ``alpha / m``,
equivalently by multiplying each p-value by ``m``. It needs no assumption about
how the tests relate to one another, and it is conservative: with 35 tests a
p-value of 0.002 is no longer evidence of anything.

**False discovery rate (FDR)** is the *expected proportion of false positives
among the rejections you make*. Controlling it at 0.05 means: of the findings
reported, on average 5% are expected to be wrong. :func:`benjamini_hochberg`
controls FDR, is far more powerful than Bonferroni when several effects are real,
and assumes the tests are independent or positively dependent.

Which to prefer
---------------

- **Bonferroni** when a single false positive is costly and the family produces
  one decision: a confirmatory analysis, a claim that will be published, a model
  that goes to production on the strength of it.
- **Benjamini-Hochberg** when the output is a *screening list* meant for
  follow-up, and a known fraction of false leads is an acceptable price for not
  missing real effects. "Which players are worth a closer look?" is an FDR
  question, not an FWER one.

Neither is a substitute for having decided the family in advance. Correcting for
the tests you happened to run, after choosing them by looking at the data, does
not restore the guarantee.

What an adjusted p-value is not
-------------------------------

**It is not a measure of effect magnitude.** An adjusted p-value depends on the
size of the family it sits in: the identical player, with the identical shots and
the identical conversion rate, gets a larger adjusted p-value simply because more
players were tested alongside them. Nothing about that player changed. Effect
size, and the confidence interval around it, must be reported separately and are
unaffected by the correction.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

logger = logging.getLogger(__name__)

Method = Literal["bonferroni", "benjamini-hochberg"]

DEFAULT_ALPHA: Final = 0.05

#: Our method names mapped to the ones statsmodels uses internally.
_STATSMODELS_METHOD: Final[dict[str, str]] = {
    "bonferroni": "bonferroni",
    "benjamini-hochberg": "fdr_bh",
}

_CONTROLS: Final[dict[str, str]] = {
    "bonferroni": "family-wise error rate",
    "benjamini-hochberg": "false discovery rate",
}


class MultipleTestingError(ValueError):
    """Raised when a correction cannot be applied to the p-values as supplied."""


@dataclass(frozen=True, slots=True)
class MultipleTestingResult:
    """Outcome of correcting one family of p-values.

    ``adjusted_p_values`` are on the same scale as the originals, so they can be
    compared to ``alpha`` directly. They are **not** effect sizes: they grow with
    the size of the family, and say nothing about how large any single effect is.
    """

    method: Method
    controls: str
    alpha: float
    n_tests: int
    labels: tuple[str, ...]
    p_values: npt.NDArray[np.float64]
    adjusted_p_values: npt.NDArray[np.float64]
    rejected: npt.NDArray[np.bool_]
    notes: tuple[str, ...] = ()

    @property
    def n_rejected(self) -> int:
        """Rejections after correction."""
        return int(np.sum(self.rejected))

    @property
    def n_rejected_uncorrected(self) -> int:
        """Rejections there would have been at the same alpha without correction."""
        return int(np.sum(self.p_values <= self.alpha))

    @property
    def expected_false_positives_uncorrected(self) -> float:
        """How many rejections uncorrected testing yields when nothing is real."""
        return self.n_tests * self.alpha

    @property
    def probability_of_any_false_positive(self) -> float:
        """FWER of the *uncorrected* family under the global null, if independent."""
        return float(1.0 - (1.0 - self.alpha) ** self.n_tests)

    def to_frame(self) -> pd.DataFrame:
        """Tabulate the family, most significant first."""
        frame = pd.DataFrame(
            {
                "p_value": self.p_values,
                f"{self.method}_adjusted": self.adjusted_p_values,
                "rejected": self.rejected,
            },
            index=pd.Index(self.labels, name="test"),
        )
        return frame.sort_values("p_value")

    def __str__(self) -> str:
        lines = [
            f"{self.method} correction (controls the {self.controls})",
            f"  tests in family   {self.n_tests}",
            f"  alpha             {self.alpha}",
            f"  rejected          {self.n_rejected} after correction, "
            f"{self.n_rejected_uncorrected} before",
            f"  under the global null, uncorrected testing would reject "
            f"{self.expected_false_positives_uncorrected:.2f} on average "
            f"({self.probability_of_any_false_positive:.1%} chance of at least one)",
        ]
        lines.extend(f"  note              {item}" for item in self.notes)
        return "\n".join(lines)


def _validate(
    p_values: npt.ArrayLike, alpha: float, labels: Sequence[str] | None
) -> tuple[npt.NDArray[np.float64], tuple[str, ...]]:
    array = np.asarray(p_values, dtype=np.float64)
    if array.ndim != 1:
        raise MultipleTestingError(
            f"Expected a one-dimensional family of p-values, got shape {array.shape}."
        )
    if array.size == 0:
        raise MultipleTestingError("Cannot correct an empty family of p-values.")
    if not np.all(np.isfinite(array)):
        raise MultipleTestingError(
            "p-values must all be finite. A test that failed to produce a p-value has to be "
            "resolved or excluded from the family deliberately, not passed through as NaN."
        )
    if np.any((array < 0.0) | (array > 1.0)):
        raise MultipleTestingError(
            f"p-values must lie in [0, 1]; got a range of [{array.min():.4g}, {array.max():.4g}]."
        )
    if not 0.0 < alpha < 1.0:
        raise MultipleTestingError(f"alpha must lie strictly between 0 and 1, got {alpha}.")

    if labels is None:
        resolved = tuple(f"test {index + 1}" for index in range(array.size))
    else:
        resolved = tuple(str(label) for label in labels)
        if len(resolved) != array.size:
            raise MultipleTestingError(f"Got {len(resolved)} labels for {array.size} p-values.")
    return array, resolved


def adjust(
    p_values: npt.ArrayLike,
    *,
    method: Method = "benjamini-hochberg",
    alpha: float = DEFAULT_ALPHA,
    labels: Sequence[str] | None = None,
) -> MultipleTestingResult:
    """Correct a family of p-values for multiple testing.

    The correction itself is delegated to
    :func:`statsmodels.stats.multitest.multipletests`; this function validates the
    inputs and attaches the interpretation.

    Args:
        p_values: One p-value per test in the family.
        method: ``"bonferroni"`` (controls FWER) or ``"benjamini-hochberg"``
            (controls FDR).
        alpha: The level at which the chosen error rate is controlled.
        labels: Optional name per test, used in :meth:`MultipleTestingResult.to_frame`.

    Raises:
        MultipleTestingError: for an empty family, non-finite or out-of-range
            p-values, an impossible alpha, or a label count that does not match.
    """
    from statsmodels.stats.multitest import multipletests

    if method not in _STATSMODELS_METHOD:
        raise MultipleTestingError(
            f"Unknown method {method!r}; choose from {sorted(_STATSMODELS_METHOD)}."
        )
    array, resolved_labels = _validate(p_values, alpha, labels)

    rejected, adjusted, _, _ = multipletests(array, alpha=alpha, method=_STATSMODELS_METHOD[method])

    notes = [
        "Adjusted p-values are not effect sizes: they grow with the size of the family, "
        "while the underlying effect does not.",
    ]
    if method == "benjamini-hochberg":
        notes.append(
            "Benjamini-Hochberg controls the expected share of false positives among the "
            "rejections, so some of the rejections are expected to be wrong by design."
        )
        notes.append(
            "It assumes the tests are independent or positively dependent; under arbitrary "
            "dependence the Benjamini-Yekutieli variant is the conservative alternative."
        )
    else:
        notes.append(
            "Bonferroni controls the probability of even one false rejection anywhere in the "
            "family, needs no assumption about dependence between tests, and is conservative."
        )
    if array.size == 1:
        notes.append("A family of one test needs no correction; the p-value is unchanged.")

    return MultipleTestingResult(
        method=method,
        controls=_CONTROLS[method],
        alpha=alpha,
        n_tests=int(array.size),
        labels=resolved_labels,
        p_values=array,
        adjusted_p_values=np.asarray(adjusted, dtype=np.float64),
        rejected=np.asarray(rejected, dtype=np.bool_),
        notes=tuple(notes),
    )


def bonferroni(
    p_values: npt.ArrayLike,
    *,
    alpha: float = DEFAULT_ALPHA,
    labels: Sequence[str] | None = None,
) -> MultipleTestingResult:
    """Bonferroni correction, controlling the family-wise error rate.

    Each p-value is multiplied by the number of tests (capped at 1), which is the
    same decision as comparing it to ``alpha / m``. Makes no assumption about
    dependence between the tests, at the cost of power.
    """
    return adjust(p_values, method="bonferroni", alpha=alpha, labels=labels)


def benjamini_hochberg(
    p_values: npt.ArrayLike,
    *,
    alpha: float = DEFAULT_ALPHA,
    labels: Sequence[str] | None = None,
) -> MultipleTestingResult:
    """Benjamini-Hochberg step-up procedure, controlling the false discovery rate.

    Ranks the p-values and compares the k-th smallest to ``k * alpha / m``,
    rejecting everything up to the largest one that passes. Far more powerful than
    Bonferroni when several effects are real; assumes independence or positive
    dependence between tests.
    """
    return adjust(p_values, method="benjamini-hochberg", alpha=alpha, labels=labels)


def compare_corrections(
    p_values: npt.ArrayLike,
    *,
    alpha: float = DEFAULT_ALPHA,
    labels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Tabulate raw, Bonferroni and Benjamini-Hochberg results side by side.

    Columns are the raw p-value, both adjusted p-values, and the three rejection
    decisions, sorted by raw p-value. Useful for showing how a conclusion changes
    once the size of the family is taken into account.
    """
    fwer = bonferroni(p_values, alpha=alpha, labels=labels)
    fdr = benjamini_hochberg(p_values, alpha=alpha, labels=labels)
    return pd.DataFrame(
        {
            "p_value": fwer.p_values,
            "bonferroni_p": fwer.adjusted_p_values,
            "benjamini_hochberg_p": fdr.adjusted_p_values,
            "rejected_uncorrected": fwer.p_values <= alpha,
            "rejected_bonferroni": fwer.rejected,
            "rejected_benjamini_hochberg": fdr.rejected,
        },
        index=pd.Index(fwer.labels, name="test"),
    ).sort_values("p_value")
