"""Metrics for probability models.

Expected goals is a **probability estimation** problem, not a classification one.
Nobody wants a yes/no verdict on whether a shot was a goal -- the outcome is
already known -- and thresholding a predicted probability at 0.5 would classify
every shot in this dataset as a miss while still being 90% "accurate". Accuracy is
therefore absent from this module, deliberately.

What the four metrics measure
-----------------------------

**ROC-AUC** measures *discrimination*: given one goal and one non-goal, how often
does the model give the goal the higher probability? It is invariant to any
monotone transformation of the predictions, so a model that outputs perfectly
ordered but wildly miscalibrated probabilities scores just as well as one that is
exactly right. It also ignores prevalence, which flatters models on imbalanced
data.

**PR-AUC** (average precision) also measures ordering, but only among the
predicted positives, so it is far more sensitive to performance on the rare class.
Unlike ROC-AUC its baseline is the prevalence itself, so the number is not
comparable across datasets with different goal rates.

**Log loss** is a *proper scoring rule*: it is minimised, in expectation, only by
the true probabilities. It punishes confident mistakes harshly -- a probability of
0.01 on a shot that scored costs 4.6 nats -- which is exactly what an xG model
should be held to.

**Brier score** is the mean squared error of the probabilities. Also proper, but
quadratic rather than logarithmic, so it is gentler on confident errors and easier
to decompose into calibration and refinement.

Why the baselines matter
------------------------

Log loss and Brier are not scale-free: 0.30 is meaningless without knowing what a
model that learns nothing would score. Every result here therefore carries the
value achieved by predicting the **base rate** for every shot, along with the skill
score ``1 - metric / baseline``, which is 0 for the naive model and 1 for a perfect
one. A model with a Brier skill score of 0.05 has captured very little, however
respectable its ROC-AUC looks.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

logger = logging.getLogger(__name__)


class MetricError(ValueError):
    """Raised when metrics cannot be computed from the inputs as supplied."""


@dataclass(frozen=True, slots=True)
class ProbabilityMetrics:
    """Discrimination and calibration-sensitive metrics for one set of predictions."""

    n: int
    positives: int
    prevalence: float
    roc_auc: float
    pr_auc: float
    log_loss: float
    brier_score: float
    log_loss_baseline: float
    brier_baseline: float
    mean_prediction: float

    @property
    def log_loss_skill(self) -> float:
        """``1 - log_loss / baseline``: 0 for the base-rate model, 1 for perfection."""
        return 1.0 - self.log_loss / self.log_loss_baseline

    @property
    def brier_skill(self) -> float:
        """``1 - brier / baseline``: 0 for the base-rate model, 1 for perfection."""
        return 1.0 - self.brier_score / self.brier_baseline

    @property
    def calibration_in_the_large(self) -> float:
        """Mean prediction minus observed prevalence: overall bias of the model."""
        return self.mean_prediction - self.prevalence

    def to_series(self) -> pd.Series:
        return pd.Series(
            {
                "n": self.n,
                "positives": self.positives,
                "prevalence": self.prevalence,
                "roc_auc": self.roc_auc,
                "pr_auc": self.pr_auc,
                "log_loss": self.log_loss,
                "log_loss_skill": self.log_loss_skill,
                "brier_score": self.brier_score,
                "brier_skill": self.brier_skill,
                "mean_prediction": self.mean_prediction,
                "calibration_in_the_large": self.calibration_in_the_large,
            }
        )

    def __str__(self) -> str:
        return "\n".join(
            [
                f"  n / positives     {self.n:,} / {self.positives:,} "
                f"(prevalence {self.prevalence:.4f})",
                f"  ROC-AUC           {self.roc_auc:.4f}   (0.5 = no discrimination)",
                f"  PR-AUC            {self.pr_auc:.4f}   "
                f"(baseline = prevalence = {self.prevalence:.4f})",
                f"  log loss          {self.log_loss:.4f}   "
                f"(base rate {self.log_loss_baseline:.4f}, skill {self.log_loss_skill:+.4f})",
                f"  Brier score       {self.brier_score:.4f}   "
                f"(base rate {self.brier_baseline:.4f}, skill {self.brier_skill:+.4f})",
                f"  mean prediction   {self.mean_prediction:.4f}   "
                f"(observed {self.prevalence:.4f}, bias {self.calibration_in_the_large:+.4f})",
            ]
        )


def _validate(
    y_true: npt.ArrayLike, y_prob: npt.ArrayLike
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    truth = np.asarray(y_true)
    probabilities = np.asarray(y_prob, dtype=np.float64)

    if truth.shape != probabilities.shape:
        raise MetricError(
            f"y_true and y_prob must have the same shape, got {truth.shape} and "
            f"{probabilities.shape}."
        )
    if truth.size == 0:
        raise MetricError("Cannot compute metrics on an empty sample.")
    if not np.all(np.isfinite(probabilities)):
        raise MetricError("Predicted probabilities contain non-finite values.")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise MetricError(
            f"Predicted probabilities must lie in [0, 1]; got a range of "
            f"[{probabilities.min():.4g}, {probabilities.max():.4g}]."
        )

    binary = np.asarray(truth, dtype=np.float64)
    if not np.all(np.isin(binary, (0.0, 1.0))):
        raise MetricError("y_true must be binary (0/1 or False/True).")
    labels = binary.astype(np.int64)
    if labels.min() == labels.max():
        raise MetricError(
            "y_true contains only one class, so discrimination metrics are undefined. This "
            "usually means the split left a fold with no goals in it."
        )
    return labels, probabilities


def evaluate_probabilities(y_true: npt.ArrayLike, y_prob: npt.ArrayLike) -> ProbabilityMetrics:
    """Score predicted probabilities against binary outcomes.

    Args:
        y_true: Binary outcomes, 0/1 or False/True.
        y_prob: Predicted probability of the positive class.

    Raises:
        MetricError: for mismatched shapes, empty input, probabilities outside
            [0, 1], non-binary outcomes, or a sample containing only one class.
    """
    labels, probabilities = _validate(y_true, y_prob)
    prevalence = float(np.mean(labels))
    baseline = np.full_like(probabilities, prevalence)

    return ProbabilityMetrics(
        n=int(labels.size),
        positives=int(labels.sum()),
        prevalence=prevalence,
        roc_auc=float(roc_auc_score(labels, probabilities)),
        pr_auc=float(average_precision_score(labels, probabilities)),
        log_loss=float(log_loss(labels, probabilities, labels=[0, 1])),
        brier_score=float(brier_score_loss(labels, probabilities)),
        log_loss_baseline=float(log_loss(labels, baseline, labels=[0, 1])),
        brier_baseline=float(brier_score_loss(labels, baseline)),
        mean_prediction=float(np.mean(probabilities)),
    )


def compare_models(y_true: npt.ArrayLike, predictions: Mapping[str, npt.ArrayLike]) -> pd.DataFrame:
    """Score several sets of predictions on the same outcomes, one row each.

    Every model must be evaluated on identical data for the comparison to mean
    anything, which is why the outcomes are passed once.
    """
    if not predictions:
        raise MetricError("No predictions supplied to compare.")
    rows = {
        name: evaluate_probabilities(y_true, probabilities).to_series()
        for name, probabilities in predictions.items()
    }
    return pd.DataFrame(rows).T
