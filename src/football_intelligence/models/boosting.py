"""Machinery shared by every gradient-boosting model.

The three boosting libraries differ in how they grow trees and how they are
called, but the surrounding design must be identical for a comparison to mean
anything: the same features, the same split, and the same rule for choosing how
many trees to keep.

Early stopping and the leakage it invites
-----------------------------------------

Boosting keeps adding trees until told to stop, so the number of trees is a
hyperparameter that has to be chosen from data. Two ways of choosing it are
wrong:

- **On the test set.** Whatever is then reported is no longer an estimate of
  performance on unseen data, because the test set helped build the model.
- **On a random row split of the training data.** Shots are clustered within
  matches. A validation slice sharing matches with the training slice sees an
  optimistic loss curve, so stopping comes too late and the model is left
  overfitted.

:func:`prepare_early_stopping_data` carves the validation slice out of the
training data **by match**, and every boosting model in this project uses it.
"""

import logging
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.pipeline import Pipeline

from football_intelligence.evaluation.validation import GroupSplit, group_train_test_split
from football_intelligence.models.preprocessing import TARGET, ModelError, feature_columns

logger = logging.getLogger(__name__)

#: Upper bound on the number of trees; early stopping chooses the actual number.
MAX_TREES: Final = 600

#: Stop when the validation loss has not improved for this many rounds.
EARLY_STOPPING_ROUNDS: Final = 40

#: Share of the *training* matches held back to decide when to stop.
VALIDATION_FRACTION: Final = 0.2


@dataclass(frozen=True, slots=True)
class BoostedFit:
    """A fitted boosting pipeline and what early stopping decided."""

    pipeline: Pipeline
    library: str
    n_trees: int
    best_iteration: int
    best_validation_loss: float
    n_validation_rows: int
    n_validation_groups: int
    stopped_early: bool

    def __str__(self) -> str:
        verdict = (
            f"stopped at {self.n_trees} trees"
            if self.stopped_early
            else f"used the full budget of {self.n_trees} trees without stopping"
        )
        return "\n".join(
            [
                f"  library           {self.library}",
                f"  early stopping    {verdict}",
                f"  best iteration    {self.best_iteration}",
                f"  validation loss   {self.best_validation_loss:.4f}",
                f"  validation set    {self.n_validation_rows:,} rows from "
                f"{self.n_validation_groups} held-out training matches",
            ]
        )


@dataclass(frozen=True, slots=True)
class EarlyStoppingData:
    """Preprocessed training and validation matrices for a boosting fit.

    The feature matrices keep whatever type the preprocessor produced: an array
    for the one-hot pipelines, a DataFrame for the passthrough pipeline that lets
    CatBoost identify its categorical columns by name.
    """

    train_features: npt.NDArray[np.float64] | pd.DataFrame
    train_labels: npt.NDArray[np.int64]
    validation_features: npt.NDArray[np.float64] | pd.DataFrame
    validation_labels: npt.NDArray[np.int64]
    split: GroupSplit


def prepare_early_stopping_data(
    pipeline: Pipeline,
    train: pd.DataFrame,
    *,
    target: str = TARGET,
    group_column: str = "match_id",
    validation_fraction: float = VALIDATION_FRACTION,
    random_state: int | None = None,
) -> EarlyStoppingData:
    """Split training data by group and preprocess both halves.

    The transformer is fitted on the inner training slice only, so the validation
    slice is transformed with statistics it did not contribute to.

    Raises:
        ModelError: if a required column is missing, the target has one value, or
            the validation slice ends up without both outcomes.
    """
    required = [*feature_columns(pipeline), target, group_column]
    missing = [column for column in required if column not in train.columns]
    if missing:
        raise ModelError(f"The dataset is missing required column(s): {missing}.")
    if len(np.unique(train[target].to_numpy())) < 2:
        raise ModelError(f"{target!r} has a single value in this data.")

    inner = group_train_test_split(
        train,
        group_column=group_column,
        test_size=validation_fraction,
        random_state=random_state,
    )
    if len(np.unique(inner.test[target].to_numpy())) < 2:
        raise ModelError(
            "The validation slice contains a single outcome, so early stopping has nothing "
            "to watch. Try a larger validation_fraction or a different seed."
        )

    columns = feature_columns(pipeline)
    preprocessor = pipeline.named_steps["features"]
    return EarlyStoppingData(
        train_features=preprocessor.fit_transform(inner.train[columns]),
        train_labels=inner.train[target].to_numpy().astype(int),
        validation_features=preprocessor.transform(inner.test[columns]),
        validation_labels=inner.test[target].to_numpy().astype(int),
        split=inner,
    )
