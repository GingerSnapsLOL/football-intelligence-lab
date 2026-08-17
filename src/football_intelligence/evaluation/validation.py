"""Validation splits that respect the structure of football data.

A random split of individual shots is the default in most tutorials and is wrong
here. Shots are clustered within matches: two shots from the same match share the
teams, the game state, the pitch and the referee. Splitting them at random puts
shots from one match on both sides of the divide, so the test set is not
independent of the training set and the estimated performance is optimistic.

The fix is to split whole **matches**. Everything from a match goes to one side or
the other, so the test set contains only matches the model has never seen -- which
is the situation the model will face in use.

This module currently holds only the group split needed by the first model.
Temporal splits, grouping by player and cross-validated comparisons of the
strategies belong to the validation task and will extend it.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger(__name__)


class ValidationError(ValueError):
    """Raised when a split cannot be formed as requested."""


@dataclass(frozen=True, slots=True)
class GroupSplit:
    """A train/test split that keeps whole groups together."""

    train: pd.DataFrame
    test: pd.DataFrame
    group_column: str

    @property
    def n_train_groups(self) -> int:
        return int(self.train[self.group_column].nunique())

    @property
    def n_test_groups(self) -> int:
        return int(self.test[self.group_column].nunique())

    @property
    def overlapping_groups(self) -> set[object]:
        """Groups appearing on both sides. Must be empty for the split to be valid."""
        return set(self.train[self.group_column]) & set(self.test[self.group_column])

    def __str__(self) -> str:
        return "\n".join(
            [
                f"  train  {len(self.train):,} rows from {self.n_train_groups} "
                f"{self.group_column} groups",
                f"  test   {len(self.test):,} rows from {self.n_test_groups} "
                f"{self.group_column} groups",
                f"  groups on both sides: {len(self.overlapping_groups)}",
            ]
        )


def group_train_test_split(
    frame: pd.DataFrame,
    *,
    group_column: str = "match_id",
    test_size: float = 0.25,
    random_state: int | None = None,
) -> GroupSplit:
    """Split rows into train and test so that no group spans both sides.

    Args:
        frame: Rows to split.
        group_column: The unit that must stay intact, ``match_id`` by default.
        test_size: Approximate share of **groups** assigned to the test set. The
            share of rows will differ, since groups vary in size.
        random_state: Seed, for a reproducible split.

    Raises:
        ValidationError: if the column is missing, or the split would leave one
            side empty.
    """
    if group_column not in frame.columns:
        raise ValidationError(f"{group_column!r} is not a column of the frame.")
    if not 0.0 < test_size < 1.0:
        raise ValidationError(f"test_size must lie strictly between 0 and 1, got {test_size}.")

    groups = frame[group_column].to_numpy()
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise ValidationError(
            f"Need at least 2 distinct {group_column} values to split, got {n_groups}."
        )

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_index, test_index = next(splitter.split(frame, groups=groups))
    split = GroupSplit(
        train=frame.iloc[train_index].reset_index(drop=True),
        test=frame.iloc[test_index].reset_index(drop=True),
        group_column=group_column,
    )

    if split.train.empty or split.test.empty:
        raise ValidationError("The split left one side empty; adjust test_size.")
    logger.info(
        "split %d rows into %d train / %d test, %d / %d %s groups",
        len(frame),
        len(split.train),
        len(split.test),
        split.n_train_groups,
        split.n_test_groups,
        group_column,
    )
    return split


def assert_no_group_leakage(split: GroupSplit) -> None:
    """Fail loudly if any group appears on both sides of the split.

    Cheap to run and worth running: a split that silently leaks is the single
    easiest way to report a model as better than it is.
    """
    overlap = split.overlapping_groups
    if overlap:
        raise ValidationError(
            f"{len(overlap)} {split.group_column} value(s) appear in both train and test, "
            f"for example {sorted(map(str, overlap))[:5]}. The evaluation would be optimistic."
        )
