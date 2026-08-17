"""Gradient-boosted trees for shot probability, using LightGBM.

The same algorithm family as the XGBoost model, with two engineering decisions
that make it fast and one that makes it dangerous on small data.

Histogram-based training
------------------------

To split a node, a tree must find the best threshold on some feature. The obvious
way is to sort the feature values and evaluate every candidate split, which costs
``O(rows x features)`` per node and repeats the sort work at every level.

LightGBM instead **bins each feature once**, up front, into at most 255 buckets,
and then works entirely with the bin indices. Finding a split becomes a scan over
histogram buckets rather than over rows: ``O(bins x features)``, independent of
the number of rows. Building a child's histogram is cheaper still, because the
histogram of one child can be obtained by subtracting its sibling's from the
parent's -- the "histogram subtraction" trick. Memory drops too, since a bin index
fits in a byte where a float needs eight.

The cost is a small loss of precision: a threshold can only fall on a bin edge.
With 255 bins that is rarely the binding constraint, and modern XGBoost uses a
histogram method by default as well -- the difference between the libraries is
narrower than it once was.

Leaf-wise growth, and why it overfits here
------------------------------------------

XGBoost grows trees **level-wise** by default: every node at a depth is split
before moving deeper, so trees stay balanced and depth alone bounds complexity.

LightGBM grows **leaf-wise**: at each step it splits whichever leaf anywhere in
the tree promises the largest loss reduction. That reaches a lower training loss
for the same number of leaves, because the budget is spent where it helps most.
It also grows deep, narrow branches down whichever path the data happens to
favour -- and on a small dataset, that path is often noise. A leaf carved out for
nine shots that happened to go in is a memorised accident, not a pattern.

This is why the configuration below is deliberately tight: **``num_leaves`` is 8,
not the default 31**, and ``min_child_samples`` is raised to 30. With 2,172
training rows and roughly 210 goals, the default settings would let a single tree
isolate a handful of shots. ``max_depth`` is set to 3 as a second, redundant
bound, so the two boosting models are constrained comparably.

The comparison is deliberately like-for-like
--------------------------------------------

Target, features, preprocessing, split, early stopping and metrics are the shared
objects used by every other model in the project. Nothing here is tuned, and no
alternative split was tried: the settings mirror the XGBoost ones wherever the two
libraries have an equivalent knob.
"""

import logging

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.pipeline import Pipeline

from football_intelligence.models.boosting import (
    EARLY_STOPPING_ROUNDS,
    MAX_TREES,
    VALIDATION_FRACTION,
    BoostedFit,
    prepare_early_stopping_data,
)
from football_intelligence.models.preprocessing import (
    BASELINE_NUMERIC,
    CONTEXT_BOOLEAN,
    CONTEXT_CATEGORICAL,
    TARGET,
    ModelError,
    build_preprocessor,
    feature_columns,
    transformed_feature_names,
)

logger = logging.getLogger(__name__)

# Re-exported so this module is a complete entry point for the LightGBM model.
__all__ = [
    "BASELINE_NUMERIC",
    "CONTEXT_BOOLEAN",
    "CONTEXT_CATEGORICAL",
    "EARLY_STOPPING_ROUNDS",
    "MAX_TREES",
    "TARGET",
    "VALIDATION_FRACTION",
    "BoostedFit",
    "ModelError",
    "build_baseline_model",
    "build_contextual_model",
    "build_lightgbm_model",
    "feature_columns",
    "feature_importance",
    "fit_with_early_stopping",
    "transformed_feature_names",
]

#: Far below LightGBM's default of 31. Leaf-wise growth with a generous leaf
#: budget is exactly how a model memorises a few hundred goals.
DEFAULT_NUM_LEAVES: int = 8

#: A leaf must be supported by at least this many shots.
DEFAULT_MIN_CHILD_SAMPLES: int = 30


def build_lightgbm_model(
    numeric: tuple[str, ...] | list[str] = BASELINE_NUMERIC,
    categorical: tuple[str, ...] | list[str] = CONTEXT_CATEGORICAL,
    boolean: tuple[str, ...] | list[str] = CONTEXT_BOOLEAN,
    *,
    learning_rate: float = 0.05,
    num_leaves: int = DEFAULT_NUM_LEAVES,
    max_depth: int = 3,
    min_child_samples: int = DEFAULT_MIN_CHILD_SAMPLES,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
    n_estimators: int = MAX_TREES,
    random_state: int | None = None,
) -> Pipeline:
    """Assemble the shared preprocessing with a LightGBM classifier.

    The settings mirror the XGBoost model wherever an equivalent knob exists, with
    ``num_leaves`` and ``min_child_samples`` tightened for the reasons in the
    module docstring. They are not the product of a search.

    ``subsample_freq`` is set to 1 because LightGBM ignores ``subsample``
    entirely unless it is told how often to resample -- a quiet default that makes
    row subsampling look enabled when it is not.
    """
    return Pipeline(
        [
            ("features", build_preprocessor(numeric, categorical, boolean)),
            (
                "model",
                LGBMClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    num_leaves=num_leaves,
                    max_depth=max_depth,
                    min_child_samples=min_child_samples,
                    subsample=subsample,
                    subsample_freq=1,
                    colsample_bytree=colsample_bytree,
                    reg_lambda=reg_lambda,
                    objective="binary",
                    random_state=random_state,
                    n_jobs=1,
                    verbosity=-1,
                ),
            ),
        ]
    )


def build_baseline_model(*, random_state: int | None = None) -> Pipeline:
    """Distance and angle only, for a like-for-like comparison."""
    return build_lightgbm_model(BASELINE_NUMERIC, (), (), random_state=random_state)


def build_contextual_model(*, random_state: int | None = None) -> Pipeline:
    """Geometry plus the shared pre-shot context."""
    return build_lightgbm_model(random_state=random_state)


def fit_with_early_stopping(
    pipeline: Pipeline,
    train: pd.DataFrame,
    *,
    target: str = TARGET,
    group_column: str = "match_id",
    validation_fraction: float = VALIDATION_FRACTION,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    random_state: int | None = None,
) -> BoostedFit:
    """Fit the pipeline, choosing the number of trees on a held-out slice of train.

    The validation slice is carved out **by match** through the shared
    :func:`~football_intelligence.models.boosting.prepare_early_stopping_data`,
    identically to the XGBoost model. The test set is never involved.

    Raises:
        ModelError: if a required column is missing, the target has one value, or
            the validation slice ends up without both outcomes.
    """
    data = prepare_early_stopping_data(
        pipeline,
        train,
        target=target,
        group_column=group_column,
        validation_fraction=validation_fraction,
        random_state=random_state,
    )
    model = pipeline.named_steps["model"]
    model.fit(
        data.train_features,
        data.train_labels,
        eval_X=data.validation_features,
        eval_y=data.validation_labels,
        eval_metric="binary_logloss",
        callbacks=[
            early_stopping(early_stopping_rounds, verbose=False),
            log_evaluation(period=0),
        ],
    )

    booster = model.booster_
    n_trees = int(booster.num_trees())
    best_iteration = int(model.best_iteration_ or n_trees)
    scores = model.best_score_.get("valid_0", {})
    best_score = float(scores.get("binary_logloss", float("nan")))

    logger.info(
        "lightgbm fitted %d trees (best iteration %d, validation loss %.4f)",
        n_trees,
        best_iteration,
        best_score,
    )
    return BoostedFit(
        pipeline=pipeline,
        library="lightgbm",
        n_trees=n_trees,
        best_iteration=best_iteration,
        best_validation_loss=best_score,
        n_validation_rows=len(data.split.test),
        n_validation_groups=data.split.n_test_groups,
        stopped_early=n_trees < model.n_estimators,
    )


def feature_importance(pipeline: Pipeline, *, importance_type: str = "gain") -> pd.DataFrame:
    """Importance of each transformed feature, largest first.

    ``gain`` totals the loss reduction attributable to splits on a feature. As
    with any built-in importance it reports how much a feature was *used*, not how
    it acts or in which direction, and it is not comparable across libraries: the
    two implementations count gain over different tree shapes.
    """
    model = pipeline.named_steps["model"]
    names = transformed_feature_names(pipeline)
    values = np.asarray(
        model.booster_.feature_importance(importance_type=importance_type), dtype=float
    )
    total = float(values.sum())
    frame = pd.DataFrame(
        {
            "feature": names,
            importance_type: values,
            "share": values / total if total else np.zeros_like(values),
        }
    )
    return frame.sort_values(importance_type, ascending=False, ignore_index=True)
