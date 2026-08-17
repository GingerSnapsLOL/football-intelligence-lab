"""Gradient-boosted trees for shot probability, using XGBoost.

What gradient boosting does
---------------------------

Fit a very shallow tree to the data. It will be wrong almost everywhere. Compute
the gradient of the loss with respect to the current prediction -- for a
log-loss classifier this is essentially "observed minus predicted" -- and fit the
*next* tree to that residual signal. Add it to the running total, scaled down by
the learning rate, and repeat. The prediction is an additive sum of hundreds of
small corrections, each one attending to what its predecessors got wrong.

Two consequences matter here. First, the model can represent interactions and
non-monotone responses that a linear logistic model cannot: a header from six
yards is not "header" plus "six yards", and a tree can say so by splitting on both.
Second, because every tree is fitted to the previous errors, boosting will happily
keep fitting noise -- which is what the regularisation below exists to prevent.

Boosting versus bagging
-----------------------

A bagged ensemble (a random forest) fits many deep trees **independently** on
bootstrap samples and averages them. Each tree has low bias and high variance, and
averaging cancels the variance. The trees can be built in parallel and adding more
never really hurts.

Boosting fits shallow trees **sequentially**, each one correcting the last. Each
tree has high bias and low variance, and the sum reduces the bias. The trees cannot
be built in parallel, and adding too many *does* hurt: the ensemble eventually
starts fitting noise. That asymmetry is the whole reason boosting needs a learning
rate and early stopping while a forest does not.

The knobs that matter, and what they do here
--------------------------------------------

- **learning_rate** (0.05): how much of each tree's correction is kept. Lower means
  slower, steadier learning that generalises better, at the cost of needing more
  trees. It trades against ``n_estimators`` almost exactly.
- **n_estimators** (capped at 600): how many corrections. Not tuned by hand --
  early stopping picks the number by watching a held-out set, which is the only
  honest way to choose it.
- **max_depth** (3): how many splits deep each tree may go, and therefore the
  order of interaction it can express. Depth 3 already allows three-way
  interactions. With fewer than 300 goals in the training data, deeper trees would
  memorise individuals rather than learn patterns.
- **min_child_weight** (5) and **subsample**/**colsample_bytree** (0.8): a leaf must
  carry real evidence, and each tree sees only part of the data and part of the
  features. Both damp the tendency to fit noise.
- **reg_lambda** (1.0): L2 shrinkage on leaf values, pulling predictions toward the
  ensemble's current average.

These are deliberately modest, defensible settings rather than a search. A large
hyperparameter sweep on 2,918 shots and one hold-out set would mostly be fitting
the split.

The same problem as the baseline
--------------------------------

Target, feature sets, preprocessing and the leakage rule come from
``models.preprocessing``, identical to the logistic baseline, so any difference in
the metrics is a difference between the models rather than between the problems
they were given.
"""

import logging

import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

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

# Re-exported so this module is a complete entry point for the boosted model.
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
    "build_xgboost_model",
    "feature_columns",
    "feature_importance",
    "fit_with_early_stopping",
    "transformed_feature_names",
]


def build_xgboost_model(
    numeric: tuple[str, ...] | list[str] = BASELINE_NUMERIC,
    categorical: tuple[str, ...] | list[str] = CONTEXT_CATEGORICAL,
    boolean: tuple[str, ...] | list[str] = CONTEXT_BOOLEAN,
    *,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    min_child_weight: float = 5.0,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
    n_estimators: int = MAX_TREES,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    random_state: int | None = None,
) -> Pipeline:
    """Assemble the shared preprocessing with an XGBoost classifier.

    Defaults are the modest, justified settings described in the module docstring.
    They are not the product of a search and should not be presented as tuned.
    """
    return Pipeline(
        [
            ("features", build_preprocessor(numeric, categorical, boolean)),
            (
                "model",
                XGBClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    min_child_weight=min_child_weight,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    reg_lambda=reg_lambda,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    early_stopping_rounds=early_stopping_rounds,
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )


def build_baseline_model(*, random_state: int | None = None) -> Pipeline:
    """Distance and angle only, for a like-for-like comparison with the baseline."""
    return build_xgboost_model(BASELINE_NUMERIC, (), (), random_state=random_state)


def build_contextual_model(*, random_state: int | None = None) -> Pipeline:
    """Geometry plus the shared pre-shot context."""
    return build_xgboost_model(random_state=random_state)


def fit_with_early_stopping(
    pipeline: Pipeline,
    train: pd.DataFrame,
    *,
    target: str = TARGET,
    group_column: str = "match_id",
    validation_fraction: float = VALIDATION_FRACTION,
    random_state: int | None = None,
) -> BoostedFit:
    """Fit the pipeline, choosing the number of trees on a held-out slice of train.

    The validation slice is carved out **by match**, exactly as the outer
    train/test split is. Using a random row split here would leak: shots from a
    match in the validation slice would also be in the training slice, early
    stopping would see an optimistic curve, and it would stop too late.

    The test set is never involved. Choosing the number of trees on the data used
    to report performance is a subtle but complete invalidation of that report.

    Args:
        pipeline: An unfitted pipeline from :func:`build_xgboost_model`.
        train: Training rows only.
        target: Target column.
        group_column: Column defining the groups that must not be split.
        validation_fraction: Share of training *groups* held back.
        random_state: Seed for the internal split.

    Raises:
        ModelError: if the data lacks a required column, has one class only, or
            leaves a validation slice without both outcomes.
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
        eval_set=[(data.validation_features, data.validation_labels)],
        verbose=False,
    )

    booster = model.get_booster()
    n_trees = int(booster.num_boosted_rounds())
    best_iteration = int(getattr(model, "best_iteration", n_trees - 1))
    best_score = float(getattr(model, "best_score", float("nan")))

    logger.info(
        "boosting fitted %d trees (best iteration %d, validation loss %.4f)",
        n_trees,
        best_iteration,
        best_score,
    )
    return BoostedFit(
        pipeline=pipeline,
        library="xgboost",
        n_trees=n_trees,
        best_iteration=best_iteration,
        best_validation_loss=best_score,
        n_validation_rows=len(data.split.test),
        n_validation_groups=data.split.n_test_groups,
        stopped_early=n_trees < model.n_estimators,
    )


def feature_importance(pipeline: Pipeline, *, importance_type: str = "gain") -> pd.DataFrame:
    """Importance of each transformed feature, largest first.

    ``gain`` is the average improvement in the loss brought by splits on a
    feature, which is the most informative of the built-in measures. It says how
    much a feature was *used*, not how it acts or in which direction: a feature can
    score highly by being split on for opposite reasons in different regions.
    Attribution per prediction needs SHAP, which comes later.
    """
    model = pipeline.named_steps["model"]
    booster = model.get_booster()
    names = transformed_feature_names(pipeline)
    scores = booster.get_score(importance_type=importance_type)

    # XGBoost keys features as f0, f1, ... when fitted on a plain array.
    values = [scores.get(f"f{index}", 0.0) for index in range(len(names))]
    total = float(sum(values))
    frame = pd.DataFrame(
        {
            "feature": names,
            importance_type: values,
            "share": [value / total if total else 0.0 for value in values],
        }
    )
    return frame.sort_values(importance_type, ascending=False, ignore_index=True)
