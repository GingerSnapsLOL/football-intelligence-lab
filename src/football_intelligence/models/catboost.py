"""Gradient-boosted trees for shot probability, using CatBoost.

The third boosting library, and the one that treats categorical variables as
first-class citizens rather than as something to be one-hot encoded away. Two
ideas do the work, and both exist to solve the same problem: **target leakage**.

The target-leakage problem in categorical encoding
--------------------------------------------------

The obvious way to give a tree a categorical feature is *target encoding*: replace
each level with the mean of the target for that level. "Head" becomes 0.111
because 11.1% of headers were goals.

Computed naively, this leaks. The mean for "Head" includes **this shot's own
outcome**, so a header that scored is handed a slightly inflated number and one
that missed a slightly deflated one. The feature therefore carries a trace of the
answer. Training loss collapses, the model leans on the encoding, and it all
evaporates on new data. The effect is worst exactly where the encoding is most
tempting: a level appearing once is encoded as its own outcome, 0 or 1.

Holding out a fold to compute the statistic helps but wastes data and leaves the
estimate noisy.

Ordered target statistics
-------------------------

CatBoost's answer is to impose an artificial "time": draw a random permutation of
the training rows and, for each row, compute its categorical statistic **using
only the rows that precede it**, smoothed toward a prior:

``encoding = (sum of targets of earlier rows in this category + a * prior) / (count + a)``

No row can see its own outcome, so the encoding is honest. Early rows in the
permutation get noisy estimates dominated by the prior, which is why several
permutations are averaged.

Ordered boosting
----------------

The same leak appears in boosting itself. Standard gradient boosting computes the
residual for a row using a model that was trained on that row, so residuals are
optimistically small and the ensemble drifts -- the "prediction shift" that makes
boosted models overfit small data.

Ordered boosting applies the same trick to the gradients: for each row, the
residual is computed from a model trained only on rows preceding it in a
permutation. It costs more compute and buys a genuinely unbiased gradient
estimate, which matters most when data is scarce -- as it is here.

Why this is useful for categorical tabular data
-----------------------------------------------

- High-cardinality categoricals cost one column, not one column per level, so the
  feature matrix stays small and the trees stay interpretable in shape.
- The encoding carries the category's *relationship to the target* rather than
  mere membership, which a tree would otherwise have to rediscover through many
  splits.
- Combinations of categoricals are constructed automatically, so interactions
  between, say, body part and shot type do not have to be engineered by hand.

When XGBoost or LightGBM is still the better choice
---------------------------------------------------

- **Mostly numeric problems.** Ordered target statistics solve a problem you do
  not have, and you pay for them in training time.
- **Low-cardinality categoricals.** With four body parts and three shot types,
  one-hot encoding costs almost nothing and is easier to reason about -- which is
  the situation in *this* dataset.
- **Large datasets or tight training budgets.** Ordered boosting maintains several
  permutations and is markedly slower per iteration; LightGBM's histogram binning
  is the faster route when rows number in the millions.
- **Ecosystem weight.** XGBoost and LightGBM have wider tooling, deployment and
  explainability support.

On identifier features
----------------------

``player_id`` and ``team_id`` are exactly the high-cardinality categoricals
CatBoost is famous for handling, and they are deliberately **not** used here. See
:data:`IDENTIFIER_FEATURES` for why; passing them requires an explicit opt-in.
"""

import logging
from typing import Final

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
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
    build_passthrough_preprocessor,
    feature_columns,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BASELINE_NUMERIC",
    "CONTEXT_BOOLEAN",
    "CONTEXT_CATEGORICAL",
    "EARLY_STOPPING_ROUNDS",
    "IDENTIFIER_FEATURES",
    "MAX_TREES",
    "TARGET",
    "VALIDATION_FRACTION",
    "BoostedFit",
    "ModelError",
    "build_baseline_model",
    "build_catboost_model",
    "build_contextual_model",
    "categorical_feature_names",
    "feature_columns",
    "feature_importance",
    "fit_with_early_stopping",
]

#: Identifier columns that are *not* used as predictors, and why. CatBoost could
#: encode them, which is precisely the temptation worth resisting.
IDENTIFIER_FEATURES: Final[dict[str, str]] = {
    "player_id": (
        "651 players with a median of 3 shots each: a per-player target statistic is "
        "estimated from three observations and is mostly noise. Worse, the evaluation split "
        "is by match, so the same players appear in train and test -- the model would be "
        "rewarded for remembering individuals rather than for modelling chances, and the "
        "score would not transfer to a player it has not seen"
    ),
    "team_id": (
        "44 teams sharing the same match-level split, so team identity crosses the "
        "train/test boundary in the same way; and xG is meant to describe the quality of a "
        "chance, not who took it"
    ),
    "match_id": "the grouping variable of the evaluation split, not a property of the shot",
}


def _reject_identifier_features(features: list[str], *, allow: bool) -> None:
    if allow:
        return
    for feature in features:
        if feature in IDENTIFIER_FEATURES:
            raise ModelError(
                f"{feature!r} is an identifier, not a shot property: "
                f"{IDENTIFIER_FEATURES[feature]}. Pass allow_identifier_features=True only "
                "with a validation strategy that splits on it."
            )


def build_catboost_model(
    numeric: tuple[str, ...] | list[str] = BASELINE_NUMERIC,
    categorical: tuple[str, ...] | list[str] = CONTEXT_CATEGORICAL,
    boolean: tuple[str, ...] | list[str] = CONTEXT_BOOLEAN,
    *,
    learning_rate: float = 0.05,
    depth: int = 3,
    l2_leaf_reg: float = 3.0,
    subsample: float = 0.8,
    iterations: int = MAX_TREES,
    allow_identifier_features: bool = False,
    random_state: int | None = None,
) -> Pipeline:
    """Assemble a passthrough preprocessor with a CatBoost classifier.

    Categorical columns reach the model as strings and are handled natively with
    ordered target statistics; numeric and boolean columns pass through unchanged.
    The settings mirror the other boosting models wherever an equivalent knob
    exists and are not the product of a search.

    Args:
        allow_identifier_features: Permit ``player_id``/``team_id``/``match_id`` as
            predictors. Off by default; see :data:`IDENTIFIER_FEATURES`.

    Raises:
        ModelError: for leaky features, repeated features, or identifier features
            without the explicit opt-in.
    """
    features = [*numeric, *categorical, *boolean]
    _reject_identifier_features(features, allow=allow_identifier_features)

    return Pipeline(
        [
            ("features", build_passthrough_preprocessor(numeric, categorical, boolean)),
            (
                "model",
                CatBoostClassifier(
                    iterations=iterations,
                    learning_rate=learning_rate,
                    depth=depth,
                    l2_leaf_reg=l2_leaf_reg,
                    subsample=subsample,
                    bootstrap_type="Bernoulli",
                    loss_function="Logloss",
                    eval_metric="Logloss",
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                    random_seed=random_state,
                    # Ordered boosting is the point of using this library, and it
                    # is the mode that helps most on a dataset this small.
                    boosting_type="Ordered",
                    allow_writing_files=False,
                    thread_count=1,
                    verbose=False,
                ),
            ),
        ]
    )


def build_baseline_model(*, random_state: int | None = None) -> Pipeline:
    """Distance and angle only, for a like-for-like comparison."""
    return build_catboost_model(BASELINE_NUMERIC, (), (), random_state=random_state)


def build_contextual_model(*, random_state: int | None = None) -> Pipeline:
    """Geometry plus the shared pre-shot context, categoricals handled natively."""
    return build_catboost_model(random_state=random_state)


def categorical_feature_names(pipeline: Pipeline) -> list[str]:
    """The columns CatBoost should treat as categorical."""
    transformer = pipeline.named_steps["features"]
    for name, _, selection in transformer.transformers:
        if name == "categorical":
            return list(selection)
    return []


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

    Uses the same match-level validation slice as the other boosting models, via
    the shared
    :func:`~football_intelligence.models.boosting.prepare_early_stopping_data`.
    The test set is never involved.

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
    categorical = categorical_feature_names(pipeline)
    model = pipeline.named_steps["model"]
    model.fit(
        data.train_features,
        data.train_labels,
        eval_set=(data.validation_features, data.validation_labels),
        cat_features=categorical or None,
        verbose=False,
    )

    n_trees = int(model.tree_count_)
    best_iteration = int(model.get_best_iteration())
    best_score = float(model.get_best_score()["validation"]["Logloss"])

    logger.info(
        "catboost fitted %d trees (best iteration %d, validation loss %.4f)",
        n_trees,
        best_iteration,
        best_score,
    )
    return BoostedFit(
        pipeline=pipeline,
        library="catboost",
        n_trees=n_trees,
        best_iteration=best_iteration,
        best_validation_loss=best_score,
        n_validation_rows=len(data.split.test),
        n_validation_groups=data.split.n_test_groups,
        stopped_early=n_trees < model.get_params()["iterations"],
    )


def feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Importance of each raw feature, largest first.

    CatBoost reports importance per *original* feature, so a categorical appears
    once rather than as a row per level -- which makes it more directly readable
    than the one-hot importances of the other two libraries, and equally
    incomparable with them.
    """
    model = pipeline.named_steps["model"]
    names = list(model.feature_names_)
    values = np.asarray(model.get_feature_importance(), dtype=float)
    total = float(values.sum())
    frame = pd.DataFrame(
        {
            "feature": names,
            "importance": values,
            "share": values / total if total else np.zeros_like(values),
        }
    )
    return frame.sort_values("importance", ascending=False, ignore_index=True)
