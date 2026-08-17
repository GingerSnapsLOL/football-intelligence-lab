"""Feature definitions and preprocessing shared by every shot model.

Model comparison is only meaningful when the models are given the same problem.
The target, the feature sets, the leakage rule and the preprocessing therefore live
here rather than in any one model's module, so a new model cannot quietly redefine
them and win by using different information.
"""

import logging
from collections.abc import Sequence
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

TARGET: Final = "goal"

#: The geometry baseline: how far out, and how much goal is visible.
BASELINE_NUMERIC: Final = ("shot_distance", "shot_angle")

#: Context available before the ball is struck.
CONTEXT_CATEGORICAL: Final = ("body_part", "shot_type", "technique")
CONTEXT_BOOLEAN: Final = ("under_pressure", "first_time")

#: Fields that must never be used as features, with the reason.
FORBIDDEN_FEATURES: Final[dict[str, str]] = {
    "statsbomb_xg": (
        "another model's estimate of the target, computed from information this model does "
        "not use; keep it as a benchmark"
    ),
    "goal": "the target itself",
    "outcome": "the target in categorical form",
    "outcome_id": "the target in categorical form",
    "end_x": "recorded after the shot was struck",
    "end_y": "recorded after the shot was struck",
    "end_z": "recorded after the shot was struck",
}

#: Categorical levels rarer than this share are pooled into an "infrequent" bucket.
MIN_CATEGORY_FREQUENCY: Final = 0.01


class ModelError(ValueError):
    """Raised when a model cannot be built or fitted as requested."""


def reject_leaky_features(features: Sequence[str]) -> None:
    """Refuse any feature that encodes the outcome or postdates the shot.

    Enforced in code rather than left to discipline: the single most damaging
    mistake available in this project is training on ``statsbomb_xg``.
    """
    for feature in features:
        if feature in FORBIDDEN_FEATURES:
            raise ModelError(
                f"{feature!r} cannot be used as a feature: {FORBIDDEN_FEATURES[feature]}."
            )


def build_preprocessor(
    numeric: Sequence[str] = BASELINE_NUMERIC,
    categorical: Sequence[str] = (),
    boolean: Sequence[str] = (),
) -> ColumnTransformer:
    """Column transformer shared by every model, so they see identical inputs.

    Numeric features are standardised; categorical features are one-hot encoded
    with the first level dropped and rare levels pooled; booleans pass through.

    Standardising is irrelevant to a tree model, which only ever compares values
    within a feature, but it is kept for the tree models too so that every model
    is fitted on exactly the same matrix.

    Raises:
        ModelError: if no features are supplied, a feature is repeated, or a
            feature is on the forbidden list.
    """
    features = [*numeric, *categorical, *boolean]
    if not features:
        raise ModelError("At least one feature is required.")
    reject_leaky_features(features)
    duplicates = {name for name in features if features.count(name) > 1}
    if duplicates:
        raise ModelError(f"Features repeated across groups: {sorted(duplicates)}.")

    transformers: list[tuple[str, object, list[str]]] = []
    if numeric:
        transformers.append(("numeric", StandardScaler(), list(numeric)))
    if categorical:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="infrequent_if_exist",
                    min_frequency=MIN_CATEGORY_FREQUENCY,
                    sparse_output=False,
                ),
                list(categorical),
            )
        )
    if boolean:
        transformers.append(("boolean", "passthrough", list(boolean)))

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def build_passthrough_preprocessor(
    numeric: Sequence[str] = BASELINE_NUMERIC,
    categorical: Sequence[str] = (),
    boolean: Sequence[str] = (),
) -> ColumnTransformer:
    """Select and order the same columns, but leave categoricals as they are.

    For a model that handles categorical variables natively -- CatBoost -- one-hot
    encoding would throw away the very information the model is designed to use.
    This transformer therefore only selects and orders columns, emitting a
    DataFrame so the downstream model can identify the categorical ones by name.

    The *information* given to the model is identical to
    :func:`build_preprocessor`; only the encoding differs, which is the point of
    the comparison.

    Raises:
        ModelError: on the same conditions as :func:`build_preprocessor`.
    """
    features = [*numeric, *categorical, *boolean]
    if not features:
        raise ModelError("At least one feature is required.")
    reject_leaky_features(features)
    duplicates = {name for name in features if features.count(name) > 1}
    if duplicates:
        raise ModelError(f"Features repeated across groups: {sorted(duplicates)}.")

    transformers: list[tuple[str, object, list[str]]] = []
    for name, selection in (
        ("numeric", list(numeric)),
        ("categorical", list(categorical)),
        ("boolean", list(boolean)),
    ):
        if selection:
            transformers.append((name, "passthrough", selection))

    transformer = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    return transformer.set_output(transform="pandas")


def feature_columns(pipeline: Pipeline) -> list[str]:
    """The raw column names a pipeline expects, in the order it consumes them."""
    transformer = pipeline.named_steps["features"]
    columns: list[str] = []
    for _, _, selection in transformer.transformers:
        columns.extend(selection)
    return columns


def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    """Column names after preprocessing, one per model input."""
    return list(pipeline.named_steps["features"].get_feature_names_out())


def prepare(pipeline: Pipeline, frame: pd.DataFrame, *, target: str = TARGET) -> pd.DataFrame:
    """Check a frame supplies everything a pipeline declares, and select it.

    Raises:
        ModelError: if a required column is missing.
    """
    required = [*feature_columns(pipeline), target]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ModelError(f"The dataset is missing required column(s): {missing}.")
    return frame[feature_columns(pipeline)]


def fit(pipeline: Pipeline, frame: pd.DataFrame, *, target: str = TARGET) -> Pipeline:
    """Fit a pipeline on a shot dataset, taking only the columns it declares.

    Raises:
        ModelError: if a required column is missing, or the target is not binary
            with both classes present.
    """
    features = prepare(pipeline, frame, target=target)
    labels = frame[target].to_numpy()
    if len(np.unique(labels)) < 2:
        raise ModelError(
            f"{target!r} has a single value in this data, so a model cannot be fitted."
        )
    pipeline.fit(features, labels.astype(int))
    return pipeline


def predict_probability(pipeline: Pipeline, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
    """Predicted probability of a goal for each row, as a 1-D array in [0, 1]."""
    probabilities = pipeline.predict_proba(frame[feature_columns(pipeline)])
    return np.asarray(probabilities[:, 1], dtype=np.float64)
