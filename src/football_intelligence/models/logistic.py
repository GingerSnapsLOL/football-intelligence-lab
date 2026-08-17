"""Interpretable logistic-regression baseline for shot probability (xG).

The first model of the project, and deliberately the simplest one that could
work. Two shot geometry features -- how far out, and how much goal is visible --
already capture most of what makes a chance good, and starting here means every
later model has something honest to beat.

Why logistic regression first
-----------------------------

- It outputs calibrated-by-construction probabilities under its own assumptions,
  which is what xG is.
- Its coefficients are readable: each is a change in the log odds of scoring per
  unit of the feature, so the model can be explained rather than merely deployed.
- It is fast enough that the cost of being wrong is a minute, not an afternoon.
- Any gain a gradient-boosted model shows later is only meaningful relative to a
  baseline that was given a fair chance.

Features, the target and the leakage rule are defined once in
``models.preprocessing`` and shared with every other model, so no model can win a
comparison by quietly using different information. In particular ``statsbomb_xg``
is never a feature: it is another model's estimate of the very quantity being
predicted, and the preprocessing layer refuses it.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from football_intelligence.models.preprocessing import (
    BASELINE_NUMERIC,
    CONTEXT_BOOLEAN,
    CONTEXT_CATEGORICAL,
    FORBIDDEN_FEATURES,
    MIN_CATEGORY_FREQUENCY,
    TARGET,
    ModelError,
    build_preprocessor,
    feature_columns,
    fit,
    predict_probability,
    reject_leaky_features,
    transformed_feature_names,
)

logger = logging.getLogger(__name__)

# Re-exported so this module remains the entry point for the baseline model
# without callers reaching into the shared preprocessing layer.
__all__ = [
    "BASELINE_NUMERIC",
    "CONTEXT_BOOLEAN",
    "CONTEXT_CATEGORICAL",
    "FORBIDDEN_FEATURES",
    "MIN_CATEGORY_FREQUENCY",
    "TARGET",
    "ModelError",
    "build_baseline_model",
    "build_contextual_model",
    "build_pipeline",
    "coefficients",
    "feature_columns",
    "fit",
    "predict_probability",
    "reject_leaky_features",
    "transformed_feature_names",
]


def build_pipeline(
    numeric: tuple[str, ...] | list[str] = BASELINE_NUMERIC,
    categorical: tuple[str, ...] | list[str] = (),
    boolean: tuple[str, ...] | list[str] = (),
    *,
    regularisation: float = 1.0,
    max_iter: int = 1_000,
    random_state: int | None = None,
) -> Pipeline:
    """Assemble the shared preprocessing with a logistic regression.

    Args:
        numeric: Continuous columns.
        categorical: Categorical columns; rare levels are pooled.
        boolean: Already-binary columns.
        regularisation: sklearn's ``C``. Smaller means stronger shrinkage.
        max_iter: Solver iteration cap.
        random_state: Seed, for reproducibility.

    Raises:
        ModelError: if no features are supplied, or if any feature leaks the
            outcome.
    """
    return Pipeline(
        [
            ("features", build_preprocessor(numeric, categorical, boolean)),
            (
                "model",
                LogisticRegression(
                    C=regularisation,
                    max_iter=max_iter,
                    random_state=random_state,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def build_baseline_model(*, random_state: int | None = None) -> Pipeline:
    """Distance and angle only: the model everything else has to beat."""
    return build_pipeline(BASELINE_NUMERIC, random_state=random_state)


def build_contextual_model(*, random_state: int | None = None) -> Pipeline:
    """Geometry plus the shared pre-shot context."""
    return build_pipeline(
        BASELINE_NUMERIC,
        CONTEXT_CATEGORICAL,
        CONTEXT_BOOLEAN,
        random_state=random_state,
    )


def coefficients(pipeline: Pipeline) -> pd.DataFrame:
    """Fitted coefficients with their odds-ratio reading, largest effect first.

    ``coefficient`` is the change in the log odds of scoring per unit of the
    (standardised) feature; ``odds_ratio`` is its exponential, the multiplicative
    change in the odds. For a standardised numeric feature "one unit" is one
    standard deviation; for a one-hot column it is the contrast against the
    dropped reference level.
    """
    model = pipeline.named_steps["model"]
    names = transformed_feature_names(pipeline)
    values = np.asarray(model.coef_).ravel()
    frame = pd.DataFrame(
        {
            "feature": names,
            "coefficient": values,
            "odds_ratio": np.exp(values),
        }
    )
    frame = frame.reindex(frame["coefficient"].abs().sort_values(ascending=False).index)
    intercept = pd.DataFrame(
        {
            "feature": ["(intercept)"],
            "coefficient": [float(np.asarray(model.intercept_).ravel()[0])],
            "odds_ratio": [float(np.exp(np.asarray(model.intercept_).ravel()[0]))],
        }
    )
    return pd.concat([intercept, frame], ignore_index=True)
