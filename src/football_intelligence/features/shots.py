"""The canonical shot modelling dataset: one row per shot, with shot geometry.

Coordinate system
-----------------

StatsBomb pitch coordinates, as published in the *StatsBomb Open Data
Specification v1.1*, Appendix 2 (Locations), and re-verified against the data in
this repository:

===========================  ==========================================
Pitch extent                 ``x`` in [0, 120], ``y`` in [0, 80]
Centre spot                  (60, 40)
Attacked goal line           ``x = 120``
Goal posts                   (120, 36) and (120, 44)
Goal centre                  (120, 40)
Crossbar height              2.67
Penalty spot                 (108, 40)
===========================  ==========================================

**One coordinate unit is one yard.** The specification does not say so directly,
but the data settles it: in-play penalties are recorded at a median of
(108.1, 40.05), i.e. 11.9 units from the goal line, and the Laws of the Game put
the penalty mark 12 yards out. The goal mouth spans 8 units, and a regulation
goal is 8 yards (7.32 m) wide. Both agree, so distances below are in yards.

**Attacking direction is already normalised.** Locations are recorded from the
acting team's perspective, always attacking toward ``x = 120``, in every period.
Verified: mean shot ``x`` is 103.8, 104.3, 105.2, 105.1 and 108.4 in periods 1-5
respectively, and exactly 1 of 2,995 shots was taken from ``x < 60``. Coordinates
must therefore **not** be flipped by half, and geometry can be computed against a
single fixed goal.

Geometry
--------

For a shot at :math:`P = (x, y)` with posts :math:`A = (120, 36)` and
:math:`B = (120, 44)`:

*Distance* is the Euclidean distance to the goal centre,

.. math:: d = \\sqrt{(120 - x)^2 + (40 - y)^2}

computed with :func:`numpy.hypot` for numerical stability.

*Angle* is the angle subtended at the shot location by the two posts — the width
of the visible goal mouth, not a proxy. With :math:`u = A - P` and
:math:`v = B - P`,

.. math:: \\theta = \\operatorname{atan2}(|u \\times v|,\\; u \\cdot v)

which expands to

.. math::

    \\theta = \\operatorname{atan2}\\bigl(\\,8\\,|120 - x|,\\;
             (120 - x)^2 + (y - 36)(y - 44)\\bigr).

``atan2`` of the cross and dot products is used rather than
:math:`\\arccos(u \\cdot v / |u||v|)` because it stays accurate for small angles
and handles obtuse angles without a sign correction. The result is in [0, pi]:
it exceeds pi/2 exactly when the shot is inside the circle through both posts,
which is correct — from six yards out and central, the goal really does subtend
more than a right angle.

Assumptions and edge cases:

- The goal mouth is treated as a line segment at ``x = 120``; the crossbar is
  ignored, so this is the planar angle, not the solid angle.
- No allowance is made for defenders or the keeper blocking part of the mouth.
  A "visible" angle would need the freeze frame, which penalties do not have.
- On the goal line between the posts the angle is pi (the posts lie on opposite
  sides); outside the posts it is 0.
- Exactly on a post the geometry is degenerate (both products are zero) and
  ``atan2(0, 0)`` returns 0.
- Non-finite or off-pitch coordinates yield NaN geometry and are flagged rather
  than silently clipped.

Not included here
-----------------

Score difference, game state and historical player/team form are deliberately
absent: each needs information ordered in time, and computing them carelessly is
the most likely source of leakage in this project. ``statsbomb_xg`` is carried as
a *benchmark* column and is excluded from :data:`FEATURE_COLUMNS`, because it is
the output of another model for the target we predict.
"""

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from football_intelligence.data import storage

logger = logging.getLogger(__name__)

# --- Pitch and goal geometry, in yards (see the module docstring) ------------ #
PITCH_LENGTH: Final = 120.0
PITCH_WIDTH: Final = 80.0
GOAL_X: Final = 120.0
GOAL_POST_LEFT_Y: Final = 36.0
GOAL_POST_RIGHT_Y: Final = 44.0
GOAL_CENTRE_Y: Final = 40.0
GOAL_WIDTH: Final = GOAL_POST_RIGHT_Y - GOAL_POST_LEFT_Y

# The furthest any on-pitch point can be from the goal centre: the far corners.
MAX_POSSIBLE_DISTANCE: Final = float(np.hypot(PITCH_LENGTH, GOAL_CENTRE_Y))

DEFAULT_DATASET_PATH: Final = Path("data/processed/shot_dataset.parquet")

TARGET_COLUMN: Final = "goal"

IDENTIFIER_COLUMNS: Final = (
    "shot_id",
    "match_id",
    "competition_id",
    "season_id",
    "team_id",
    "player_id",
)

NUMERIC_FEATURES: Final = ("shot_distance", "shot_angle", "x", "y", "minute", "period")
CATEGORICAL_FEATURES: Final = (
    "shot_type",
    "body_part",
    "technique",
    "play_pattern",
    "position",
)
BOOLEAN_FEATURES: Final = (
    "under_pressure",
    "first_time",
    "one_on_one",
    "open_goal",
    "deflected",
    "aerial_won",
    "follows_dribble",
    "redirect",
)
FEATURE_COLUMNS: Final = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES

#: Model outputs carried for comparison only. Never train on these.
BENCHMARK_COLUMNS: Final = ("statsbomb_xg",)

#: Columns describing data collection rather than football.
PROVENANCE_COLUMNS: Final = ("has_freeze_frame", "high_fidelity_coordinates", "has_valid_location")

CONTEXT_COLUMNS: Final = ("player", "team", "match_date", "stage", "second")

DATASET_COLUMNS: Final = (
    *IDENTIFIER_COLUMNS,
    *CONTEXT_COLUMNS,
    *FEATURE_COLUMNS,
    *PROVENANCE_COLUMNS,
    *BENCHMARK_COLUMNS,
    TARGET_COLUMN,
)


class ShotFeatureError(RuntimeError):
    """Raised when the shot dataset cannot be built or read."""


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def _as_float_array(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    return np.asarray(values, dtype=np.float64)


def shot_distance(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Euclidean distance in yards from a shot location to the goal centre.

    Args:
        x: Shot x coordinate(s), 0 at the defended goal line, 120 at the attacked one.
        y: Shot y coordinate(s), 0 to 80 across the pitch.

    Returns:
        Distance to (120, 40). NaN wherever an input is NaN. Scalars come back as
        zero-dimensional arrays.
    """
    return np.hypot(GOAL_X - _as_float_array(x), GOAL_CENTRE_Y - _as_float_array(y))


def shot_angle(
    x: npt.ArrayLike, y: npt.ArrayLike, *, degrees: bool = False
) -> npt.NDArray[np.float64]:
    """Angle subtended at the shot location by the two goal posts.

    This is the true goal-mouth angle from the posts at (120, 36) and (120, 44),
    not a proxy. See the module docstring for the derivation and edge cases.

    Args:
        x: Shot x coordinate(s).
        y: Shot y coordinate(s).
        degrees: Return degrees instead of radians.

    Returns:
        Angle in [0, pi] (or [0, 180] in degrees). NaN wherever an input is NaN.
    """
    x_array = _as_float_array(x)
    y_array = _as_float_array(y)

    # Both posts share the same x, so the cross product collapses to the goal
    # width times the distance to the goal line.
    to_goal_line = GOAL_X - x_array
    cross = GOAL_WIDTH * np.abs(to_goal_line)
    dot = to_goal_line**2 + (y_array - GOAL_POST_LEFT_Y) * (y_array - GOAL_POST_RIGHT_Y)

    angle = np.arctan2(cross, dot)
    return np.degrees(angle) if degrees else angle


def is_valid_location(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    """Whether coordinates are finite and inside the 120 x 80 pitch."""
    x_array = _as_float_array(x)
    y_array = _as_float_array(y)
    return (
        np.isfinite(x_array)
        & np.isfinite(y_array)
        & (x_array >= 0.0)
        & (x_array <= PITCH_LENGTH)
        & (y_array >= 0.0)
        & (y_array <= PITCH_WIDTH)
    )


# --------------------------------------------------------------------------- #
# Dataset construction
# --------------------------------------------------------------------------- #


def build_shot_features(
    shots: pd.DataFrame, matches: pd.DataFrame, *, include_shootouts: bool = False
) -> pd.DataFrame:
    """Build the canonical one-row-per-shot modelling dataset.

    Args:
        shots: The processed ``shots`` table.
        matches: The processed ``matches`` table, joined for competition, season,
            date and collection-fidelity context.
        include_shootouts: Keep period-5 penalty shootouts. They are excluded by
            default: they are a different data-generating process, convert at
            roughly six times the in-play rate and are not part of the scoreline.

    Returns:
        A frame with :data:`DATASET_COLUMNS`, indexed 0..n-1.
    """
    missing = {"x", "y", "is_goal", "match_id", "player_id"} - set(shots.columns)
    if missing:
        raise ShotFeatureError(f"shots table is missing required columns {sorted(missing)}.")

    selected = shots if include_shootouts else shots[~shots["is_shootout"]]
    if selected.empty:
        raise ShotFeatureError("No shots left to build a dataset from.")

    match_context = matches[
        [
            "match_id",
            "competition_id",
            "season_id",
            "match_date",
            "stage",
            "shot_fidelity_version",
        ]
    ]
    merged = selected.merge(match_context, on="match_id", how="left", validate="many_to_one")
    if merged["competition_id"].isna().any():
        orphans = int(merged["competition_id"].isna().sum())
        raise ShotFeatureError(f"{orphans} shot(s) reference a match missing from the match table.")

    frame = merged.rename(columns={"event_id": "shot_id", "is_goal": TARGET_COLUMN})

    valid = is_valid_location(frame["x"], frame["y"])
    frame["has_valid_location"] = valid
    invalid_count = int((~valid).sum())
    if invalid_count:
        logger.warning(
            "%d shot(s) have missing or off-pitch coordinates; their geometry is NaN.",
            invalid_count,
        )

    # Geometry is computed only where the location is usable. Clipping an
    # off-pitch coordinate onto the pitch would invent a location.
    usable_x = frame["x"].where(valid)
    usable_y = frame["y"].where(valid)
    frame["shot_distance"] = shot_distance(usable_x, usable_y)
    frame["shot_angle"] = shot_angle(usable_x, usable_y)

    # StatsBomb tags high-precision collection per match; T03 found this is
    # perfectly confounded with competition in the current subset.
    frame["high_fidelity_coordinates"] = frame["shot_fidelity_version"].notna()

    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(bool)
    return frame[list(DATASET_COLUMNS)].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Sanity checks
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SanityReport:
    """Descriptive checks on a built shot dataset."""

    shots: int
    goals: int
    goal_rate: float
    distance_min: float
    distance_max: float
    distance_mean: float
    angle_min_degrees: float
    angle_max_degrees: float
    angle_mean_degrees: float
    invalid_locations: int
    missing_by_column: dict[str, int] = field(default_factory=dict)
    problems: tuple[str, ...] = ()

    def __str__(self) -> str:
        lines = [
            f"shots            {self.shots:,}",
            f"goals            {self.goals:,}",
            f"goal rate        {self.goal_rate:.4f}",
            f"distance (yd)    min {self.distance_min:.2f}  "
            f"mean {self.distance_mean:.2f}  max {self.distance_max:.2f}",
            f"angle (degrees)  min {self.angle_min_degrees:.2f}  "
            f"mean {self.angle_mean_degrees:.2f}  max {self.angle_max_degrees:.2f}",
            f"invalid location {self.invalid_locations}",
        ]
        if self.missing_by_column:
            lines.append("missing values:")
            lines.extend(
                f"  {column:<24} {count:,}" for column, count in self.missing_by_column.items()
            )
        else:
            lines.append("missing values:  none")
        lines.append(
            "problems:        none"
            if not self.problems
            else "problems:\n" + "\n".join(f"  - {problem}" for problem in self.problems)
        )
        return "\n".join(lines)


def check_shot_dataset(frame: pd.DataFrame) -> SanityReport:
    """Compute descriptive checks and flag impossible values.

    Anything listed in ``problems`` means the dataset is wrong, not merely
    surprising: an out-of-range angle or a duplicated shot identifier cannot
    happen if the pipeline is correct.
    """
    problems: list[str] = []

    if TARGET_COLUMN not in frame.columns:
        raise ShotFeatureError(f"Dataset has no {TARGET_COLUMN!r} column.")

    goals = int(frame[TARGET_COLUMN].sum())
    goal_rate = float(frame[TARGET_COLUMN].mean()) if len(frame) else float("nan")

    distance = frame["shot_distance"]
    angle_degrees = np.degrees(frame["shot_angle"])

    if not frame["shot_id"].is_unique:
        duplicates = int(frame["shot_id"].duplicated().sum())
        problems.append(f"{duplicates} duplicate shot_id value(s)")
    for column in ("match_id", "player_id", "team_id", TARGET_COLUMN):
        if frame[column].isna().any():
            problems.append(f"{column} contains null values")
    if not 0.0 <= goal_rate <= 1.0:
        problems.append(f"goal rate {goal_rate} outside [0, 1]")
    if (distance < 0).any():
        problems.append("negative shot_distance")
    if (distance > MAX_POSSIBLE_DISTANCE).any():
        problems.append(f"shot_distance exceeding the pitch diagonal {MAX_POSSIBLE_DISTANCE:.2f}")
    if ((angle_degrees < 0) | (angle_degrees > 180)).any():
        problems.append("shot_angle outside [0, 180] degrees")

    # Geometry must be present exactly where the location is usable.
    inconsistent = int((frame["has_valid_location"] & distance.isna()).sum())
    if inconsistent:
        problems.append(f"{inconsistent} row(s) with a valid location but no geometry")

    missing = frame.isna().sum()
    missing_by_column = {
        str(column): int(count) for column, count in missing.items() if int(count) > 0
    }

    return SanityReport(
        shots=len(frame),
        goals=goals,
        goal_rate=goal_rate,
        distance_min=float(distance.min()),
        distance_max=float(distance.max()),
        distance_mean=float(distance.mean()),
        angle_min_degrees=float(angle_degrees.min()),
        angle_max_degrees=float(angle_degrees.max()),
        angle_mean_degrees=float(angle_degrees.mean()),
        invalid_locations=int((~frame["has_valid_location"]).sum()),
        missing_by_column=missing_by_column,
        problems=tuple(problems),
    )


# --------------------------------------------------------------------------- #
# Parquet IO
# --------------------------------------------------------------------------- #


def write_shot_dataset(frame: pd.DataFrame, path: Path = DEFAULT_DATASET_PATH) -> Path:
    """Write the canonical dataset, replacing any previous build."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, engine="pyarrow", compression=storage.PARQUET_COMPRESSION, index=False)
    logger.info("wrote %s (%d rows, %.2f MB)", path, len(frame), path.stat().st_size / 1_000_000)
    return path


def read_shot_dataset(path: Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """Read the canonical dataset."""
    if not path.exists():
        raise ShotFeatureError(f"{path} does not exist. Build it first with `make features`.")
    return pd.read_parquet(path)


def build_shot_dataset(
    processed_root: Path = storage.DEFAULT_PROCESSED_ROOT,
    path: Path = DEFAULT_DATASET_PATH,
    *,
    include_shootouts: bool = False,
) -> tuple[pd.DataFrame, SanityReport]:
    """Build the canonical dataset from the processed tables and persist it."""
    shots = storage.read_table("shots", processed_root)
    matches = storage.read_table("matches", processed_root)
    frame = build_shot_features(shots, matches, include_shootouts=include_shootouts)
    report = check_shot_dataset(frame)
    write_shot_dataset(frame, path)
    return frame, report


# --------------------------------------------------------------------------- #
# Command line entry point
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        prog="python -m football_intelligence.features.shots",
        description="Build the canonical shot modelling dataset.",
    )
    parser.add_argument("--processed-root", type=Path, default=storage.DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument(
        "--include-shootouts",
        action="store_true",
        help="Keep period-5 shootout penalties (excluded by default).",
    )
    arguments = parser.parse_args(argv)

    try:
        _, report = build_shot_dataset(
            arguments.processed_root,
            arguments.output,
            include_shootouts=arguments.include_shootouts,
        )
    except (ShotFeatureError, storage.StorageError) as error:
        logger.error("%s", error)
        return 1

    print(report)
    if report.problems:
        logger.error("dataset failed %d sanity check(s)", len(report.problems))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
