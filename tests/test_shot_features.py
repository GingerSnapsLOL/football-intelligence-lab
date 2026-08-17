"""Tests for shot geometry and the canonical shot dataset.

The geometry cases are chosen so the expected answer can be checked by hand from
the StatsBomb pitch: the goal spans (120, 36) to (120, 44), so a shot on the
centre line at distance ``d`` from the goal line sees the mouth at an angle of
``2 * arctan(4 / d)``.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from football_intelligence.data import statsbomb, storage
from football_intelligence.features import shots as shot_features
from football_intelligence.features.shots import (
    ShotFeatureError,
    is_valid_location,
    shot_angle,
    shot_distance,
)


def distance(x: float, y: float) -> float:
    return float(shot_distance(x, y))


def angle_degrees(x: float, y: float) -> float:
    return float(shot_angle(x, y, degrees=True))


# --------------------------------------------------------------------------- #
# Distance
# --------------------------------------------------------------------------- #


def test_distance_from_the_penalty_spot_is_twelve_yards() -> None:
    # The Laws of the Game put the penalty mark 12 yards out, and StatsBomb
    # records penalties at (108, 40). This is what fixes the unit as the yard.
    assert distance(108.0, 40.0) == pytest.approx(12.0)


def test_distance_on_the_goal_line_at_the_centre_is_zero() -> None:
    assert distance(120.0, 40.0) == pytest.approx(0.0)


def test_distance_at_a_post_is_half_the_goal_width() -> None:
    assert distance(120.0, 36.0) == pytest.approx(4.0)
    assert distance(120.0, 44.0) == pytest.approx(4.0)


def test_distance_is_a_plain_right_triangle() -> None:
    # 3-4-5 triangle: 3 yards out from the goal line, 4 yards off centre.
    assert distance(117.0, 44.0) == pytest.approx(5.0)


def test_distance_from_the_halfway_line_is_sixty_yards() -> None:
    assert distance(60.0, 40.0) == pytest.approx(60.0)


def test_distance_is_symmetric_about_the_goal_centre_line() -> None:
    assert distance(100.0, 30.0) == pytest.approx(distance(100.0, 50.0))


def test_distance_grows_as_the_shot_moves_away() -> None:
    values = [distance(x, 40.0) for x in (118.0, 110.0, 100.0, 80.0)]
    assert values == sorted(values)


# --------------------------------------------------------------------------- #
# Angle
# --------------------------------------------------------------------------- #


def test_angle_from_the_penalty_spot() -> None:
    # From (108, 40) each post is 12 yards ahead and 4 to the side, so the half
    # angle is arctan(4/12) and the full angle twice that.
    expected = 2 * math.degrees(math.atan(4.0 / 12.0))
    assert expected == pytest.approx(36.8699, abs=1e-4)
    assert angle_degrees(108.0, 40.0) == pytest.approx(expected)


def test_angle_ten_yards_out_and_central() -> None:
    expected = 2 * math.degrees(math.atan(4.0 / 10.0))
    assert expected == pytest.approx(43.6028, abs=1e-4)
    assert angle_degrees(110.0, 40.0) == pytest.approx(expected)


def test_angle_from_the_halfway_line_is_small() -> None:
    expected = 2 * math.degrees(math.atan(4.0 / 60.0))
    assert expected == pytest.approx(7.6281, abs=1e-4)
    assert angle_degrees(60.0, 40.0) == pytest.approx(expected)


def test_angle_on_the_goal_line_between_the_posts_is_a_straight_line() -> None:
    # Standing on the line between the posts, one post is to each side.
    assert angle_degrees(120.0, 40.0) == pytest.approx(180.0)
    assert angle_degrees(120.0, 37.0) == pytest.approx(180.0)


def test_angle_on_the_goal_line_outside_the_posts_is_zero() -> None:
    # Both posts lie in the same direction along the line, so nothing is visible.
    assert angle_degrees(120.0, 50.0) == pytest.approx(0.0)
    assert angle_degrees(120.0, 20.0) == pytest.approx(0.0)


def test_angle_exactly_at_a_post_is_degenerate_and_returns_zero() -> None:
    # Both the cross and dot products vanish; atan2(0, 0) is 0 by convention.
    assert angle_degrees(120.0, 36.0) == pytest.approx(0.0)
    assert angle_degrees(120.0, 44.0) == pytest.approx(0.0)


def test_angle_exceeds_a_right_angle_only_inside_the_circle_through_the_posts() -> None:
    # On the centre line the angle passes 90 degrees when the distance to the
    # goal line drops below the half goal width.
    assert angle_degrees(120.0 - 4.0, 40.0) == pytest.approx(90.0)
    assert angle_degrees(120.0 - 3.0, 40.0) > 90.0
    assert angle_degrees(120.0 - 5.0, 40.0) < 90.0


def test_angle_is_symmetric_about_the_goal_centre_line() -> None:
    assert angle_degrees(100.0, 32.0) == pytest.approx(angle_degrees(100.0, 48.0))
    assert angle_degrees(110.0, 39.0) == pytest.approx(angle_degrees(110.0, 41.0))


def test_angle_shrinks_with_distance_along_the_centre_line() -> None:
    values = [angle_degrees(x, 40.0) for x in (114.0, 108.0, 96.0, 60.0)]
    assert values == sorted(values, reverse=True)


def test_a_wide_shot_sees_less_goal_than_a_central_one_at_the_same_distance() -> None:
    central = angle_degrees(108.0, 40.0)
    wide = angle_degrees(108.0, 10.0)
    assert wide < central
    # 30 yards off centre from 12 yards out is a very tight angle.
    assert wide < 6.0


def test_angle_matches_an_independent_arccos_implementation() -> None:
    """Cross-check atan2(cross, dot) against the law-of-cosines formulation."""
    rng = np.random.default_rng(20240816)
    xs = rng.uniform(60.0, 119.0, size=200)
    ys = rng.uniform(1.0, 79.0, size=200)

    for x, y in zip(xs, ys, strict=True):
        to_left = np.array([120.0 - x, 36.0 - y])
        to_right = np.array([120.0 - x, 44.0 - y])
        cosine = to_left.dot(to_right) / (np.linalg.norm(to_left) * np.linalg.norm(to_right))
        expected = math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))
        assert angle_degrees(float(x), float(y)) == pytest.approx(expected, abs=1e-8)


def test_angle_stays_within_zero_and_pi_across_the_whole_pitch() -> None:
    grid_x, grid_y = np.meshgrid(np.linspace(0.0, 120.0, 121), np.linspace(0.0, 80.0, 81))
    angles = shot_angle(grid_x, grid_y)

    assert np.all(angles >= 0.0)
    assert np.all(angles <= math.pi + 1e-12)
    assert np.all(np.isfinite(angles))


# --------------------------------------------------------------------------- #
# Missing and malformed coordinates
# --------------------------------------------------------------------------- #


def test_missing_coordinates_propagate_as_nan() -> None:
    assert math.isnan(distance(float("nan"), 40.0))
    assert math.isnan(distance(100.0, float("nan")))
    assert math.isnan(angle_degrees(float("nan"), 40.0))
    assert math.isnan(angle_degrees(100.0, float("nan")))


def test_geometry_is_vectorised_over_arrays() -> None:
    xs = [108.0, 110.0, float("nan")]
    ys = [40.0, 40.0, 40.0]

    distances = shot_distance(xs, ys)
    angles = shot_angle(xs, ys, degrees=True)

    assert distances[0] == pytest.approx(12.0)
    assert distances[1] == pytest.approx(10.0)
    assert math.isnan(float(distances[2]))
    assert angles[0] == pytest.approx(36.8699, abs=1e-4)
    assert math.isnan(float(angles[2]))


def test_is_valid_location_rejects_off_pitch_and_non_finite_coordinates() -> None:
    xs = [60.0, -1.0, 121.0, 60.0, 60.0, float("nan"), float("inf")]
    ys = [40.0, 40.0, 40.0, -0.5, 80.5, 40.0, 40.0]

    valid = is_valid_location(xs, ys)

    assert valid.tolist() == [True, False, False, False, False, False, False]


def test_is_valid_location_accepts_the_pitch_boundary() -> None:
    assert is_valid_location([0.0, 120.0, 60.0, 60.0], [40.0, 40.0, 0.0, 80.0]).all()


# --------------------------------------------------------------------------- #
# Dataset construction
# --------------------------------------------------------------------------- #


@pytest.fixture
def processed_tables(raw_root: Path) -> dict[str, pd.DataFrame]:
    return storage.build_tables([statsbomb.CompetitionSeason(900, 1)], raw_root)


def test_dataset_excludes_shootouts_by_default(
    processed_tables: dict[str, pd.DataFrame],
) -> None:
    dataset = shot_features.build_shot_features(
        processed_tables["shots"], processed_tables["matches"]
    )

    # The fixtures hold 4 shots, one of which is a period-5 shootout penalty.
    assert len(dataset) == 3
    assert dataset["goal"].sum() == 2
    assert dataset["period"].max() < 5


def test_dataset_can_include_shootouts(processed_tables: dict[str, pd.DataFrame]) -> None:
    dataset = shot_features.build_shot_features(
        processed_tables["shots"], processed_tables["matches"], include_shootouts=True
    )

    assert len(dataset) == 4
    assert dataset["goal"].sum() == 3


def test_dataset_preserves_the_identifiers_needed_for_grouped_analysis(
    processed_tables: dict[str, pd.DataFrame],
) -> None:
    dataset = shot_features.build_shot_features(
        processed_tables["shots"], processed_tables["matches"]
    )

    for column in shot_features.IDENTIFIER_COLUMNS:
        assert column in dataset.columns
        assert dataset[column].notna().all()

    assert set(dataset["match_id"]) == {5001, 5002}
    assert set(dataset["competition_id"]) == {900}
    assert set(dataset["season_id"]) == {1}
    assert dataset["match_id"].dtype == "int64"
    assert dataset["player_id"].dtype == "int64"
    assert dataset["team_id"].dtype == "int64"
    assert dataset["shot_id"].is_unique


def test_dataset_target_is_boolean(processed_tables: dict[str, pd.DataFrame]) -> None:
    dataset = shot_features.build_shot_features(
        processed_tables["shots"], processed_tables["matches"]
    )

    assert dataset["goal"].dtype == "bool"


def test_dataset_geometry_matches_the_standalone_functions(
    processed_tables: dict[str, pd.DataFrame],
) -> None:
    dataset = shot_features.build_shot_features(
        processed_tables["shots"], processed_tables["matches"]
    )
    row = dataset[dataset["shot_id"] == "00000000-0000-0000-0000-000000000003"].iloc[0]

    # Fixture shot at (110, 40): 10 yards out, dead central.
    assert row["shot_distance"] == pytest.approx(10.0)
    assert math.degrees(row["shot_angle"]) == pytest.approx(43.6028, abs=1e-4)


def test_dataset_columns_are_exactly_the_declared_schema(
    processed_tables: dict[str, pd.DataFrame],
) -> None:
    dataset = shot_features.build_shot_features(
        processed_tables["shots"], processed_tables["matches"]
    )

    assert list(dataset.columns) == list(shot_features.DATASET_COLUMNS)
    # The benchmark column must never be part of the modelling feature set.
    assert "statsbomb_xg" in dataset.columns
    assert "statsbomb_xg" not in shot_features.FEATURE_COLUMNS
    assert shot_features.TARGET_COLUMN not in shot_features.FEATURE_COLUMNS


def test_invalid_locations_yield_nan_geometry_rather_than_a_clipped_guess(
    processed_tables: dict[str, pd.DataFrame],
) -> None:
    shots = processed_tables["shots"].copy()
    shots.loc[shots.index[0], "x"] = 130.0  # beyond the goal line
    shots.loc[shots.index[1], "y"] = np.nan  # missing coordinate

    dataset = shot_features.build_shot_features(shots, processed_tables["matches"])

    unusable = dataset[~dataset["has_valid_location"]]
    assert len(unusable) == 2
    assert unusable["shot_distance"].isna().all()
    assert unusable["shot_angle"].isna().all()
    # The raw coordinates are kept as recorded, not silently corrected.
    assert 130.0 in set(dataset["x"].dropna())


def test_building_from_shots_without_a_matching_match_fails_loudly(
    processed_tables: dict[str, pd.DataFrame],
) -> None:
    matches = processed_tables["matches"]

    with pytest.raises(ShotFeatureError, match="reference a match missing"):
        shot_features.build_shot_features(processed_tables["shots"], matches.iloc[:0])


def test_building_without_required_columns_fails_loudly(
    processed_tables: dict[str, pd.DataFrame],
) -> None:
    shots = processed_tables["shots"].drop(columns=["x"])

    with pytest.raises(ShotFeatureError, match="missing required columns"):
        shot_features.build_shot_features(shots, processed_tables["matches"])


# --------------------------------------------------------------------------- #
# Sanity checks and IO
# --------------------------------------------------------------------------- #


def test_sanity_report_describes_a_clean_dataset(
    processed_tables: dict[str, pd.DataFrame],
) -> None:
    dataset = shot_features.build_shot_features(
        processed_tables["shots"], processed_tables["matches"]
    )

    report = shot_features.check_shot_dataset(dataset)

    assert report.shots == 3
    assert report.goals == 2
    assert report.goal_rate == pytest.approx(2 / 3)
    assert report.invalid_locations == 0
    assert report.problems == ()
    assert 0.0 <= report.angle_min_degrees <= report.angle_max_degrees <= 180.0
    assert report.distance_min >= 0.0


def test_sanity_report_flags_duplicate_identifiers(
    processed_tables: dict[str, pd.DataFrame],
) -> None:
    dataset = shot_features.build_shot_features(
        processed_tables["shots"], processed_tables["matches"]
    )
    duplicated = pd.concat([dataset, dataset.iloc[:1]], ignore_index=True)

    report = shot_features.check_shot_dataset(duplicated)

    assert any("duplicate shot_id" in problem for problem in report.problems)


def test_sanity_report_flags_impossible_geometry(
    processed_tables: dict[str, pd.DataFrame],
) -> None:
    dataset = shot_features.build_shot_features(
        processed_tables["shots"], processed_tables["matches"]
    )
    dataset.loc[0, "shot_distance"] = -1.0
    dataset.loc[1, "shot_angle"] = 4.0  # radians, above pi

    report = shot_features.check_shot_dataset(dataset)

    assert any("negative shot_distance" in problem for problem in report.problems)
    assert any("outside [0, 180]" in problem for problem in report.problems)


def test_dataset_round_trips_through_parquet(
    processed_tables: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    dataset = shot_features.build_shot_features(
        processed_tables["shots"], processed_tables["matches"]
    )
    path = tmp_path / "shot_dataset.parquet"

    shot_features.write_shot_dataset(dataset, path)
    restored = shot_features.read_shot_dataset(path)

    pd.testing.assert_frame_equal(restored, dataset, check_dtype=False)
    assert restored["match_id"].dtype == "int64"
    assert restored["goal"].dtype == "bool"


def test_reading_a_dataset_that_was_never_built_points_at_the_build_command(
    tmp_path: Path,
) -> None:
    with pytest.raises(ShotFeatureError, match="make features"):
        shot_features.read_shot_dataset(tmp_path / "absent.parquet")


def test_build_shot_dataset_end_to_end(raw_root: Path, tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    storage.build_processed_tables([statsbomb.CompetitionSeason(900, 1)], raw_root, processed_root)
    path = tmp_path / "shot_dataset.parquet"

    dataset, report = shot_features.build_shot_dataset(processed_root, path)

    assert path.exists()
    assert len(dataset) == 3
    assert report.problems == ()
