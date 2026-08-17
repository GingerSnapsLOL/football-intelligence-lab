"""Tests for the raw JSON to Parquet normalisation layer.

Everything runs on the tiny fixture tree from conftest.py; no network, no real
StatsBomb data.
"""

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from football_intelligence.data import statsbomb, storage
from football_intelligence.data.statsbomb import JSONObject
from football_intelligence.data.storage import StorageError

FIXTURES = Path(__file__).parent / "fixtures" / "statsbomb"


def _fixture_json(name: str) -> list[JSONObject]:
    payload = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    return payload


# --------------------------------------------------------------------------- #
# Normalisation: matches
# --------------------------------------------------------------------------- #


def test_normalize_match_flattens_nested_objects() -> None:
    record = _fixture_json("matches.json")[0]

    row = storage.normalize_match(record)

    assert row["match_id"] == 5001
    assert row["competition_id"] == 900
    assert row["season_id"] == 1
    assert row["competition_name"] == "Fixture Cup"
    assert row["home_team"] == "Alpha FC"
    assert row["home_team_id"] == 10
    assert row["stage"] == "Group Stage"
    assert row["referee"] == "A. Referee"
    assert row["total_goals"] == 3


def test_normalize_match_tolerates_missing_optional_fields() -> None:
    # The second fixture match has no referee, stadium or kick-off time.
    row = storage.normalize_match(_fixture_json("matches.json")[1])

    assert row["match_id"] == 5002
    assert row["referee"] is None
    assert row["stadium"] is None
    assert row["kick_off"] is None
    assert row["total_goals"] == 0


def test_normalize_match_reports_a_missing_required_field() -> None:
    with pytest.raises(StorageError, match="missing required field 'match_id'"):
        storage.normalize_match({"home_score": 1, "away_score": 0})


# --------------------------------------------------------------------------- #
# Normalisation: events
# --------------------------------------------------------------------------- #


def test_normalize_event_keeps_only_common_attributes() -> None:
    pass_event = _fixture_json("events.json")[1]

    row = storage.normalize_event(pass_event, 5001)

    assert row["event_id"] == "00000000-0000-0000-0000-000000000002"
    assert row["match_id"] == 5001
    assert row["type"] == "Pass"
    assert row["type_id"] == 30
    assert row["player_id"] == 101
    assert (row["x"], row["y"]) == (60.0, 40.0)
    # The type-specific payload is deliberately not flattened into events.
    assert "pass" not in row
    assert set(row) == set(storage.EVENT_DTYPES)


def test_normalize_event_treats_absent_booleans_as_false() -> None:
    shot_event = _fixture_json("events.json")[2]
    plain_event = _fixture_json("events.json")[1]

    assert storage.normalize_event(shot_event, 5001)["under_pressure"] is True
    assert storage.normalize_event(plain_event, 5001)["under_pressure"] is False
    assert storage.normalize_event(plain_event, 5001)["off_camera"] is False


def test_normalize_event_handles_events_without_location_or_type() -> None:
    # The last fixture event has neither a type block nor a location.
    row = storage.normalize_event(_fixture_json("events.json")[4], 5001)

    assert row["type"] is None
    assert row["x"] is None
    assert row["y"] is None
    assert row["player_id"] is None


# --------------------------------------------------------------------------- #
# Normalisation: shots
# --------------------------------------------------------------------------- #


def test_normalize_shot_flattens_the_shot_payload() -> None:
    shot = statsbomb.events_of_type(_fixture_json("events.json"), "Shot")[0]

    row = storage.normalize_shot(shot, 5001)

    assert row["event_id"] == "00000000-0000-0000-0000-000000000003"
    assert row["player_id"] == 101
    assert row["team_id"] == 10
    assert row["outcome"] == "Goal"
    assert row["is_goal"] is True
    assert row["shot_type"] == "Open Play"
    assert row["body_part"] == "Right Foot"
    assert row["statsbomb_xg"] == pytest.approx(0.42)
    assert (row["x"], row["y"]) == (110.0, 40.0)
    assert (row["end_x"], row["end_y"], row["end_z"]) == (120.0, 40.0, 0.5)
    assert row["has_freeze_frame"] is True
    assert row["first_time"] is True
    assert row["is_shootout"] is False


def test_normalize_shot_defaults_absent_flags_to_false() -> None:
    sparse = statsbomb.events_of_type(_fixture_json("events.json"), "Shot")[1]

    row = storage.normalize_shot(sparse, 5001)

    assert row["is_goal"] is False
    assert row["technique"] is None
    assert row["end_z"] is None  # two-element end_location: the shot stayed low
    assert row["under_pressure"] is False
    assert all(row[flag] is False for flag in storage.SHOT_FLAGS)


def test_normalize_shot_marks_period_five_as_a_shootout() -> None:
    shootout = statsbomb.events_of_type(_fixture_json("events_5002.json"), "Shot")[1]

    row = storage.normalize_shot(shootout, 5002)

    assert row["period"] == 5
    assert row["is_shootout"] is True
    assert row["shot_type"] == "Penalty"
    assert row["is_goal"] is True


def test_normalize_shot_rejects_a_shot_without_a_player() -> None:
    shot = statsbomb.events_of_type(_fixture_json("events.json"), "Shot")[0]
    del shot["player"]

    with pytest.raises(StorageError, match=r"missing required field 'player\.id'"):
        storage.normalize_shot(shot, 5001)


# --------------------------------------------------------------------------- #
# Table assembly
# --------------------------------------------------------------------------- #


def test_build_tables_produces_the_expected_row_counts(raw_root: Path) -> None:
    tables = storage.build_tables([statsbomb.CompetitionSeason(900, 1)], raw_root)

    assert set(tables) == set(storage.TABLE_NAMES)
    assert len(tables["matches"]) == 2  # 5001 and 5002
    assert len(tables["events"]) == 9  # 5 + 4
    assert len(tables["shots"]) == 4  # 2 per match
    assert len(tables["players"]) == 4  # 101, 102, 202, 303
    assert int(tables["shots"]["is_goal"].sum()) == 3
    assert int(tables["shots"]["is_shootout"].sum()) == 1


def test_identifiers_are_preserved_as_integers(raw_root: Path) -> None:
    tables = storage.build_tables([statsbomb.CompetitionSeason(900, 1)], raw_root)

    assert tables["matches"]["match_id"].dtype == "int64"
    assert tables["matches"]["competition_id"].dtype == "int64"
    assert tables["shots"]["match_id"].dtype == "int64"
    assert tables["shots"]["player_id"].dtype == "int64"
    assert tables["players"]["player_id"].dtype == "int64"
    # Nullable identifiers stay integral rather than degrading to float.
    assert tables["events"]["player_id"].dtype == "Int64"
    assert sorted(tables["players"]["player_id"].tolist()) == [101, 102, 202, 303]
    assert sorted(tables["matches"]["match_id"].tolist()) == [5001, 5002]


def test_shots_are_a_subset_of_events(raw_root: Path) -> None:
    tables = storage.build_tables([statsbomb.CompetitionSeason(900, 1)], raw_root)

    assert set(tables["shots"]["event_id"]) <= set(tables["events"]["event_id"])
    assert set(tables["shots"]["match_id"]) <= set(tables["matches"]["match_id"])


def test_players_aggregate_squad_listings_across_matches(raw_root: Path) -> None:
    tables = storage.build_tables([statsbomb.CompetitionSeason(900, 1)], raw_root)
    players = tables["players"].set_index("player_id")

    # Player 101 is listed in both fixture matches.
    assert players.loc[101, "squad_listings"] == 2
    assert players.loc[101, "display_name"] == "Player One"  # nickname preferred
    assert players.loc[202, "squad_listings"] == 1
    # Listed twice but only once with a recorded position: squad listings are not
    # appearances.
    assert players.loc[102, "squad_listings"] == 2
    assert players.loc[102, "matches_with_position"] == 1
    assert players.loc[102, "display_name"] == "Player Reserve"  # nickname is null


def test_build_shots_returns_a_typed_empty_frame_when_a_match_has_no_shots() -> None:
    events = [event for event in _fixture_json("events.json") if "shot" not in event]

    shots = storage.build_shots(events, 5001)

    assert len(shots) == 0
    assert list(shots.columns) == list(storage.SHOT_DTYPES)


# --------------------------------------------------------------------------- #
# Duplicates
# --------------------------------------------------------------------------- #


def test_duplicate_events_are_dropped_and_reported(caplog: pytest.LogCaptureFixture) -> None:
    events = _fixture_json("events.json")
    duplicated = [*events, events[0]]

    with caplog.at_level(logging.WARNING):
        frame = storage._deduplicate(storage.build_events(duplicated, 5001), "event_id", "events")

    assert len(frame) == len(events)
    assert frame["event_id"].is_unique
    assert "dropped 1 duplicate row(s) on event_id" in caplog.text


def test_duplicate_matches_are_dropped(caplog: pytest.LogCaptureFixture) -> None:
    records = _fixture_json("matches.json")

    with caplog.at_level(logging.WARNING):
        matches = storage.build_matches([*records, records[0]])

    assert len(matches) == 2
    assert matches["match_id"].is_unique
    assert "duplicate row(s) on match_id" in caplog.text


# --------------------------------------------------------------------------- #
# Parquet round trip
# --------------------------------------------------------------------------- #


def test_parquet_round_trip_preserves_values_and_identifier_types(
    raw_root: Path, tmp_path: Path
) -> None:
    processed_root = tmp_path / "processed"
    tables = storage.build_tables([statsbomb.CompetitionSeason(900, 1)], raw_root)

    for name in storage.TABLE_NAMES:
        summary = storage.write_table(tables[name], name, processed_root)
        assert summary.rows == len(tables[name])
        assert summary.bytes_written > 0

        restored = storage.read_table(name, processed_root)
        assert list(restored.columns) == list(tables[name].columns)
        assert len(restored) == len(tables[name])
        pd.testing.assert_frame_equal(
            restored, tables[name], check_dtype=False, check_index_type=False
        )

    restored_shots = storage.read_table("shots", processed_root)
    assert restored_shots["match_id"].dtype == "int64"
    assert restored_shots["player_id"].dtype == "int64"
    assert restored_shots["is_goal"].dtype == "bool"
    assert restored_shots["statsbomb_xg"].dtype == "float64"
    assert pd.api.types.is_string_dtype(restored_shots["outcome"])


def test_build_processed_tables_writes_every_table(raw_root: Path, tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"

    summaries = storage.build_processed_tables(
        [statsbomb.CompetitionSeason(900, 1)], raw_root, processed_root
    )

    assert set(summaries) == set(storage.TABLE_NAMES)
    for name, summary in summaries.items():
        assert summary.path == processed_root / f"{name}.parquet"
        assert summary.path.exists()
        assert summary.bytes_written == summary.path.stat().st_size


def test_rebuilding_replaces_the_previous_parquet(raw_root: Path, tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    selection = [statsbomb.CompetitionSeason(900, 1)]

    storage.build_processed_tables(selection, raw_root, processed_root)
    first = storage.read_table("shots", processed_root)
    storage.build_processed_tables(selection, raw_root, processed_root)
    second = storage.read_table("shots", processed_root)

    # The build is reproducible: rerunning must not append or reorder rows.
    pd.testing.assert_frame_equal(first, second)


# --------------------------------------------------------------------------- #
# Paths and errors
# --------------------------------------------------------------------------- #


def test_table_path_rejects_an_unknown_table(tmp_path: Path) -> None:
    with pytest.raises(StorageError, match="Unknown table 'lineups'"):
        storage.table_path("lineups", tmp_path)


def test_reading_a_table_that_was_never_built_points_at_the_build_command(
    tmp_path: Path,
) -> None:
    with pytest.raises(StorageError, match="make process"):
        storage.read_table("shots", tmp_path)


def test_building_without_matches_fails_clearly() -> None:
    with pytest.raises(StorageError, match="No match records"):
        storage.build_matches([])
