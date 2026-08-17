"""Tests for the DuckDB analytical queries, run against fixture-built Parquet."""

from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest

from football_intelligence.data import queries, statsbomb, storage
from football_intelligence.data.storage import StorageError


@pytest.fixture
def connection(raw_root: Path, tmp_path: Path) -> Iterator[duckdb.DuckDBPyConnection]:
    """A DuckDB connection over Parquet tables built from the fixture matches."""
    processed_root = tmp_path / "processed"
    storage.build_processed_tables([statsbomb.CompetitionSeason(900, 1)], raw_root, processed_root)
    with queries.connect(processed_root) as open_connection:
        yield open_connection


def test_connect_requires_the_tables_to_exist(tmp_path: Path) -> None:
    with pytest.raises(StorageError, match="make process"):
        queries.connect(tmp_path / "never-built")


def test_views_expose_every_table(connection: duckdb.DuckDBPyConnection) -> None:
    for name in storage.TABLE_NAMES:
        rows = connection.execute(f"SELECT count(*) FROM {name}").fetchone()
        assert rows is not None
        assert rows[0] > 0


def test_match_counts_by_competition(connection: duckdb.DuckDBPyConnection) -> None:
    result = queries.match_counts_by_competition(connection)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["competition_id"] == 900
    assert row["season_id"] == 1
    assert row["competition_name"] == "Fixture Cup"
    assert row["matches"] == 2
    assert row["goals"] == 3  # 2-1 and 0-0
    assert row["goals_per_match"] == pytest.approx(1.5)


def test_shots_by_player_excludes_shootouts_by_default(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    result = queries.shots_by_player(connection).set_index("player_id")

    # Four shots exist, but player 101's second is a shootout penalty.
    assert result["shots"].sum() == 3
    assert result.loc[101, "shots"] == 1
    assert result.loc[202, "shots"] == 1
    assert result.loc[303, "shots"] == 1
    assert 102 not in result.index  # never took a shot
    assert result.loc[101, "player"] == "Player One"  # display name from players


def test_shots_by_player_can_include_shootouts(connection: duckdb.DuckDBPyConnection) -> None:
    result = queries.shots_by_player(connection, include_shootouts=True).set_index("player_id")

    assert result["shots"].sum() == 4
    assert result.loc[101, "shots"] == 2


def test_shots_by_player_honours_min_shots_and_limit(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    assert len(queries.shots_by_player(connection, min_shots=2)) == 0
    assert len(queries.shots_by_player(connection, limit=1)) == 1


def test_goals_by_player(connection: duckdb.DuckDBPyConnection) -> None:
    result = queries.goals_by_player(connection).set_index("player_id")

    # Two in-play goals: player 101 in match 5001, player 303 in match 5002.
    assert set(result.index) == {101, 303}
    assert result.loc[101, "goals"] == 1
    assert result.loc[303, "goals"] == 1
    assert 202 not in result.index  # took a shot, scored nothing


def test_goals_by_player_counts_shootout_goals_only_when_asked(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    included = queries.goals_by_player(connection, include_shootouts=True).set_index("player_id")

    assert included.loc[101, "goals"] == 2  # in-play goal plus the shootout penalty


def test_shots_by_team(connection: duckdb.DuckDBPyConnection) -> None:
    result = queries.shots_by_team(connection).set_index("team")

    assert set(result.index) == {"Alpha FC", "Beta FC", "Gamma FC"}
    assert result.loc["Alpha FC", "shots"] == 1  # the shootout penalty is excluded
    assert result.loc["Alpha FC", "goals"] == 1
    assert result.loc["Beta FC", "shots"] == 1
    assert result.loc["Beta FC", "goals"] == 0
    assert result.loc["Gamma FC", "matches"] == 1

    # Matches played come from the match table, not from the shots: Alpha FC
    # appears in both fixture matches but has only one non-shootout shot.
    assert result.loc["Alpha FC", "matches"] == 2
    assert result.loc["Alpha FC", "shots_per_match"] == pytest.approx(0.5)


def test_player_conversion_rate_is_null_when_there_are_no_shots(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    result = queries.player_conversion_rates(connection).set_index("player_id")

    # Every player appears, including those who never shot.
    assert set(result.index) == {101, 102, 202, 303}

    # No attempts: the rate is undefined, not zero.
    undefined = result["conversion_rate"].isna()
    assert result.loc[102, "shots"] == 0
    assert bool(undefined.loc[102])
    assert not bool(undefined.loc[202])

    # Attempts but no goals is a genuine zero, and must stay distinguishable.
    assert result.loc[202, "shots"] == 1
    assert result.loc[202, "conversion_rate"] == pytest.approx(0.0)

    assert result.loc[101, "conversion_rate"] == pytest.approx(1.0)


def test_player_conversion_rates_can_require_a_minimum_of_shots(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    result = queries.player_conversion_rates(connection, min_shots=1)

    assert set(result["player_id"]) == {101, 202, 303}


def test_team_conversion_rates_guard_the_denominator(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    result = queries.team_conversion_rates(connection).set_index("team")

    assert result.loc["Alpha FC", "conversion_rate"] == pytest.approx(1.0)
    assert result.loc["Beta FC", "conversion_rate"] == pytest.approx(0.0)
    assert (result["shots"] > 0).all()
