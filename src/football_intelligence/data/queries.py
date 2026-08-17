"""Analytical SQL over the processed Parquet tables.

DuckDB reads the Parquet files directly through views created in an in-memory
database. No ``.duckdb`` file is persisted: the Parquet files are already the
source of truth, a database file would be a second copy that can silently go
stale, and DuckDB's Parquet scanner is fast enough at this size. If a future task
needs persistent indexes, materialised views or concurrent writers, that is the
point to reconsider.

Every query here excludes penalty shootouts by default (``is_shootout``): they
are a separate data-generating process and are not part of the match score.
"""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import duckdb
import pandas as pd

from football_intelligence.data.storage import (
    DEFAULT_PROCESSED_ROOT,
    TABLE_NAMES,
    StorageError,
    table_path,
)

logger = logging.getLogger(__name__)

# Guards the "shots per player" style queries against long tails of one-shot
# players when a caller wants a readable ranking.
DEFAULT_MIN_SHOTS: Final = 1


def connect(processed_root: Path = DEFAULT_PROCESSED_ROOT) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB with a view over each processed Parquet table.

    Raises:
        StorageError: if a table has not been built yet.
    """
    connection = duckdb.connect(database=":memory:")
    for name in TABLE_NAMES:
        path = table_path(name, processed_root)
        if not path.exists():
            connection.close()
            raise StorageError(f"{path} does not exist. Build it first with `make process`.")
        # The table names come from a module constant and the path from the
        # caller's configuration, so there is no untrusted input here; the quote
        # doubling keeps paths containing apostrophes valid.
        literal = str(path).replace("'", "''")
        connection.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{literal}')")
    return connection


def _shootout_filter(include_shootouts: bool, alias: str = "s") -> str:
    return "TRUE" if include_shootouts else f"NOT {alias}.is_shootout"


def match_counts_by_competition(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Matches, date range and goals per competition season."""
    return connection.execute(
        """
        SELECT
            competition_id,
            season_id,
            competition_name,
            season_name,
            count(*)                   AS matches,
            min(match_date)            AS first_match,
            max(match_date)            AS last_match,
            -- DuckDB widens integer sums to HUGEINT, which pandas would surface
            -- as a float; the cast keeps goal counts integral.
            sum(total_goals)::BIGINT   AS goals,
            round(avg(total_goals), 2) AS goals_per_match
        FROM matches
        GROUP BY competition_id, season_id, competition_name, season_name
        ORDER BY matches DESC, competition_id
        """
    ).df()


def shots_by_player(
    connection: duckdb.DuckDBPyConnection,
    *,
    min_shots: int = DEFAULT_MIN_SHOTS,
    limit: int | None = None,
    include_shootouts: bool = False,
) -> pd.DataFrame:
    """Shot counts per player, joined to the player dimension for stable names."""
    return connection.execute(
        f"""
        SELECT
            p.player_id,
            p.display_name AS player,
            p.team,
            count(*) AS shots,
            round(avg(s.statsbomb_xg), 4) AS mean_statsbomb_xg
        FROM shots s
        JOIN players p USING (player_id)
        WHERE {_shootout_filter(include_shootouts)}
        GROUP BY p.player_id, p.display_name, p.team
        HAVING count(*) >= ?
        ORDER BY shots DESC, player
        {"LIMIT ?" if limit is not None else ""}
        """,
        [min_shots] if limit is None else [min_shots, limit],
    ).df()


def goals_by_player(
    connection: duckdb.DuckDBPyConnection,
    *,
    limit: int | None = None,
    include_shootouts: bool = False,
) -> pd.DataFrame:
    """Goals scored per player, from shot events only.

    Own goals are recorded as their own event type and are therefore absent here,
    which is correct for a *finishing* metric but means these counts do not
    reproduce a team's scoreline.
    """
    return connection.execute(
        f"""
        SELECT
            p.player_id,
            p.display_name AS player,
            p.team,
            count(*) FILTER (WHERE s.is_goal) AS goals,
            count(*) AS shots
        FROM shots s
        JOIN players p USING (player_id)
        WHERE {_shootout_filter(include_shootouts)}
        GROUP BY p.player_id, p.display_name, p.team
        HAVING count(*) FILTER (WHERE s.is_goal) > 0
        ORDER BY goals DESC, shots, player
        {"LIMIT ?" if limit is not None else ""}
        """,
        [] if limit is None else [limit],
    ).df()


def shots_by_team(
    connection: duckdb.DuckDBPyConnection, *, include_shootouts: bool = False
) -> pd.DataFrame:
    """Shots, goals and expected goals per team.

    ``matches`` counts matches *played*, taken from the match table rather than
    from the shots themselves, so a team that failed to register a shot in a
    match still has it in the denominator of ``shots_per_match``.
    """
    return connection.execute(
        f"""
        WITH appearances AS (
            SELECT home_team AS team, match_id FROM matches
            UNION ALL
            SELECT away_team AS team, match_id FROM matches
        ),
        played AS (
            SELECT team, count(DISTINCT match_id) AS matches
            FROM appearances
            GROUP BY team
        )
        SELECT
            s.team,
            p.matches,
            count(*) AS shots,
            count(*) FILTER (WHERE s.is_goal) AS goals,
            round(sum(s.statsbomb_xg), 2) AS statsbomb_xg,
            round(count(*)::DOUBLE / NULLIF(p.matches, 0), 2) AS shots_per_match
        FROM shots s
        JOIN played p ON p.team = s.team
        WHERE {_shootout_filter(include_shootouts)}
        GROUP BY s.team, p.matches
        ORDER BY shots DESC, s.team
        """
    ).df()


def player_conversion_rates(
    connection: duckdb.DuckDBPyConnection,
    *,
    min_shots: int = 0,
    include_shootouts: bool = False,
) -> pd.DataFrame:
    """Conversion rate per player, over *every* player in the dimension table.

    The join is a LEFT JOIN, so players who never took a shot appear with zero
    shots. Their conversion rate is ``NULL`` rather than 0 or a division error:
    with no attempts the quantity is undefined, and reporting 0% would rank a
    goalkeeper who never shot alongside a striker who missed everything.
    """
    return connection.execute(
        f"""
        SELECT
            p.player_id,
            p.display_name AS player,
            p.team,
            p.squad_listings,
            count(s.event_id) AS shots,
            count(s.event_id) FILTER (WHERE s.is_goal) AS goals,
            round(
                count(s.event_id) FILTER (WHERE s.is_goal)::DOUBLE
                    / NULLIF(count(s.event_id), 0),
                4
            ) AS conversion_rate,
            round(sum(s.statsbomb_xg), 3) AS statsbomb_xg
        FROM players p
        LEFT JOIN shots s
               ON s.player_id = p.player_id
              AND {_shootout_filter(include_shootouts)}
        GROUP BY p.player_id, p.display_name, p.team, p.squad_listings
        HAVING count(s.event_id) >= ?
        ORDER BY goals DESC, shots DESC, player
        """,
        [min_shots],
    ).df()


def team_conversion_rates(
    connection: duckdb.DuckDBPyConnection, *, include_shootouts: bool = False
) -> pd.DataFrame:
    """Conversion rate per team, guarding the denominator the same way."""
    return connection.execute(
        f"""
        SELECT
            s.team,
            count(*) AS shots,
            count(*) FILTER (WHERE s.is_goal) AS goals,
            round(
                count(*) FILTER (WHERE s.is_goal)::DOUBLE / NULLIF(count(*), 0), 4
            ) AS conversion_rate
        FROM shots s
        WHERE {_shootout_filter(include_shootouts)}
        GROUP BY s.team
        ORDER BY conversion_rate DESC NULLS LAST, shots DESC
        """
    ).df()


# --------------------------------------------------------------------------- #
# Command line demonstration
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        prog="python -m football_intelligence.data.queries",
        description="Run the analytical queries against the processed Parquet tables.",
    )
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--top", type=int, default=10, help="Rows to show per ranking.")
    arguments = parser.parse_args(argv)

    try:
        connection = connect(arguments.processed_root)
    except StorageError as error:
        logger.error("%s", error)
        return 1

    with connection:
        pd.set_option("display.width", 140)
        sections = (
            ("matches by competition season", match_counts_by_competition(connection)),
            ("shots by player", shots_by_player(connection, limit=arguments.top)),
            ("goals by player", goals_by_player(connection, limit=arguments.top)),
            ("shots by team", shots_by_team(connection).head(arguments.top)),
            (
                "conversion rate by player (min 20 shots)",
                player_conversion_rates(connection, min_shots=20).head(arguments.top),
            ),
        )
        for title, frame in sections:
            print(f"\n=== {title} ===")
            print(frame.to_string(index=False))

        undefined = player_conversion_rates(connection)
        no_shots = undefined[undefined["shots"] == 0]
        print(
            f"\nplayers with no shots: {len(no_shots)} "
            f"(conversion_rate is NULL for all of them: "
            f"{bool(no_shots['conversion_rate'].isna().all())})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
