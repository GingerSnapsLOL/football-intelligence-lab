"""Normalise raw StatsBomb JSON into analytical Parquet tables.

Raw JSON is nested, schema-variable by event type, and expensive to scan: reading
every event of the development subset costs ~350 MB of parsing to reach the 0.7%
of rows that are shots. This module converts it once into four flat tables that
DuckDB and pandas can query cheaply.

Tables written to ``data/processed``:

``matches``
    One row per match. Dimension table for grouping, temporal splits and the
    coordinate-fidelity flags that differ between competitions.
``events``
    One row per event, carrying only the attributes common to every event type.
    Type-specific payloads (pass, carry, duel, ...) are deliberately not
    flattened here; ``shots`` is the one type we model, and widening the table
    for the rest would be speculative.
``shots``
    One row per shot with the shot payload flattened. This is the raw modelling
    grain; derived geometry (distance, angle) belongs to the feature layer.
``players``
    One row per player, aggregated from the lineup files, with squad-listing
    counts usable as exposure denominators.

Design notes:

- Raw files are only ever read. Nothing here writes to ``data/raw``.
- Identifiers stay integers, never floats, and text stays text.
- ``is_shootout`` marks period-5 penalty shootouts, which are a different
  data-generating process and are excluded from the analytical queries by
  default.
- ``statsbomb_xg`` is retained as a *benchmark*, not a feature: it is the output
  of StatsBomb's own model for the quantity we intend to predict.
"""

import argparse
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd

from football_intelligence.data import statsbomb
from football_intelligence.data.statsbomb import CompetitionSeason, JSONObject

logger = logging.getLogger(__name__)

DEFAULT_PROCESSED_ROOT: Final = Path("data/processed")
TABLE_NAMES: Final = ("matches", "events", "shots", "players")
PARQUET_COMPRESSION: Final = "zstd"

# StatsBomb records period 5 as the penalty shootout (specification, general
# attributes). Shootout penalties are not part of the match score and convert at
# roughly six times the in-play rate.
SHOOTOUT_PERIOD: Final = 5

# Shot booleans that StatsBomb omits entirely when false.
SHOT_FLAGS: Final = (
    "first_time",
    "one_on_one",
    "open_goal",
    "deflected",
    "aerial_won",
    "follows_dribble",
    "redirect",
)

MATCH_DTYPES: Final[dict[str, str]] = {
    "match_id": "int64",
    "competition_id": "int64",
    "season_id": "int64",
    "competition_name": "string",
    "season_name": "string",
    "kick_off": "string",
    "stage_id": "Int64",
    "stage": "string",
    "match_week": "Int64",
    "home_team_id": "int64",
    "home_team": "string",
    "away_team_id": "int64",
    "away_team": "string",
    "home_score": "int64",
    "away_score": "int64",
    "total_goals": "int64",
    "referee": "string",
    "stadium": "string",
    "shot_fidelity_version": "string",
    "xy_fidelity_version": "string",
}

EVENT_DTYPES: Final[dict[str, str]] = {
    "event_id": "string",
    "match_id": "int64",
    "index": "int64",
    "period": "int64",
    "timestamp": "string",
    "minute": "int64",
    "second": "int64",
    "type_id": "Int64",
    "type": "string",
    "possession": "Int64",
    "possession_team": "string",
    "play_pattern": "string",
    "team_id": "Int64",
    "team": "string",
    "player_id": "Int64",
    "player": "string",
    "position": "string",
    "x": "float64",
    "y": "float64",
    "duration": "float64",
    "under_pressure": "bool",
    "out": "bool",
    "off_camera": "bool",
}

SHOT_DTYPES: Final[dict[str, str]] = {
    "event_id": "string",
    "match_id": "int64",
    "index": "int64",
    "period": "int64",
    "timestamp": "string",
    "minute": "int64",
    "second": "int64",
    "team_id": "int64",
    "team": "string",
    "player_id": "int64",
    "player": "string",
    "position": "string",
    "play_pattern": "string",
    "x": "float64",
    "y": "float64",
    "end_x": "float64",
    "end_y": "float64",
    "end_z": "float64",
    "outcome_id": "Int64",
    "outcome": "string",
    "shot_type": "string",
    "body_part": "string",
    "technique": "string",
    "statsbomb_xg": "float64",
    "key_pass_id": "string",
    "is_goal": "bool",
    "is_shootout": "bool",
    "has_freeze_frame": "bool",
    "under_pressure": "bool",
    **dict.fromkeys(SHOT_FLAGS, "bool"),
}

PLAYER_DTYPES: Final[dict[str, str]] = {
    "player_id": "int64",
    "player_name": "string",
    "player_nickname": "string",
    "display_name": "string",
    "country": "string",
    "team_id": "int64",
    "team": "string",
    "squad_listings": "int64",
    "matches_with_position": "int64",
}

PRIMARY_KEYS: Final[dict[str, str]] = {
    "matches": "match_id",
    "events": "event_id",
    "shots": "event_id",
    "players": "player_id",
}


class StorageError(RuntimeError):
    """Raised when raw data cannot be normalised or a table cannot be read."""


@dataclass(frozen=True, slots=True)
class TableSummary:
    """What was written for one table."""

    name: str
    path: Path
    rows: int
    columns: int
    bytes_written: int

    def __str__(self) -> str:
        return (
            f"{self.name:<9} {self.rows:>8,} rows  {self.columns:>3} cols  "
            f"{self.bytes_written / 1_000_000:>7.2f} MB  {self.path}"
        )


# --------------------------------------------------------------------------- #
# Small readers for nested JSON
# --------------------------------------------------------------------------- #


# These three helpers read arbitrary positions in a decoded JSON document, where
# the value really can be any JSON type. `Any` is the honest annotation and is
# confined to this block; every caller converts the result to a concrete type.


def _nested(record: JSONObject, *keys: str) -> Any:  # noqa: ANN401
    """Read a nested value, returning ``None`` if any level is missing."""
    current: Any = record
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _coordinate(location: Any, index: int) -> float | None:  # noqa: ANN401
    """Read one component of a StatsBomb ``[x, y]`` or ``[x, y, z]`` array."""
    if isinstance(location, list) and len(location) > index:
        value = location[index]
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _require(value: Any, field: str, context: str) -> Any:  # noqa: ANN401
    """Return ``value``, or fail naming the field and the record it came from."""
    if value is None:
        raise StorageError(f"{context} is missing required field {field!r}.")
    return value


# --------------------------------------------------------------------------- #
# Normalisation (pure functions over raw records)
# --------------------------------------------------------------------------- #


def normalize_match(record: JSONObject) -> JSONObject:
    """Flatten one raw match record into a ``matches`` row."""
    match_id = _require(record.get("match_id"), "match_id", "Match record")
    home_score = int(_require(record.get("home_score"), "home_score", f"Match {match_id}"))
    away_score = int(_require(record.get("away_score"), "away_score", f"Match {match_id}"))
    return {
        "match_id": int(match_id),
        "competition_id": int(_nested(record, "competition", "competition_id")),
        "season_id": int(_nested(record, "season", "season_id")),
        "competition_name": _nested(record, "competition", "competition_name"),
        "season_name": _nested(record, "season", "season_name"),
        "match_date": record.get("match_date"),
        "kick_off": record.get("kick_off"),
        "stage_id": _nested(record, "competition_stage", "id"),
        "stage": _nested(record, "competition_stage", "name"),
        "match_week": record.get("match_week"),
        "home_team_id": int(_nested(record, "home_team", "home_team_id")),
        "home_team": _nested(record, "home_team", "home_team_name"),
        "away_team_id": int(_nested(record, "away_team", "away_team_id")),
        "away_team": _nested(record, "away_team", "away_team_name"),
        "home_score": home_score,
        "away_score": away_score,
        "total_goals": home_score + away_score,
        "referee": _nested(record, "referee", "name"),
        "stadium": _nested(record, "stadium", "name"),
        # Collection precision differs between competitions; keeping the flags
        # makes that visible to anything comparing coordinate-derived features.
        "shot_fidelity_version": _nested(record, "metadata", "shot_fidelity_version"),
        "xy_fidelity_version": _nested(record, "metadata", "xy_fidelity_version"),
    }


def normalize_event(event: JSONObject, match_id: int) -> JSONObject:
    """Flatten the attributes common to every event type into an ``events`` row.

    Type-specific payloads are intentionally dropped here; shots are handled by
    :func:`normalize_shot`.
    """
    event_id = _require(event.get("id"), "id", f"Event in match {match_id}")
    location = event.get("location")
    return {
        "event_id": str(event_id),
        "match_id": match_id,
        "index": int(_require(event.get("index"), "index", f"Event {event_id}")),
        "period": int(_require(event.get("period"), "period", f"Event {event_id}")),
        "timestamp": event.get("timestamp"),
        "minute": int(event.get("minute", 0)),
        "second": int(event.get("second", 0)),
        "type_id": _nested(event, "type", "id"),
        "type": _nested(event, "type", "name"),
        "possession": event.get("possession"),
        "possession_team": _nested(event, "possession_team", "name"),
        "play_pattern": _nested(event, "play_pattern", "name"),
        "team_id": _nested(event, "team", "id"),
        "team": _nested(event, "team", "name"),
        "player_id": _nested(event, "player", "id"),
        "player": _nested(event, "player", "name"),
        "position": _nested(event, "position", "name"),
        "x": _coordinate(location, 0),
        "y": _coordinate(location, 1),
        "duration": event.get("duration"),
        # Absent means false in StatsBomb's encoding, not unknown.
        "under_pressure": bool(event.get("under_pressure", False)),
        "out": bool(event.get("out", False)),
        "off_camera": bool(event.get("off_camera", False)),
    }


def normalize_shot(event: JSONObject, match_id: int) -> JSONObject:
    """Flatten one shot event into a ``shots`` row.

    Raises:
        StorageError: if the shot lacks the payload, player or team that every
            shot in the source data carries.
    """
    event_id = _require(event.get("id"), "id", f"Shot in match {match_id}")
    context = f"Shot {event_id}"
    shot = _require(event.get("shot"), "shot", context)
    location = event.get("location")
    period = int(_require(event.get("period"), "period", context))
    end_location = shot.get("end_location")

    record = {
        "event_id": str(event_id),
        "match_id": match_id,
        "index": int(_require(event.get("index"), "index", context)),
        "period": period,
        "timestamp": event.get("timestamp"),
        "minute": int(event.get("minute", 0)),
        "second": int(event.get("second", 0)),
        "team_id": int(_require(_nested(event, "team", "id"), "team.id", context)),
        "team": _nested(event, "team", "name"),
        "player_id": int(_require(_nested(event, "player", "id"), "player.id", context)),
        "player": _nested(event, "player", "name"),
        "position": _nested(event, "position", "name"),
        "play_pattern": _nested(event, "play_pattern", "name"),
        "x": _coordinate(location, 0),
        "y": _coordinate(location, 1),
        "end_x": _coordinate(end_location, 0),
        "end_y": _coordinate(end_location, 1),
        "end_z": _coordinate(end_location, 2),
        "outcome_id": _nested(shot, "outcome", "id"),
        "outcome": _nested(shot, "outcome", "name"),
        "shot_type": _nested(shot, "type", "name"),
        "body_part": _nested(shot, "body_part", "name"),
        "technique": _nested(shot, "technique", "name"),
        "statsbomb_xg": shot.get("statsbomb_xg"),
        "key_pass_id": shot.get("key_pass_id"),
        "is_goal": _nested(shot, "outcome", "name") == "Goal",
        "is_shootout": period == SHOOTOUT_PERIOD,
        "has_freeze_frame": "freeze_frame" in shot,
        "under_pressure": bool(event.get("under_pressure", False)),
    }
    for flag in SHOT_FLAGS:
        record[flag] = bool(shot.get(flag, False))
    return record


def normalize_squad_listing(player: JSONObject, team: JSONObject, match_id: int) -> JSONObject:
    """Flatten one lineup entry into a player-match squad listing."""
    player_id = _require(player.get("player_id"), "player_id", f"Lineup entry in match {match_id}")
    nickname = player.get("player_nickname")
    full_name = player.get("player_name")
    positions = player.get("positions") or []
    return {
        "player_id": int(player_id),
        "player_name": full_name,
        "player_nickname": nickname,
        "display_name": nickname or full_name,
        "country": _nested(player, "country", "name"),
        "team_id": int(_require(team.get("team_id"), "team_id", f"Lineup in match {match_id}")),
        "team": team.get("team_name"),
        "match_id": match_id,
        "has_position": len(positions) > 0,
    }


# --------------------------------------------------------------------------- #
# Frame assembly
# --------------------------------------------------------------------------- #


def _deduplicate(frame: pd.DataFrame, key: str, table: str) -> pd.DataFrame:
    """Drop repeated primary keys, keeping the first and reporting the loss."""
    duplicated = frame.duplicated(subset=key)
    count = int(duplicated.sum())
    if count:
        logger.warning(
            "%s: dropped %d duplicate row(s) on %s (kept first occurrence)", table, count, key
        )
        frame = frame.loc[~duplicated]
    return frame.reset_index(drop=True)


def _apply_schema(frame: pd.DataFrame, dtypes: dict[str, str], table: str) -> pd.DataFrame:
    """Order columns and enforce dtypes so identifiers never become floats."""
    missing = set(dtypes) - set(frame.columns)
    if missing:
        raise StorageError(f"{table}: normalised rows are missing columns {sorted(missing)}.")
    extra = [column for column in frame.columns if column not in dtypes]
    return frame[[*dtypes.keys(), *extra]].astype(dtypes)


def build_matches(records: Iterable[JSONObject]) -> pd.DataFrame:
    """Build the ``matches`` table from raw match records."""
    frame = pd.DataFrame([normalize_match(record) for record in records])
    if frame.empty:
        raise StorageError("No match records supplied; cannot build the matches table.")
    frame = _apply_schema(frame, MATCH_DTYPES, "matches")
    frame["match_date"] = pd.to_datetime(frame["match_date"])
    frame = _deduplicate(frame, "match_id", "matches")
    return frame.sort_values("match_id", ignore_index=True)


def build_events(events: Iterable[JSONObject], match_id: int) -> pd.DataFrame:
    """Build ``events`` rows for a single match."""
    return _apply_schema(
        pd.DataFrame([normalize_event(event, match_id) for event in events]),
        EVENT_DTYPES,
        "events",
    )


def build_shots(events: Iterable[JSONObject], match_id: int) -> pd.DataFrame:
    """Build ``shots`` rows for a single match from its full event list."""
    shots = statsbomb.events_of_type(events, "Shot")
    if not shots:
        return _apply_schema(pd.DataFrame(columns=list(SHOT_DTYPES)), SHOT_DTYPES, "shots")
    return _apply_schema(
        pd.DataFrame([normalize_shot(event, match_id) for event in shots]),
        SHOT_DTYPES,
        "shots",
    )


def build_players(listings: Iterable[JSONObject]) -> pd.DataFrame:
    """Aggregate player-match squad listings into one row per player."""
    frame = pd.DataFrame(list(listings))
    if frame.empty:
        raise StorageError("No lineup entries supplied; cannot build the players table.")
    players = frame.groupby("player_id", as_index=False).agg(
        player_name=("player_name", "first"),
        player_nickname=("player_nickname", "first"),
        display_name=("display_name", "first"),
        country=("country", "first"),
        team_id=("team_id", "first"),
        team=("team", "first"),
        squad_listings=("match_id", "nunique"),
        matches_with_position=("has_position", "sum"),
    )
    players = _apply_schema(players, PLAYER_DTYPES, "players")
    return players.sort_values("player_id", ignore_index=True)


def build_tables(
    selection: Sequence[CompetitionSeason] = statsbomb.DEFAULT_DEVELOPMENT_SUBSET,
    raw_root: Path = statsbomb.DEFAULT_RAW_ROOT,
) -> dict[str, pd.DataFrame]:
    """Read raw JSON for ``selection`` and return the four analytical tables.

    Events are read one match at a time so peak memory stays proportional to a
    single match rather than to the whole subset.
    """
    match_records: list[JSONObject] = []
    for season in selection:
        match_records.extend(
            statsbomb.load_matches(season.competition_id, season.season_id, raw_root)
        )
    matches = build_matches(match_records)

    event_frames: list[pd.DataFrame] = []
    shot_frames: list[pd.DataFrame] = []
    listings: list[JSONObject] = []

    for match_id in matches["match_id"].tolist():
        events = statsbomb.load_events(match_id, raw_root)
        event_frames.append(build_events(events, match_id))
        shot_frames.append(build_shots(events, match_id))
        for team in statsbomb.load_lineups(match_id, raw_root):
            for player in team.get("lineup", []):
                listings.append(normalize_squad_listing(player, team, match_id))

    events_table = _deduplicate(
        pd.concat(event_frames, ignore_index=True), "event_id", "events"
    ).sort_values(["match_id", "index"], ignore_index=True)
    shots_table = _deduplicate(
        pd.concat(shot_frames, ignore_index=True), "event_id", "shots"
    ).sort_values(["match_id", "index"], ignore_index=True)

    logger.info(
        "normalised %d matches, %d events, %d shots, %d squad listings",
        len(matches),
        len(events_table),
        len(shots_table),
        len(listings),
    )
    return {
        "matches": matches,
        "events": events_table,
        "shots": shots_table,
        "players": build_players(listings),
    }


# --------------------------------------------------------------------------- #
# Parquet IO
# --------------------------------------------------------------------------- #


def table_path(name: str, processed_root: Path = DEFAULT_PROCESSED_ROOT) -> Path:
    """Location of one processed table."""
    if name not in TABLE_NAMES:
        raise StorageError(f"Unknown table {name!r}; expected one of {list(TABLE_NAMES)}.")
    return processed_root / f"{name}.parquet"


def write_table(
    frame: pd.DataFrame, name: str, processed_root: Path = DEFAULT_PROCESSED_ROOT
) -> TableSummary:
    """Write one table to Parquet, replacing any previous build."""
    path = table_path(name, processed_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, engine="pyarrow", compression=PARQUET_COMPRESSION, index=False)
    summary = TableSummary(
        name=name,
        path=path,
        rows=len(frame),
        columns=len(frame.columns),
        bytes_written=path.stat().st_size,
    )
    logger.info("wrote %s", summary)
    return summary


def read_table(name: str, processed_root: Path = DEFAULT_PROCESSED_ROOT) -> pd.DataFrame:
    """Read one processed table."""
    path = table_path(name, processed_root)
    if not path.exists():
        raise StorageError(f"{path} does not exist. Build it first with `make process`.")
    return pd.read_parquet(path)


def build_processed_tables(
    selection: Sequence[CompetitionSeason] = statsbomb.DEFAULT_DEVELOPMENT_SUBSET,
    raw_root: Path = statsbomb.DEFAULT_RAW_ROOT,
    processed_root: Path = DEFAULT_PROCESSED_ROOT,
) -> dict[str, TableSummary]:
    """Normalise raw JSON and write every analytical table to Parquet."""
    tables = build_tables(selection, raw_root)
    return {name: write_table(tables[name], name, processed_root) for name in TABLE_NAMES}


# --------------------------------------------------------------------------- #
# Command line entry point
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m football_intelligence.data.storage",
        description="Normalise raw StatsBomb JSON into Parquet tables.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=statsbomb.DEFAULT_RAW_ROOT,
        help=f"Source of raw JSON (default: {statsbomb.DEFAULT_RAW_ROOT}).",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
        help=f"Destination for Parquet tables (default: {DEFAULT_PROCESSED_ROOT}).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = _build_parser().parse_args(argv)
    try:
        summaries = build_processed_tables(
            statsbomb.DEFAULT_DEVELOPMENT_SUBSET,
            arguments.raw_root,
            arguments.processed_root,
        )
    except (StorageError, statsbomb.StatsBombError) as error:
        logger.error("%s", error)
        return 1

    total = sum(summary.bytes_written for summary in summaries.values())
    logger.info("Done: %d tables, %.2f MB total", len(summaries), total / 1_000_000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
