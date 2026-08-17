"""Reproducible acquisition of StatsBomb Open Data.

The upstream repository (https://github.com/statsbomb/open-data) publishes plain
JSON files. This module mirrors that layout verbatim under ``data/raw/statsbomb``::

    competitions.json                       every competition/season pair
    matches/{competition_id}/{season_id}.json   match list for one season
    events/{match_id}.json                  event stream for one match
    lineups/{match_id}.json                 both starting XIs and substitutes

Only the requested competition/season subsets are fetched -- the full repository
is several gigabytes and is never needed here.

Raw files are written byte-for-byte as published. Nothing in this module edits a
downloaded file: normalisation belongs downstream in ``data/interim`` and
``data/processed``. Downloads are idempotent -- per-match files are skipped once
present, and the two index files are revalidated by ETag -- so re-running an
acquisition costs a few small conditional requests. The only non-StatsBomb file
written is the ``.etags.json`` validator sidecar.

Data is provided by StatsBomb under the StatsBomb Open Data User Agreement and
must not be committed to this repository. Attribution is required when results
derived from it are published.
"""

import argparse
import json
import logging
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

# StatsBomb JSON records are deeply nested and schema-variable by event type.
# They are deliberately kept as plain dictionaries so the source schema stays
# visible to the reader instead of being hidden behind wrapper objects.
JSONObject = dict[str, Any]

OPEN_DATA_BASE_URL: Final = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
DEFAULT_RAW_ROOT: Final = Path("data/raw/statsbomb")
REQUEST_TIMEOUT_SECONDS: Final = 60.0
USER_AGENT: Final = "football-intelligence-lab/0.1 (research project)"
HTTP_NOT_MODIFIED: Final = 304
HTTP_NOT_FOUND: Final = 404

# ETags of the mutable index files, kept beside the mirror so it stays portable.
# This is the only file this module writes that did not come from StatsBomb.
ETAG_STORE_NAME: Final = ".etags.json"


class StatsBombError(RuntimeError):
    """Base error for StatsBomb acquisition and loading problems."""


class StatsBombNotFoundError(StatsBombError):
    """A requested resource does not exist upstream or locally."""


@dataclass(frozen=True, slots=True)
class CompetitionSeason:
    """One competition/season pair, the unit of selection in the open data."""

    competition_id: int
    season_id: int
    competition_name: str | None = None
    season_name: str | None = None
    competition_gender: str | None = None

    def __str__(self) -> str:
        label = f"{self.competition_name or '?'} {self.season_name or '?'}"
        return f"{self.competition_id}/{self.season_id} ({label.strip()})"


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """Outcome of a single file request."""

    path: Path
    url: str
    downloaded: bool
    bytes_written: int


@dataclass(frozen=True, slots=True)
class DownloadSummary:
    """Aggregate outcome of acquiring one or more competition seasons."""

    results: tuple[DownloadResult, ...]

    @property
    def downloaded_files(self) -> int:
        return sum(1 for result in self.results if result.downloaded)

    @property
    def skipped_files(self) -> int:
        return sum(1 for result in self.results if not result.downloaded)

    @property
    def bytes_written(self) -> int:
        return sum(result.bytes_written for result in self.results)

    def __str__(self) -> str:
        megabytes = self.bytes_written / 1_000_000
        return (
            f"{self.downloaded_files} file(s) downloaded ({megabytes:.1f} MB), "
            f"{self.skipped_files} already up to date"
        )


# A deliberately small starting dataset: two men's international tournaments with
# consistent StatsBomb collection. Together they carry roughly 3,000 shots across
# 115 matches -- enough for EDA, hypothesis testing and a first xG model, while
# staying at a few percent of the full open-data repository. Player-level repeated
# measures are limited (a player appears in at most ~7 matches), so a club season
# should be added before the longitudinal/hierarchical tasks.
DEFAULT_DEVELOPMENT_SUBSET: Final[tuple[CompetitionSeason, ...]] = (
    CompetitionSeason(43, 3, "FIFA World Cup", "2018", "male"),
    CompetitionSeason(55, 43, "UEFA Euro", "2020", "male"),
)


# --------------------------------------------------------------------------- #
# Local paths (mirroring the upstream layout exactly)
# --------------------------------------------------------------------------- #


def competitions_path(raw_root: Path = DEFAULT_RAW_ROOT) -> Path:
    return raw_root / "competitions.json"


def matches_path(competition_id: int, season_id: int, raw_root: Path = DEFAULT_RAW_ROOT) -> Path:
    return raw_root / "matches" / str(competition_id) / f"{season_id}.json"


def events_path(match_id: int, raw_root: Path = DEFAULT_RAW_ROOT) -> Path:
    return raw_root / "events" / f"{match_id}.json"


def lineups_path(match_id: int, raw_root: Path = DEFAULT_RAW_ROOT) -> Path:
    return raw_root / "lineups" / f"{match_id}.json"


def etag_store_path(raw_root: Path = DEFAULT_RAW_ROOT) -> Path:
    """Location of the download-validator sidecar (not StatsBomb data)."""
    return raw_root / ETAG_STORE_NAME


# --------------------------------------------------------------------------- #
# HTTP transfer
# --------------------------------------------------------------------------- #


def _fetch(url: str, *, etag: str | None) -> tuple[bytes, str | None] | None:
    """Return ``(body, etag)``, or ``None`` when the server reports it unchanged.

    Passing ``etag`` turns this into a conditional GET. GitHub's raw host serves
    no ``Last-Modified`` header but does honour ``If-None-Match``, so an unchanged
    file costs one small 304 round trip instead of a full transfer.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if etag is not None:
        request.add_header("If-None-Match", etag)

    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            body: bytes = response.read()
            new_etag: str | None = response.headers.get("ETag")
    except urllib.error.HTTPError as error:
        if error.code == HTTP_NOT_MODIFIED:
            return None
        if error.code == HTTP_NOT_FOUND:
            raise StatsBombNotFoundError(
                f"{url} does not exist in StatsBomb Open Data (HTTP 404). "
                "Check the competition/season/match identifier against competitions.json."
            ) from error
        raise StatsBombError(f"Request for {url} failed with HTTP {error.code}.") from error
    except urllib.error.URLError as error:
        raise StatsBombError(f"Could not reach {url}: {error.reason}") from error

    return body, new_etag


def _read_etags(store: Path) -> dict[str, str]:
    """Read the URL-to-ETag map, treating a missing or damaged store as empty."""
    if not store.exists():
        return {}
    try:
        recorded = json.loads(store.read_text(encoding="utf-8"))
    except ValueError:
        logger.warning("Ignoring unreadable ETag store %s; files will be re-downloaded.", store)
        return {}
    if not isinstance(recorded, dict):
        return {}
    return {str(url): str(etag) for url, etag in recorded.items()}


def _store_etag(store: Path, url: str, etag: str) -> None:
    recorded = _read_etags(store)
    recorded[url] = etag
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(json.dumps(recorded, indent=2, sort_keys=True), encoding="utf-8")


def _write_atomically(destination: Path, body: bytes) -> None:
    """Write ``body`` verbatim, never leaving a partial file behind on failure."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.part")
    temporary.write_bytes(body)
    temporary.replace(destination)


def _download_json(
    url: str, destination: Path, *, force: bool, etag_store: Path | None
) -> DownloadResult:
    """Download ``url`` to ``destination`` unless the local copy is already current.

    Per-match event and lineup files never change once published, so their mere
    existence is enough to skip them (``etag_store=None``). The competitions and
    match-list indexes do grow as StatsBomb releases new data, so those are
    revalidated against a recorded ETag and only rewritten when they differ.
    """
    if destination.exists() and not force:
        if etag_store is None:
            logger.debug("Present, skipping: %s", destination)
            return DownloadResult(destination, url, downloaded=False, bytes_written=0)

        outcome = _fetch(url, etag=_read_etags(etag_store).get(url))
        if outcome is None:
            logger.debug("Unchanged upstream, skipping: %s", destination)
            return DownloadResult(destination, url, downloaded=False, bytes_written=0)
    else:
        outcome = _fetch(url, etag=None)
        if outcome is None:  # pragma: no cover - unconditional requests never yield 304
            raise StatsBombError(f"Unexpected 'not modified' response for {url}.")

    body, etag = outcome

    # Guard against truncated bodies or an HTML error page being stored as raw data.
    try:
        json.loads(body)
    except ValueError as error:
        raise StatsBombError(f"Response from {url} was not valid JSON: {error}") from error

    _write_atomically(destination, body)
    if etag_store is not None and etag is not None:
        _store_etag(etag_store, url, etag)
    logger.info("Downloaded %s (%.1f KB)", destination, len(body) / 1000)
    return DownloadResult(destination, url, downloaded=True, bytes_written=len(body))


# --------------------------------------------------------------------------- #
# Acquisition
# --------------------------------------------------------------------------- #


def download_competitions(
    raw_root: Path = DEFAULT_RAW_ROOT, *, force: bool = False
) -> DownloadResult:
    """Fetch the competition/season index."""
    return _download_json(
        f"{OPEN_DATA_BASE_URL}/competitions.json",
        competitions_path(raw_root),
        force=force,
        etag_store=etag_store_path(raw_root),
    )


def download_matches(
    competition_id: int,
    season_id: int,
    raw_root: Path = DEFAULT_RAW_ROOT,
    *,
    force: bool = False,
) -> DownloadResult:
    """Fetch the match list for one competition season."""
    return _download_json(
        f"{OPEN_DATA_BASE_URL}/matches/{competition_id}/{season_id}.json",
        matches_path(competition_id, season_id, raw_root),
        force=force,
        etag_store=etag_store_path(raw_root),
    )


def download_events(
    match_id: int, raw_root: Path = DEFAULT_RAW_ROOT, *, force: bool = False
) -> DownloadResult:
    """Fetch the event stream for one match."""
    return _download_json(
        f"{OPEN_DATA_BASE_URL}/events/{match_id}.json",
        events_path(match_id, raw_root),
        force=force,
        etag_store=None,
    )


def download_lineups(
    match_id: int, raw_root: Path = DEFAULT_RAW_ROOT, *, force: bool = False
) -> DownloadResult:
    """Fetch the lineups for one match."""
    return _download_json(
        f"{OPEN_DATA_BASE_URL}/lineups/{match_id}.json",
        lineups_path(match_id, raw_root),
        force=force,
        etag_store=None,
    )


def download_competition_season(
    competition_id: int,
    season_id: int,
    raw_root: Path = DEFAULT_RAW_ROOT,
    *,
    include_lineups: bool = True,
    force: bool = False,
    limit_matches: int | None = None,
) -> DownloadSummary:
    """Acquire the match list and per-match files for one competition season.

    ``limit_matches`` truncates the match list, which keeps smoke tests and first
    experiments to a handful of files.
    """
    results: list[DownloadResult] = [
        download_matches(competition_id, season_id, raw_root, force=force)
    ]

    matches = load_matches(competition_id, season_id, raw_root)
    identifiers = match_ids(matches)
    if limit_matches is not None:
        if limit_matches < 1:
            raise ValueError(f"limit_matches must be at least 1, got {limit_matches}.")
        identifiers = identifiers[:limit_matches]

    logger.info(
        "Competition %s season %s: acquiring %d of %d matches",
        competition_id,
        season_id,
        len(identifiers),
        len(matches),
    )
    for match_id in identifiers:
        results.append(download_events(match_id, raw_root, force=force))
        if include_lineups:
            results.append(download_lineups(match_id, raw_root, force=force))

    return DownloadSummary(tuple(results))


def download_subset(
    selection: Sequence[CompetitionSeason] = DEFAULT_DEVELOPMENT_SUBSET,
    raw_root: Path = DEFAULT_RAW_ROOT,
    *,
    include_lineups: bool = True,
    force: bool = False,
    limit_matches: int | None = None,
) -> DownloadSummary:
    """Acquire every competition season in ``selection``, plus the competition index."""
    results: list[DownloadResult] = [download_competitions(raw_root, force=force)]
    for season in selection:
        summary = download_competition_season(
            season.competition_id,
            season.season_id,
            raw_root,
            include_lineups=include_lineups,
            force=force,
            limit_matches=limit_matches,
        )
        results.extend(summary.results)
    return DownloadSummary(tuple(results))


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _load_json_array(path: Path, description: str) -> list[JSONObject]:
    if not path.exists():
        raise StatsBombNotFoundError(
            f"{description} not found at {path}. Acquire it first, e.g. with `make data`."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise StatsBombError(f"{path} does not contain valid JSON: {error}") from error

    if not isinstance(payload, list):
        raise StatsBombError(
            f"Expected {path} to contain a JSON array of records, got {type(payload).__name__}."
        )
    return payload


def load_competitions(raw_root: Path = DEFAULT_RAW_ROOT) -> list[JSONObject]:
    """Load the raw competition/season records."""
    return _load_json_array(competitions_path(raw_root), "competitions.json")


def load_matches(
    competition_id: int, season_id: int, raw_root: Path = DEFAULT_RAW_ROOT
) -> list[JSONObject]:
    """Load the raw match records for one competition season."""
    return _load_json_array(
        matches_path(competition_id, season_id, raw_root),
        f"Match list for competition {competition_id} season {season_id}",
    )


def load_events(match_id: int, raw_root: Path = DEFAULT_RAW_ROOT) -> list[JSONObject]:
    """Load the raw event records for one match."""
    return _load_json_array(events_path(match_id, raw_root), f"Events for match {match_id}")


def load_lineups(match_id: int, raw_root: Path = DEFAULT_RAW_ROOT) -> list[JSONObject]:
    """Load the raw lineup records for one match (one entry per team)."""
    return _load_json_array(lineups_path(match_id, raw_root), f"Lineups for match {match_id}")


# --------------------------------------------------------------------------- #
# Small parsing helpers
# --------------------------------------------------------------------------- #


def competition_seasons(competitions: Iterable[JSONObject]) -> list[CompetitionSeason]:
    """Extract selectable competition/season pairs from raw competition records.

    Identifiers are required; display names and gender are optional and reported
    as ``None`` when absent.
    """
    seasons: list[CompetitionSeason] = []
    for record in competitions:
        try:
            competition_id = int(record["competition_id"])
            season_id = int(record["season_id"])
        except KeyError as error:
            raise StatsBombError(
                f"Competition record is missing required key {error}: {record!r}"
            ) from error
        seasons.append(
            CompetitionSeason(
                competition_id=competition_id,
                season_id=season_id,
                competition_name=record.get("competition_name"),
                season_name=record.get("season_name"),
                competition_gender=record.get("competition_gender"),
            )
        )
    return seasons


def match_ids(matches: Iterable[JSONObject]) -> list[int]:
    """Extract match identifiers from raw match records, preserving file order."""
    identifiers: list[int] = []
    for record in matches:
        if "match_id" not in record:
            raise StatsBombError(f"Match record is missing 'match_id': {record!r}")
        identifiers.append(int(record["match_id"]))
    return identifiers


def events_of_type(events: Iterable[JSONObject], type_name: str) -> list[JSONObject]:
    """Select events by their StatsBomb type name, e.g. ``"Shot"`` or ``"Pass"``.

    Events whose ``type`` block is absent or malformed are skipped rather than
    raising, because event payloads vary widely by type.
    """
    selected: list[JSONObject] = []
    for event in events:
        event_type = event.get("type")
        if isinstance(event_type, dict) and event_type.get("name") == type_name:
            selected.append(event)
    return selected


def player_names(lineups: Iterable[JSONObject]) -> dict[int, str]:
    """Map player id to display name across both teams of a match.

    ``player_nickname`` is optional upstream and often null; the full
    ``player_name`` is used whenever a nickname is unavailable.
    """
    names: dict[int, str] = {}
    for team in lineups:
        for player in team.get("lineup", []):
            if "player_id" not in player:
                raise StatsBombError(f"Lineup entry is missing 'player_id': {player!r}")
            nickname = player.get("player_nickname")
            full_name = player.get("player_name")
            display_name = nickname or full_name
            if display_name is None:
                raise StatsBombError(
                    f"Lineup entry {player['player_id']} has neither a name nor a nickname."
                )
            names[int(player["player_id"])] = str(display_name)
    return names


# --------------------------------------------------------------------------- #
# Command line entry point
# --------------------------------------------------------------------------- #


def _parse_competition_season(value: str) -> CompetitionSeason:
    competition_text, separator, season_text = value.partition("/")
    if not separator:
        raise argparse.ArgumentTypeError(
            f"Expected COMPETITION_ID/SEASON_ID (for example 43/3), got {value!r}."
        )
    try:
        return CompetitionSeason(int(competition_text), int(season_text))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Competition and season identifiers must be integers, got {value!r}."
        ) from error


def _build_parser() -> argparse.ArgumentParser:
    default_selection = " ".join(
        f"{season.competition_id}/{season.season_id}" for season in DEFAULT_DEVELOPMENT_SUBSET
    )
    parser = argparse.ArgumentParser(
        prog="python -m football_intelligence.data.statsbomb",
        description="Download a subset of StatsBomb Open Data into data/raw/statsbomb.",
    )
    parser.add_argument(
        "--competition-season",
        dest="selection",
        action="append",
        type=_parse_competition_season,
        metavar="COMP/SEASON",
        help=f"Competition season to acquire; repeatable. Default: {default_selection}",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help=f"Destination for raw JSON (default: {DEFAULT_RAW_ROOT}).",
    )
    parser.add_argument(
        "--limit-matches",
        type=int,
        default=None,
        help="Acquire only the first N matches of each season.",
    )
    parser.add_argument(
        "--skip-lineups", action="store_true", help="Download events but not lineups."
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download files that already exist locally."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_competitions",
        help="Print the available competition seasons and exit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = _build_parser().parse_args(argv)

    try:
        if arguments.list_competitions:
            download_competitions(arguments.raw_root, force=arguments.force)
            for season in competition_seasons(load_competitions(arguments.raw_root)):
                print(season)
            return 0

        selection = arguments.selection or list(DEFAULT_DEVELOPMENT_SUBSET)
        summary = download_subset(
            selection,
            arguments.raw_root,
            include_lineups=not arguments.skip_lineups,
            force=arguments.force,
            limit_matches=arguments.limit_matches,
        )
    except StatsBombError as error:
        logger.error("%s", error)
        return 1

    logger.info("Done: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
