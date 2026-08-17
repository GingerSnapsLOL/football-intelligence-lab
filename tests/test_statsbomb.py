"""Offline tests for the StatsBomb acquisition layer.

No test in this module touches the network: HTTP transfer is stubbed at
``statsbomb._fetch`` and every payload comes from the tiny hand-written fixtures
in ``tests/fixtures/statsbomb``.
"""

import argparse
import json
import urllib.error
from email.message import Message
from pathlib import Path

import pytest

from football_intelligence.data import statsbomb
from football_intelligence.data.statsbomb import (
    CompetitionSeason,
    StatsBombError,
    StatsBombNotFoundError,
)

FIXTURES = Path(__file__).parent / "fixtures" / "statsbomb"


def _fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _fixture_json(name: str) -> list[dict[str, object]]:
    payload = json.loads(_fixture_bytes(name))
    assert isinstance(payload, list)
    return payload


# The ``raw_root`` fixture lives in conftest.py and is shared with the storage
# tests, which need both fixture matches present.


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


def test_local_paths_mirror_the_upstream_layout(tmp_path: Path) -> None:
    root = tmp_path / "statsbomb"
    assert statsbomb.competitions_path(root) == root / "competitions.json"
    assert statsbomb.matches_path(43, 3, root) == root / "matches" / "43" / "3.json"
    assert statsbomb.events_path(7580, root) == root / "events" / "7580.json"
    assert statsbomb.lineups_path(7580, root) == root / "lineups" / "7580.json"


# --------------------------------------------------------------------------- #
# Parsing: competitions
# --------------------------------------------------------------------------- #


def test_competition_seasons_parses_identifiers_and_names() -> None:
    seasons = statsbomb.competition_seasons(_fixture_json("competitions.json"))

    assert len(seasons) == 2
    assert seasons[0] == CompetitionSeason(900, 1, "Fixture Cup", "2018", "male")


def test_competition_seasons_tolerates_missing_optional_gender() -> None:
    seasons = statsbomb.competition_seasons(_fixture_json("competitions.json"))

    # The second fixture record deliberately omits competition_gender.
    assert seasons[1].competition_gender is None
    assert seasons[1].competition_name == "Fixture League"


def test_competition_seasons_rejects_records_without_identifiers() -> None:
    with pytest.raises(StatsBombError, match="missing required key"):
        statsbomb.competition_seasons([{"competition_name": "No Identifiers"}])


# --------------------------------------------------------------------------- #
# Parsing: matches
# --------------------------------------------------------------------------- #


def test_match_ids_preserve_file_order() -> None:
    assert statsbomb.match_ids(_fixture_json("matches.json")) == [5001, 5002]


def test_matches_load_when_optional_fields_are_absent(raw_root: Path) -> None:
    matches = statsbomb.load_matches(900, 1, raw_root)

    # Referee and stadium are optional upstream and missing from the second match.
    assert "referee" in matches[0]
    assert "referee" not in matches[1]
    assert "stadium" not in matches[1]
    assert matches[1]["match_id"] == 5002


def test_match_ids_rejects_records_without_match_id() -> None:
    with pytest.raises(StatsBombError, match="missing 'match_id'"):
        statsbomb.match_ids([{"match_date": "2018-06-14"}])


# --------------------------------------------------------------------------- #
# Parsing: events
# --------------------------------------------------------------------------- #


def test_events_of_type_selects_only_shots() -> None:
    events = _fixture_json("events.json")

    shots = statsbomb.events_of_type(events, "Shot")

    assert len(shots) == 2
    assert [shot["id"] for shot in shots] == [
        "00000000-0000-0000-0000-000000000003",
        "00000000-0000-0000-0000-000000000004",
    ]


def test_events_of_type_skips_events_without_a_type_block() -> None:
    events = _fixture_json("events.json")

    # The last fixture event has no "type" key at all; it must not raise.
    assert statsbomb.events_of_type(events, "Pass") == [events[1]]


def test_shot_parsing_tolerates_missing_optional_fields() -> None:
    shots = statsbomb.events_of_type(_fixture_json("events.json"), "Shot")
    complete, sparse = shots

    assert complete["under_pressure"] is True
    assert "freeze_frame" in complete["shot"]

    # under_pressure, first_time, technique and freeze_frame are all optional and
    # absent from the second shot: downstream feature code must expect that.
    assert "under_pressure" not in sparse
    assert "technique" not in sparse["shot"]
    assert "first_time" not in sparse["shot"]


def test_events_load_from_raw_root(raw_root: Path) -> None:
    events = statsbomb.load_events(5001, raw_root)

    assert len(events) == 5
    assert len(statsbomb.events_of_type(events, "Shot")) == 2


# --------------------------------------------------------------------------- #
# Parsing: lineups
# --------------------------------------------------------------------------- #


def test_player_names_prefer_nickname_and_fall_back_to_full_name() -> None:
    names = statsbomb.player_names(_fixture_json("lineups.json"))

    assert names[101] == "Player One"  # nickname present
    assert names[102] == "Player Reserve"  # nickname explicitly null
    assert names[202] == "Player Three"  # nickname key absent entirely


def test_player_names_rejects_entries_without_an_identifier() -> None:
    with pytest.raises(StatsBombError, match="missing 'player_id'"):
        statsbomb.player_names([{"lineup": [{"player_name": "Nameless"}]}])


# --------------------------------------------------------------------------- #
# Loading errors
# --------------------------------------------------------------------------- #


def test_loading_a_missing_file_points_at_the_acquisition_command(tmp_path: Path) -> None:
    with pytest.raises(StatsBombNotFoundError, match="make data"):
        statsbomb.load_events(999999, tmp_path)


def test_loading_invalid_json_reports_the_offending_path(tmp_path: Path) -> None:
    path = statsbomb.events_path(1, tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(StatsBombError, match="does not contain valid JSON"):
        statsbomb.load_events(1, tmp_path)


def test_loading_a_json_object_instead_of_an_array_is_rejected(tmp_path: Path) -> None:
    path = statsbomb.events_path(1, tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"events": []}', encoding="utf-8")

    with pytest.raises(StatsBombError, match="JSON array"):
        statsbomb.load_events(1, tmp_path)


# --------------------------------------------------------------------------- #
# Download behaviour (HTTP stubbed)
# --------------------------------------------------------------------------- #


def _stub_transfer(
    monkeypatch: pytest.MonkeyPatch, payloads: dict[str, bytes], requested: list[str]
) -> None:
    """Serve ``payloads`` by URL instead of performing real HTTP requests."""

    def fake_fetch(url: str, *, etag: str | None) -> tuple[bytes, str | None] | None:
        requested.append(url)
        if url not in payloads:
            raise StatsBombNotFoundError(f"{url} does not exist in StatsBomb Open Data")
        return payloads[url], f'"etag-for-{url}"'

    monkeypatch.setattr(statsbomb, "_fetch", fake_fetch)


def test_download_writes_the_response_bytes_verbatim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = _fixture_bytes("events.json")
    url = f"{statsbomb.OPEN_DATA_BASE_URL}/events/5001.json"
    requested: list[str] = []
    _stub_transfer(monkeypatch, {url: body}, requested)

    result = statsbomb.download_events(5001, tmp_path)

    assert result.downloaded is True
    assert result.bytes_written == len(body)
    # Raw data must be preserved exactly as published, not re-serialised.
    assert statsbomb.events_path(5001, tmp_path).read_bytes() == body


def test_existing_match_files_are_not_downloaded_again(
    raw_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested: list[str] = []
    _stub_transfer(monkeypatch, {}, requested)

    result = statsbomb.download_events(5001, raw_root)

    assert result.downloaded is False
    assert result.bytes_written == 0
    assert requested == []  # no request was issued at all


def test_force_redownloads_an_existing_file(
    raw_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replacement = b"[]"
    url = f"{statsbomb.OPEN_DATA_BASE_URL}/events/5001.json"
    requested: list[str] = []
    _stub_transfer(monkeypatch, {url: replacement}, requested)

    result = statsbomb.download_events(5001, raw_root, force=True)

    assert result.downloaded is True
    assert requested == [url]
    assert statsbomb.events_path(5001, raw_root).read_bytes() == replacement


def test_index_downloads_record_the_etag_for_later_revalidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = f"{statsbomb.OPEN_DATA_BASE_URL}/competitions.json"
    requested: list[str] = []
    _stub_transfer(monkeypatch, {url: _fixture_bytes("competitions.json")}, requested)

    statsbomb.download_competitions(tmp_path)

    recorded = json.loads(statsbomb.etag_store_path(tmp_path).read_text(encoding="utf-8"))
    assert recorded == {url: f'"etag-for-{url}"'}


def test_unchanged_index_files_are_revalidated_by_etag_and_not_rewritten(
    raw_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = f"{statsbomb.OPEN_DATA_BASE_URL}/competitions.json"
    statsbomb._store_etag(statsbomb.etag_store_path(raw_root), url, '"cached-etag"')
    original = statsbomb.competitions_path(raw_root).read_bytes()
    sent_etags: list[str | None] = []

    def fake_fetch(url: str, *, etag: str | None) -> tuple[bytes, str | None] | None:
        sent_etags.append(etag)
        return None  # server reports HTTP 304 Not Modified

    monkeypatch.setattr(statsbomb, "_fetch", fake_fetch)

    result = statsbomb.download_competitions(raw_root)

    assert result.downloaded is False
    assert sent_etags == ['"cached-etag"']  # conditional request, not a blind refetch
    assert statsbomb.competitions_path(raw_root).read_bytes() == original


def test_a_damaged_etag_store_is_ignored_rather_than_fatal(
    raw_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    statsbomb.etag_store_path(raw_root).write_text("{ truncated", encoding="utf-8")
    url = f"{statsbomb.OPEN_DATA_BASE_URL}/competitions.json"
    requested: list[str] = []
    _stub_transfer(monkeypatch, {url: _fixture_bytes("competitions.json")}, requested)

    result = statsbomb.download_competitions(raw_root)

    assert result.downloaded is True  # falls back to an unconditional fetch
    assert requested == [url]


def test_download_rejects_a_response_that_is_not_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = f"{statsbomb.OPEN_DATA_BASE_URL}/events/5001.json"
    requested: list[str] = []
    _stub_transfer(monkeypatch, {url: b"<html>404 Not Found</html>"}, requested)

    with pytest.raises(StatsBombError, match="not valid JSON"):
        statsbomb.download_events(5001, tmp_path)

    # Nothing partial may be left behind in the raw tree.
    assert not statsbomb.events_path(5001, tmp_path).exists()


def test_download_competition_season_acquires_matches_events_and_lineups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = statsbomb.OPEN_DATA_BASE_URL
    payloads = {
        f"{base}/matches/900/1.json": _fixture_bytes("matches.json"),
        f"{base}/events/5001.json": _fixture_bytes("events.json"),
        f"{base}/lineups/5001.json": _fixture_bytes("lineups.json"),
    }
    requested: list[str] = []
    _stub_transfer(monkeypatch, payloads, requested)

    summary = statsbomb.download_competition_season(900, 1, tmp_path, limit_matches=1)

    assert summary.downloaded_files == 3
    assert summary.skipped_files == 0
    assert requested == list(payloads)
    # limit_matches=1 must stop before the second match in the fixture list.
    assert not statsbomb.events_path(5002, tmp_path).exists()


def test_download_competition_season_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = statsbomb.OPEN_DATA_BASE_URL
    payloads = {
        f"{base}/matches/900/1.json": _fixture_bytes("matches.json"),
        f"{base}/events/5001.json": _fixture_bytes("events.json"),
        f"{base}/lineups/5001.json": _fixture_bytes("lineups.json"),
    }
    requested: list[str] = []
    _stub_transfer(monkeypatch, payloads, requested)

    statsbomb.download_competition_season(900, 1, tmp_path, limit_matches=1)

    def fetch_unchanged(url: str, *, etag: str | None) -> tuple[bytes, str | None] | None:
        requested.append(url)
        return None  # index revalidation returns 304 on the second run

    monkeypatch.setattr(statsbomb, "_fetch", fetch_unchanged)
    second = statsbomb.download_competition_season(900, 1, tmp_path, limit_matches=1)

    assert second.downloaded_files == 0
    assert second.skipped_files == 3
    assert second.bytes_written == 0


def test_limit_matches_must_be_positive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = statsbomb.OPEN_DATA_BASE_URL
    requested: list[str] = []
    _stub_transfer(
        monkeypatch, {f"{base}/matches/900/1.json": _fixture_bytes("matches.json")}, requested
    )

    with pytest.raises(ValueError, match="at least 1"):
        statsbomb.download_competition_season(900, 1, tmp_path, limit_matches=0)


def test_missing_competition_season_raises_a_useful_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested: list[str] = []
    _stub_transfer(monkeypatch, {}, requested)

    with pytest.raises(StatsBombNotFoundError, match="does not exist"):
        statsbomb.download_matches(999, 999, tmp_path)


def test_http_404_is_translated_into_a_domain_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_not_found(*args: object, **kwargs: object) -> None:
        raise urllib.error.HTTPError(
            url="https://example.invalid/x.json",
            code=404,
            msg="Not Found",
            hdrs=Message(),
            fp=None,
        )

    monkeypatch.setattr(urllib.request, "urlopen", raise_not_found)

    with pytest.raises(StatsBombNotFoundError, match="HTTP 404"):
        statsbomb._fetch("https://example.invalid/x.json", etag=None)


def test_network_failure_is_translated_into_a_domain_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_url_error(*args: object, **kwargs: object) -> None:
        raise urllib.error.URLError("name resolution failed")

    monkeypatch.setattr(urllib.request, "urlopen", raise_url_error)

    with pytest.raises(StatsBombError, match="Could not reach"):
        statsbomb._fetch("https://example.invalid/x.json", etag=None)


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def test_competition_season_argument_accepts_the_documented_format() -> None:
    assert statsbomb._parse_competition_season("43/3") == CompetitionSeason(43, 3)


@pytest.mark.parametrize("value", ["43", "43-3", "cup/3"])
def test_competition_season_argument_rejects_malformed_input(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match=r"43/3|integers"):
        statsbomb._parse_competition_season(value)


def test_default_subset_is_small_and_explicit() -> None:
    assert (
        CompetitionSeason(43, 3, "FIFA World Cup", "2018", "male"),
        CompetitionSeason(55, 43, "UEFA Euro", "2020", "male"),
    ) == statsbomb.DEFAULT_DEVELOPMENT_SUBSET
