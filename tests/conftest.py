"""Shared fixtures: a raw StatsBomb tree built from the tiny JSON fixtures.

The layout mirrors the upstream one exactly, so anything that reads through
``football_intelligence.data.statsbomb`` works against it offline.

Fixture data covers two matches of one competition season:

- match 5001: Alpha FC vs Beta FC, 5 events, 2 shots (1 goal)
- match 5002: Alpha FC vs Gamma FC, 4 events, 2 shots (1 open-play goal and one
  period-5 shootout penalty)
"""

from pathlib import Path

import pytest

from football_intelligence.data import statsbomb

FIXTURES = Path(__file__).parent / "fixtures" / "statsbomb"


def fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


@pytest.fixture
def raw_root(tmp_path: Path) -> Path:
    """A raw data root populated with the fixture files in upstream layout."""
    root = tmp_path / "statsbomb"

    def place(path: Path, fixture: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(fixture_bytes(fixture))

    place(statsbomb.competitions_path(root), "competitions.json")
    place(statsbomb.matches_path(900, 1, root), "matches.json")
    place(statsbomb.events_path(5001, root), "events.json")
    place(statsbomb.lineups_path(5001, root), "lineups.json")
    place(statsbomb.events_path(5002, root), "events_5002.json")
    place(statsbomb.lineups_path(5002, root), "lineups_5002.json")
    return root
