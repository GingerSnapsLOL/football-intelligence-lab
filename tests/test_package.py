"""Smoke tests: the package is importable and installed as distribution metadata."""

from importlib.metadata import version

import football_intelligence


def test_package_is_importable_from_src_layout() -> None:
    assert football_intelligence.__doc__ is not None


def test_version_matches_installed_distribution_metadata() -> None:
    assert football_intelligence.__version__ == version("football-intelligence-lab")
    assert football_intelligence.__version__ != "0.0.0.dev0"
