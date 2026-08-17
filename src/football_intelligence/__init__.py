"""Football Intelligence Lab.

Data and modelling toolkit for football match and event analysis.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("football-intelligence-lab")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
