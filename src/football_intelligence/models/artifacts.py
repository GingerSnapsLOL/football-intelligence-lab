"""Persist and load trained shot-probability models.

Artifacts live under ``artifacts/<task>/<model>/`` as a fitted sklearn pipeline
plus two JSON files. The API reads these on demand and never trains.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import joblib
from sklearn.pipeline import Pipeline

from football_intelligence.evaluation.metrics import ProbabilityMetrics

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACTS_ROOT: Final = Path("artifacts")
TASK_NAME: Final = "shot_goal_probability"

MODEL_FILENAME: Final = "model.joblib"
METRICS_FILENAME: Final = "metrics.json"
METADATA_FILENAME: Final = "metadata.json"

INTEGER_METRIC_KEYS: Final = frozenset({"n", "positives"})


class ArtifactError(RuntimeError):
    """Raised when a model artifact cannot be written or read."""


@dataclass(frozen=True, slots=True)
class ArtifactInfo:
    """On-disk description of one trained model, without the pipeline itself."""

    name: str
    task: str
    features: tuple[str, ...]
    training_samples: int
    test_samples: int
    validation_strategy: str
    metrics: dict[str, float | int]
    metadata: dict[str, object]


def metrics_as_dict(metrics: ProbabilityMetrics) -> dict[str, float | int]:
    """Convert :class:`ProbabilityMetrics` into JSON-serialisable numbers."""
    payload: dict[str, float | int] = {}
    for key, value in metrics.to_series().items():
        name = str(key)
        payload[name] = int(value) if name in INTEGER_METRIC_KEYS else float(value)
    return payload


def artifact_dir(name: str, root: Path = DEFAULT_ARTIFACTS_ROOT) -> Path:
    """Directory for one named model under a task."""
    if not name or name != Path(name).name or "/" in name or "\\" in name:
        raise ArtifactError(f"Invalid model name {name!r}.")
    return root / TASK_NAME / name


def save_artifact(
    name: str,
    pipeline: Pipeline,
    metrics: ProbabilityMetrics,
    metadata: dict[str, object],
    root: Path = DEFAULT_ARTIFACTS_ROOT,
) -> Path:
    """Write a fitted pipeline, metrics and metadata. Replaces any previous copy."""
    directory = artifact_dir(name, root)
    directory.mkdir(parents=True, exist_ok=True)

    metric_payload = metrics_as_dict(metrics)
    record = {
        "task": TASK_NAME,
        "model": name,
        **metadata,
        "metrics": metric_payload,
    }

    joblib.dump(pipeline, directory / MODEL_FILENAME)
    (directory / METRICS_FILENAME).write_text(
        json.dumps(metric_payload, indent=2) + "\n", encoding="utf-8"
    )
    (directory / METADATA_FILENAME).write_text(
        json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
    )
    logger.info("wrote artifact %s to %s", name, directory)
    return directory


def list_artifacts(root: Path = DEFAULT_ARTIFACTS_ROOT) -> list[str]:
    """Names of complete artifacts (pipeline + metrics + metadata)."""
    task_dir = root / TASK_NAME
    if not task_dir.is_dir():
        return []
    names: list[str] = []
    for path in sorted(task_dir.iterdir()):
        if (
            path.is_dir()
            and (path / MODEL_FILENAME).is_file()
            and (path / METRICS_FILENAME).is_file()
            and (path / METADATA_FILENAME).is_file()
        ):
            names.append(path.name)
    return names


def load_metadata(name: str, root: Path = DEFAULT_ARTIFACTS_ROOT) -> dict[str, object]:
    """Read ``metadata.json`` for a named artifact."""
    path = artifact_dir(name, root) / METADATA_FILENAME
    if not path.is_file():
        raise ArtifactError(f"No metadata for model {name!r} at {path}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ArtifactError(f"Metadata for {name!r} is not a JSON object.")
    return cast(dict[str, object], payload)


def load_metrics(name: str, root: Path = DEFAULT_ARTIFACTS_ROOT) -> dict[str, float | int]:
    """Read ``metrics.json`` for a named artifact."""
    path = artifact_dir(name, root) / METRICS_FILENAME
    if not path.is_file():
        raise ArtifactError(f"No metrics for model {name!r} at {path}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ArtifactError(f"Metrics for {name!r} are not a JSON object.")
    parsed: dict[str, float | int] = {}
    for key, value in payload.items():
        if not isinstance(value, (int, float)):
            raise ArtifactError(f"Metric {key!r} for {name!r} is not numeric.")
        parsed[str(key)] = int(value) if str(key) in INTEGER_METRIC_KEYS else float(value)
    return parsed


def load_pipeline(name: str, root: Path = DEFAULT_ARTIFACTS_ROOT) -> Pipeline:
    """Load a fitted sklearn pipeline from disk. Does not train."""
    path = artifact_dir(name, root) / MODEL_FILENAME
    if not path.is_file():
        raise ArtifactError(f"No pipeline for model {name!r} at {path}.")
    loaded = joblib.load(path)
    if not isinstance(loaded, Pipeline):
        raise ArtifactError(f"Artifact {name!r} is not an sklearn Pipeline.")
    return loaded


def describe_artifact(name: str, root: Path = DEFAULT_ARTIFACTS_ROOT) -> ArtifactInfo:
    """Load metadata and metrics without unpickling the pipeline."""
    metadata = load_metadata(name, root)
    metrics = load_metrics(name, root)
    features_raw = metadata.get("features", [])
    features = tuple(str(item) for item in features_raw) if isinstance(features_raw, list) else ()
    return ArtifactInfo(
        name=name,
        task=str(metadata.get("task", TASK_NAME)),
        features=features,
        training_samples=_as_int(metadata.get("training_samples", 0)),
        test_samples=_as_int(metadata.get("test_samples", 0)),
        validation_strategy=str(metadata.get("validation_strategy", "unknown")),
        metrics=metrics,
        metadata=metadata,
    )


def _as_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default
