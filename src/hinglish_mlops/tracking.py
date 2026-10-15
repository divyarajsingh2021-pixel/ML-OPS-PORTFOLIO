"""Experiment tracking with MLflow.

Runs against a local SQLite database (``mlflow.db``, git-ignored) so everything
stays offline and dependency-free while avoiding MLflow's deprecated file-store
backend. Each training run logs the model name, key hyper-parameters, per-split
metrics and the fitted model artifact. Tracking is optional and must never break
training -- failures are reported loudly instead of being silently swallowed.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "hinglish-sentiment"
# Scalars only: MLflow params must be primitives, not estimators or tuples.
_PARAM_TYPES = (str, int, float, bool)


def _scalar_params(pipeline: object) -> dict[str, Any]:
    """Flatten a pipeline's hyper-parameters to scalar entries."""
    try:
        params = pipeline.get_params(deep=True)
    except AttributeError:
        return {}
    return {k: v for k, v in params.items() if isinstance(v, _PARAM_TYPES)}


def _scalar_metrics(metrics: dict[str, object]) -> dict[str, float]:
    return {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}


def log_run(
    *,
    model_name: str,
    metrics: dict[str, dict[str, object]],
    pipeline: object | None = None,
    artifact_paths: Sequence[Path] = (),
    tracking_uri: str | None = None,
    experiment_name: str = EXPERIMENT_NAME,
) -> str:
    """Log one training run and return its MLflow run id.

    ``metrics`` maps split name -> metrics dict (only scalars are logged).
    ``tracking_uri`` defaults to the project-local SQLite store.
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri or DEFAULT_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.set_tag("model", model_name)
        if pipeline is not None:
            mlflow.log_params(_scalar_params(pipeline))
        for split, split_metrics in metrics.items():
            mlflow.log_metrics(
                {f"{split}_{k}": v for k, v in _scalar_metrics(split_metrics).items()}
            )
        for path in artifact_paths:
            if Path(path).exists():
                mlflow.log_artifact(str(path))
        return run.info.run_id
