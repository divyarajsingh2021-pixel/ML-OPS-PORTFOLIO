"""Train and evaluate a registered sentiment model.

CLI::

    python -m hinglish_mlops.train --model baseline_tfidf_logreg
    python -m hinglish_mlops.train --model improved_tfidf_svc

Fits the chosen model on the processed training split, evaluates on
train/val/test and the out-of-distribution eval set, then writes the fitted
pipeline to ``models/<model>.joblib`` (git-ignored) and metrics to
``reports/<model>_metrics.json`` (committed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import __version__
from .config import MODELS_DIR, PROCESSED_DIR, RANDOM_SEED, REPORTS_DIR
from .dataset import EVAL_NAME, load_all
from .evaluation import classification_metrics
from .modeling import BASELINE_NAME, MODEL_BUILDERS, build_model, save_model


def model_artifact_name(model_name: str) -> str:
    """Filename of the fitted model artifact for ``model_name``."""
    return f"{model_name}.joblib"


def metrics_artifact_name(model_name: str) -> str:
    """Filename of the metrics report for ``model_name``."""
    return f"{model_name}_metrics.json"


def train_model(
    model_name: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    eval_frame: pd.DataFrame | None = None,
    *,
    random_state: int = RANDOM_SEED,
) -> tuple[object, dict[str, dict[str, object]]]:
    """Fit a registered model and evaluate it on every provided split."""
    pipeline = build_model(model_name, random_state)
    pipeline.fit(train["text_clean"], train["label"])

    sets = {"train": train, "val": val, "test": test}
    if eval_frame is not None:
        sets[EVAL_NAME] = eval_frame

    metrics = {
        name: classification_metrics(frame["label"], pipeline.predict(frame["text_clean"]))
        for name, frame in sets.items()
    }
    return pipeline, metrics


def run(
    model_name: str = BASELINE_NAME,
    processed_dir: Path = PROCESSED_DIR,
    model_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
    *,
    track: bool = True,
    tracking_uri: str | None = None,
) -> dict[str, object]:
    """Load data, train a model and write model + metrics artifacts."""
    splits, eval_frame = load_all(processed_dir)
    pipeline, metrics = train_model(
        model_name, splits["train"], splits["val"], splits["test"], eval_frame
    )

    model_path = save_model(pipeline, model_dir / model_artifact_name(model_name))
    payload: dict[str, object] = {
        "model": model_name,
        "version": __version__,
        "metrics": metrics,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / metrics_artifact_name(model_name)).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    if track:
        payload["mlflow_run_id"] = _log_to_mlflow(
            model_name, metrics, pipeline, model_path, tracking_uri
        )
    return payload


def _log_to_mlflow(model_name, metrics, pipeline, model_path, tracking_uri):
    """Best-effort MLflow logging: report failures instead of breaking training."""
    from .tracking import log_run

    try:
        return log_run(
            model_name=model_name,
            metrics=metrics,
            pipeline=pipeline,
            artifact_paths=[model_path],
            tracking_uri=tracking_uri,
        )
    except Exception as exc:
        # Tracking must never fail a training run; surface the problem instead.
        print(f"[warn] MLflow tracking skipped: {exc}")
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a sentiment model.")
    parser.add_argument(
        "--model",
        default=BASELINE_NAME,
        choices=sorted(MODEL_BUILDERS),
        help="registered model to train",
    )
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--no-track", action="store_true", help="skip MLflow tracking")
    parser.add_argument("--tracking-uri", default=None, help="MLflow URI (default: local mlruns)")
    args = parser.parse_args(argv)

    payload = run(
        args.model,
        args.processed_dir,
        args.model_dir,
        args.reports_dir,
        track=not args.no_track,
        tracking_uri=args.tracking_uri,
    )
    print(f"model: {payload['model']} v{payload['version']}")
    for name, info in payload["metrics"].items():
        print(f"{name:>10}: acc {info['accuracy']:.3f}  macro-F1 {info['macro_f1']:.3f}")
    print(f"Saved model to {args.model_dir / model_artifact_name(args.model)}")
    print(f"Saved metrics to {args.reports_dir / metrics_artifact_name(args.model)}")
    if payload.get("mlflow_run_id"):
        print(f"MLflow run id: {payload['mlflow_run_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
