"""Train and evaluate the baseline sentiment model.

CLI::

    python -m hinglish_mlops.train

Fits TF-IDF (word) + logistic regression on the processed training split,
evaluates on train/val/test and the out-of-distribution eval set, then writes
the fitted pipeline to ``models/baseline.joblib`` and metrics to
``reports/baseline_metrics.json`` (both git-ignored except the JSON report,
which is committed).
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
from .modeling import build_baseline_pipeline, save_model

MODEL_NAME = "baseline_tfidf_logreg"
MODEL_FILE = "baseline.joblib"
METRICS_FILE = "baseline_metrics.json"


def train_baseline(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    eval_frame: pd.DataFrame | None = None,
    *,
    random_state: int = RANDOM_SEED,
) -> tuple[object, dict[str, dict[str, object]]]:
    """Fit the baseline and evaluate it on every provided split."""
    pipeline = build_baseline_pipeline(random_state=random_state)
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
    processed_dir: Path = PROCESSED_DIR,
    model_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, object]:
    """Load data, train the baseline and write model + metrics artifacts."""
    splits, eval_frame = load_all(processed_dir)
    pipeline, metrics = train_baseline(splits["train"], splits["val"], splits["test"], eval_frame)

    save_model(pipeline, model_dir / MODEL_FILE)
    payload: dict[str, object] = {
        "model": MODEL_NAME,
        "version": __version__,
        "metrics": metrics,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / METRICS_FILE).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the baseline sentiment model.")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args(argv)

    payload = run(args.processed_dir, args.model_dir, args.reports_dir)
    print(f"model: {payload['model']} v{payload['version']}")
    for name, info in payload["metrics"].items():
        print(f"{name:>10}: acc {info['accuracy']:.3f}  macro-F1 {info['macro_f1']:.3f}")
    print(f"Saved model to {args.model_dir / MODEL_FILE}")
    print(f"Saved metrics to {args.reports_dir / METRICS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
