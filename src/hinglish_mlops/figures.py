"""Confusion-matrix figures for trained models.

Renders a per-class confusion matrix for a saved model on one data split and
writes it to ``reports/figures/`` (PNGs are git-ignored; regenerate them with
``inv figures``). Uses the Agg backend so it also works headless (CI, servers).

CLI::

    python -m hinglish_mlops.figures --model improved_tfidf_svc --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede pyplot so figures work without a display

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from .config import FIGURES_DIR, LABELS, MODELS_DIR, PROCESSED_DIR  # noqa: E402
from .dataset import load_split  # noqa: E402
from .modeling import BASELINE_NAME, load_model  # noqa: E402
from .train import model_artifact_name  # noqa: E402


def confusion_figure(y_true: pd.Series, y_pred: object, out_path: Path, title: str = "") -> Path:
    """Render a confusion matrix over the canonical label order and save it."""
    matrix = confusion_matrix(y_true, y_pred, labels=list(LABELS))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=list(LABELS),
        yticklabels=list(LABELS),
        ax=ax,
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(title or "Confusion matrix")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run(
    model_name: str = BASELINE_NAME,
    split: str = "test",
    model_dir: Path = MODELS_DIR,
    processed_dir: Path = PROCESSED_DIR,
    figures_dir: Path = FIGURES_DIR,
) -> Path:
    """Save the confusion-matrix figure for a trained model on ``split``."""
    pipeline = load_model(model_dir / model_artifact_name(model_name))
    frame = load_split(split, processed_dir)
    predictions = pipeline.predict(frame["text_clean"])
    out_path = figures_dir / f"{model_name}_confusion_matrix.png"
    return confusion_figure(
        frame["label"],
        predictions,
        out_path,
        title=f"{model_name} on {split} split",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render model evaluation figures.")
    parser.add_argument("--model", default=BASELINE_NAME, help="registered model name")
    parser.add_argument("--split", default="test", help="SentiMix split to evaluate on")
    parser.add_argument("--model-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args(argv)

    try:
        out_path = run(
            args.model,
            args.split,
            args.model_dir,
            args.processed_dir,
            args.figures_dir,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
