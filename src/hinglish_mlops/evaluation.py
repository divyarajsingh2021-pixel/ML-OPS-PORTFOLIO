"""Classification metrics used across training, evaluation and monitoring."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .config import LABELS


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: tuple[str, ...] = LABELS,
) -> dict[str, object]:
    """Return accuracy, macro/weighted F1, per-class scores and a confusion matrix."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), zero_division=0
    )
    per_class = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }
    total = int(support.sum())
    weighted_f1 = float((f1 * support).sum() / total) if total else 0.0
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1.mean()),
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(labels)).tolist(),
    }
