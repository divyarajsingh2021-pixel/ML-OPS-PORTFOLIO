"""Integration tests for the training entrypoint."""

import json

import pandas as pd
import pytest

from hinglish_mlops import train
from hinglish_mlops.modeling import BASELINE_NAME, IMPROVED_NAME

TEXTS = [
    "bakwas bakwas",
    "bakwas ghatiya",
    "theek theek",
    "theek hai",
    "badhiya badhiya",
    "badhiya shandar",
]
LABELS = ["negative", "negative", "neutral", "neutral", "positive", "positive"]


def _frame(texts, labels):
    return pd.DataFrame(
        {
            "uid": [str(i) for i in range(len(texts))],
            "text": texts,
            "text_clean": texts,
            "label": labels,
        }
    )


def _write_processed(processed_dir):
    processed_dir.mkdir(parents=True, exist_ok=True)
    _frame(TEXTS, LABELS).to_csv(processed_dir / "sentimix_train.csv", index=False)
    _frame(TEXTS, LABELS).to_csv(processed_dir / "sentimix_val.csv", index=False)
    _frame(
        ["bakwas ghatiya", "theek hai", "badhiya shandar"],
        ["negative", "neutral", "positive"],
    ).to_csv(processed_dir / "sentimix_test.csv", index=False)
    _frame(["badhiya shandar"], ["positive"]).to_csv(processed_dir / "shae_eval.csv", index=False)


@pytest.mark.parametrize("model_name", [BASELINE_NAME, IMPROVED_NAME])
def test_train_model_separates_classes(model_name):
    pipeline, metrics = train.train_model(
        model_name, _frame(TEXTS, LABELS), _frame(TEXTS, LABELS), _frame(TEXTS, LABELS)
    )
    assert metrics["train"]["accuracy"] == 1.0
    assert set(metrics) == {"train", "val", "test"}


@pytest.mark.parametrize("model_name", [BASELINE_NAME, IMPROVED_NAME])
def test_run_writes_model_and_metrics(tmp_path, model_name):
    processed = tmp_path / "processed"
    _write_processed(processed)

    payload = train.run(model_name, processed, tmp_path / "models", tmp_path / "reports")

    assert (tmp_path / "models" / train.model_artifact_name(model_name)).exists()
    metrics_path = tmp_path / "reports" / train.metrics_artifact_name(model_name)
    assert metrics_path.exists()

    on_disk = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert on_disk["model"] == model_name
    assert on_disk["metrics"]["test"]["accuracy"] == 1.0
    assert "shae_eval" in payload["metrics"]


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        train.train_model(
            "nope", _frame(TEXTS, LABELS), _frame(TEXTS, LABELS), _frame(TEXTS, LABELS)
        )
