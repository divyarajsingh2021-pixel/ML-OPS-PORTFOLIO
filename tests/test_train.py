"""Integration tests for the baseline training entrypoint."""

import json

import pandas as pd

from hinglish_mlops import train

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


def test_train_baseline_separates_classes():
    pipeline, metrics = train.train_baseline(
        _frame(TEXTS, LABELS), _frame(TEXTS, LABELS), _frame(TEXTS, LABELS)
    )
    assert metrics["train"]["accuracy"] == 1.0
    assert set(metrics) == {"train", "val", "test"}


def test_run_writes_model_and_metrics(tmp_path):
    processed = tmp_path / "processed"
    _write_processed(processed)

    payload = train.run(processed, tmp_path / "models", tmp_path / "reports")

    assert (tmp_path / "models" / train.MODEL_FILE).exists()
    metrics_path = tmp_path / "reports" / train.METRICS_FILE
    assert metrics_path.exists()

    on_disk = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert on_disk["model"] == train.MODEL_NAME
    assert on_disk["metrics"]["test"]["accuracy"] == 1.0
    assert "shae_eval" in payload["metrics"]
