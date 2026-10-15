"""Tests for the model-comparison report generator."""

import json

import pytest

from hinglish_mlops import compare
from hinglish_mlops.modeling import BASELINE_NAME
from hinglish_mlops.train import metrics_artifact_name

PAYLOAD = {
    "model": BASELINE_NAME,
    "version": "0.1.0",
    "metrics": {
        "train": {"accuracy": 0.9, "macro_f1": 0.9},
        "val": {"accuracy": 0.6, "macro_f1": 0.61},
        "test": {"accuracy": 0.66, "macro_f1": 0.665},
        "shae_eval": {"accuracy": 0.4, "macro_f1": 0.399},
    },
}


def test_render_includes_models_and_splits():
    md = compare.render({BASELINE_NAME: PAYLOAD})
    assert "# Model comparison" in md
    assert BASELINE_NAME in md
    assert "| baseline_tfidf_logreg | 0.900 | 0.610 | 0.665 | 0.399 |" in md


def test_build_writes_comparison_file(tmp_path):
    (tmp_path / metrics_artifact_name(BASELINE_NAME)).write_text(
        json.dumps(PAYLOAD), encoding="utf-8"
    )
    path = compare.build(tmp_path)
    assert path.name == compare.COMPARISON_FILE
    assert BASELINE_NAME in path.read_text(encoding="utf-8")


def test_build_without_metrics_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="train a model first"):
        compare.build(tmp_path)


def test_load_metrics_missing_returns_none(tmp_path):
    assert compare.load_metrics(BASELINE_NAME, tmp_path) is None
