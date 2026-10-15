"""Tests for Evidently drift monitoring."""

import json

import numpy as np
import pandas as pd
import pytest

from hinglish_mlops import monitoring
from hinglish_mlops.monitoring import (
    REPORT_FILENAME,
    SUMMARY_FILENAME,
    TEXT_COLUMN,
    drop_constant_columns,
    features_from_frame,
    run_monitoring,
    text_features,
)


def _frame(n: int, mean_words: int, label_probs=None, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    texts = []
    for k in rng.integers(max(1, mean_words - 5), mean_words + 5, n):
        noise = rng.choice(["1", "!!", "OK", ""])
        texts.append(" ".join(["accha"] * int(k)) + (f" {noise}" if noise else ""))
    labels = rng.choice(
        ["negative", "neutral", "positive"], n, p=label_probs or [1 / 3, 1 / 3, 1 / 3]
    )
    return pd.DataFrame({TEXT_COLUMN: texts, "label": labels})


def test_text_features_known_values():
    features = text_features(pd.Series(["Bahut Accha 123!!!"])).loc[0]
    assert features["text_length"] == 18
    assert features["word_count"] == 3
    assert features["avg_word_length"] == pytest.approx(16 / 3)
    assert features["digit_share"] == pytest.approx(3 / 18)
    assert features["uppercase_share"] == pytest.approx(2 / 18)
    assert features["special_char_share"] == pytest.approx(3 / 18)


def test_text_features_handles_blank_text():
    features = text_features(pd.Series([""])).loc[0]
    assert features["text_length"] == 0
    assert features["word_count"] == 0
    assert features["avg_word_length"] == 0.0
    assert features["digit_share"] == 0.0


def test_features_from_frame_adds_labels():
    features = features_from_frame(_frame(10, 8, seed=1))
    assert "label" in features.columns
    assert len(features) == 10


def test_features_from_frame_requires_text_column():
    with pytest.raises(ValueError, match="text_clean"):
        features_from_frame(pd.DataFrame({"other": ["x"]}))


def test_run_monitoring_reports_no_drift_for_same_distribution(tmp_path):
    summary = run_monitoring(_frame(400, 12, seed=1), _frame(400, 12, seed=2), tmp_path)

    assert summary["dataset_drift"] is False
    assert set(summary["columns"]) >= {"text_length", "word_count", "label"}
    html = (tmp_path / REPORT_FILENAME).read_text(encoding="utf-8")
    assert len(html) > 1000
    on_disk = json.loads((tmp_path / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(summary))


def test_run_monitoring_detects_feature_drift(tmp_path):
    summary = run_monitoring(
        _frame(600, 10, seed=1), _frame(600, 120, seed=2), tmp_path, drift_share=0.2
    )
    assert summary["dataset_drift"] is True
    assert summary["columns"]["text_length"]["drift_detected"] is True
    assert summary["columns"]["word_count"]["drift_detected"] is True
    assert summary["drifted_column_count"] >= 2


def test_run_monitoring_detects_label_drift(tmp_path):
    summary = run_monitoring(
        _frame(1000, 12, seed=1),
        _frame(800, 12, label_probs=[0.8, 0.1, 0.1], seed=2),
        tmp_path,
    )

    assert summary["columns"]["label"]["drift_detected"] is True


def test_main_reports_missing_split(tmp_path, capsys):
    from hinglish_mlops.monitoring import main

    return_code = main(
        ["--reference", "ghost", "--current", "val", "--processed-dir", str(tmp_path)]
    )

    assert return_code == 2
    assert "error:" in capsys.readouterr().err


def test_drop_constant_columns_keeps_informative_ones():
    reference = pd.DataFrame({"flat": [1, 1, 1], "varying": [1, 2, 3]})
    current = pd.DataFrame({"flat": [1, 1, 1], "varying": [4, 5, 6]})
    ref_out, cur_out = drop_constant_columns(reference, current)
    assert list(ref_out.columns) == ["varying"]
    assert list(cur_out.columns) == ["varying"]


def test_drop_constant_columns_rejects_all_constant():
    with pytest.raises(ValueError, match="constant"):
        drop_constant_columns(pd.DataFrame({"a": [1, 1]}), pd.DataFrame({"a": [2, 2]}))


@pytest.mark.parametrize(
    ("method", "value", "threshold", "expected"),
    [
        ("K-S p_value", 0.001, 0.05, True),
        ("K-S p_value", 0.4, 0.05, False),
        ("Wasserstein distance (normed)", 0.5, 0.1, True),
        ("Jensen-Shannon distance", 0.0, 0.1, False),
        ("Wasserstein distance (normed)", 0.5, None, None),
    ],
)
def test_column_drift_direction(method, value, threshold, expected):
    result = monitoring._column_drift(
        {"config": {"method": method, "threshold": threshold}, "value": value}
    )
    assert result["drift_detected"] is expected
