"""Tests for the classification metrics helper."""

from hinglish_mlops.evaluation import classification_metrics


def test_perfect_predictions():
    y = ["negative", "neutral", "positive", "negative"]
    metrics = classification_metrics(y, y)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert metrics["per_class"]["negative"]["support"] == 2


def test_mixed_predictions():
    y_true = ["negative", "negative", "positive", "positive"]
    y_pred = ["negative", "positive", "positive", "negative"]
    metrics = classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.5
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert metrics["n"] == 4


def test_missing_class_has_zero_support():
    metrics = classification_metrics(["neutral"], ["neutral"])
    assert metrics["per_class"]["positive"]["support"] == 0
    assert metrics["per_class"]["positive"]["f1"] == 0.0
